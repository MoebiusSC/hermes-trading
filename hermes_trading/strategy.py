"""Strategy rules, shared by the live worker (loop.py) and the backtester (backtest.py), so a
backtest runs exactly the rules the worker trades.

strategy.yaml fields (the ones marked "since v2" are optional; missing means the original
behaviour, so an old file trades as it always did):
  entry.indicator      rsi
  entry.direction      long | short
  entry.threshold      RSI level: long enters below it, short above it
  entry.timeframe      candle size the RSI and ATR are computed on: 1m | 5m | 15m     (since v2, default 1m)
  exit_rsi             long exits when RSI rises to it; short when it falls to 100 - it  (since v2, default 70)
  trend_filter         off | 1h | 4h: only enter with the trend, close above (long) or
                       below (short) the EMA(50) of that timeframe                      (since v2, default off)
  stop_loss_pct        stop distance in % of the entry price, used when stop_atr_mult is 0
  stop_atr_mult        stop distance in ATR(14) multiples; 0 = use stop_loss_pct         (since v2, default 0)
  take_profit_r        target distance in multiples of the stop distance                (default 2)
  position_size_r      % of the account lost if the stop is hit
  max_hold_min         close a position after this many minutes; 0 = no limit          (since v2, default 0)

Indicators use closed candles only: the forming candle is dropped, so live and backtest agree.
"""
from __future__ import annotations

import time

import numpy as np

RSI_PERIOD = 14
ATR_PERIOD = 14
TREND_EMA = 50
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
ENTRY_TIMEFRAMES = ("1m", "5m", "15m")
TREND_FILTERS = ("off", "1h", "4h")
ENTRY_BARS = 100          # candles fetched for the entry timeframe (RSI/ATR warm-up)
TREND_BARS = TREND_EMA * 2 + 20

# Cost model for simulated fills: % per side. Stocks fill at Alpaca (commission-free, real spread).
DEFAULT_COSTS = {"crypto": {"fee_pct": 0.1, "slippage_pct": 0.02}, "stock": {"fee_pct": 0.0, "slippage_pct": 0.0}}


def params(strategy: dict) -> dict:
    """The strategy as flat numbers, with the original behaviour for fields it doesn't set."""
    entry = strategy["entry"]
    if entry.get("indicator", "rsi") != "rsi":
        raise ValueError(f"unsupported indicator {entry['indicator']!r} (only 'rsi')")
    if entry["direction"] not in ("long", "short"):
        raise ValueError(f"unsupported direction {entry['direction']!r}")
    p = {
        "direction": entry["direction"],
        "threshold": float(entry["threshold"]),
        "timeframe": str(entry.get("timeframe", "1m")),
        "exit_rsi": float(strategy.get("exit_rsi", 70)),
        "trend_filter": str(strategy.get("trend_filter", "off")),
        "stop_loss_pct": float(strategy["stop_loss_pct"]),
        "stop_atr_mult": float(strategy.get("stop_atr_mult", 0) or 0),
        "take_profit_r": float(strategy.get("take_profit_r", 2.0)),
        "position_size_r": float(strategy["position_size_r"]),
        "max_hold_min": float(strategy.get("max_hold_min", 0) or 0),
    }
    if p["timeframe"] not in ENTRY_TIMEFRAMES:
        raise ValueError(f"unsupported entry timeframe {p['timeframe']!r} ({', '.join(ENTRY_TIMEFRAMES)})")
    if p["trend_filter"] not in TREND_FILTERS:
        raise ValueError(f"unsupported trend filter {p['trend_filter']!r} ({', '.join(TREND_FILTERS)})")
    return p


def costs(goal: dict, stock: bool) -> tuple[float, float]:
    """(fee, slippage) as fractions per side, from goal.yaml `costs:` or the defaults."""
    kind = "stock" if stock else "crypto"
    c = {**DEFAULT_COSTS[kind], **((goal.get("costs") or {}).get(kind) or {})}
    return float(c["fee_pct"]) / 100, float(c["slippage_pct"]) / 100


# --- indicators (Wilder smoothing, as TradingView and the original loop) ---------------


def rsi_series(closes) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) < 2:
        return out
    delta = np.diff(c)
    gain, loss = np.clip(delta, 0, None), np.clip(-delta, 0, None)
    a = 1 / RSI_PERIOD
    g = l = 0.0
    for i in range(len(delta)):
        if i == 0:
            g, l = gain[0], loss[0]
        else:
            g, l = g + a * (gain[i] - g), l + a * (loss[i] - l)
        if i + 1 >= RSI_PERIOD:
            out[i + 1] = (50.0 if g == 0 else 100.0) if l == 0 else 100 - 100 / (1 + g / l)
    return out


def atr_series(high, low, close) -> np.ndarray:
    h, lo, c = (np.asarray(x, dtype=float) for x in (high, low, close))
    out = np.full(len(c), np.nan)
    if len(c) < 2:
        return out
    tr = np.maximum(h[1:] - lo[1:], np.maximum(abs(h[1:] - c[:-1]), abs(lo[1:] - c[:-1])))
    a, v = 1 / ATR_PERIOD, 0.0
    for i in range(len(tr)):
        v = tr[i] if i == 0 else v + a * (tr[i] - v)
        if i + 1 >= ATR_PERIOD:
            out[i + 1] = v
    return out


def ema_series(closes, n: int = TREND_EMA) -> np.ndarray:
    c = np.asarray(closes, dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) < n:
        return out
    k, v = 2 / (n + 1), float(np.mean(c[:n]))
    out[n - 1] = v
    for i in range(n, len(c)):
        v = v + k * (c[i] - v)
        out[i] = v
    return out


def closed(candles: dict, tf: str, now_ms: float | None = None) -> dict:
    """Drop the still-forming last candle. candles: {t, open, high, low, close} lists, t = bar start ms."""
    now_ms = time.time() * 1000 if now_ms is None else now_ms
    n = len(candles["t"])
    if n and candles["t"][-1] + TF_SECONDS[tf] * 1000 > now_ms:
        return {k: v[:-1] for k, v in candles.items()}
    return candles


# --- rules ---------------------------------------------------------------------------------


def trend_ok(direction: str, trend_closes) -> bool | None:
    """With the trend? None when there isn't enough history for the EMA."""
    ema = ema_series(trend_closes)
    if not len(ema) or np.isnan(ema[-1]):
        return None
    last = float(trend_closes[-1])
    return last > ema[-1] if direction == "long" else last < ema[-1]


def entry_fires(p: dict, rsi_value: float | None, trend: bool | None) -> bool:
    if rsi_value is None or np.isnan(rsi_value):
        return False
    if p["trend_filter"] != "off" and trend is not True:
        return False
    return rsi_value < p["threshold"] if p["direction"] == "long" else rsi_value > p["threshold"]


def stop_distance(p: dict, price: float, atr: float | None) -> float:
    if p["stop_atr_mult"] > 0 and atr is not None and not np.isnan(atr) and atr > 0:
        return p["stop_atr_mult"] * atr
    return price * p["stop_loss_pct"] / 100


def levels(p: dict, direction: str, fill: float, dist: float) -> tuple[float, float]:
    sign = 1 if direction == "long" else -1
    return fill - sign * dist, fill + sign * dist * p["take_profit_r"]


def size(p: dict, equity: float, price: float, dist: float) -> float:
    """Units such that hitting the stop loses position_size_r % of equity; never leveraged."""
    return min((equity * p["position_size_r"] / 100) / dist, equity / price)


def exit_reason(pos: dict, p: dict, last: float, rsi_value: float | None, now_ms: float) -> str | None:
    is_long = pos["direction"] == "long"
    if (last <= pos["stop"]) if is_long else (last >= pos["stop"]):
        return "stop_loss"
    if (last >= pos["target"]) if is_long else (last <= pos["target"]):
        return "take_profit"
    if rsi_value is not None and not np.isnan(rsi_value):
        if (rsi_value >= p["exit_rsi"]) if is_long else (rsi_value <= 100 - p["exit_rsi"]):
            return "rsi_exit"
    if p["max_hold_min"] > 0 and pos.get("opened_ms") and now_ms - pos["opened_ms"] >= p["max_hold_min"] * 60000:
        return "time_exit"
    return None


def fill(price: float, side: str, slippage: float) -> float:
    """Simulated fill: buys pay up, sells give up the slippage."""
    return price * (1 + slippage) if side == "buy" else price * (1 - slippage)
