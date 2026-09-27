"""Strategy rules, shared by the live worker (loop.py) and the backtester (backtest.py), so a
backtest runs exactly the rules the worker trades.

strategy.yaml fields (the ones marked "since vN" are optional; missing means the original
behaviour, so an old file trades as it always did):
  entry.indicator      rsi | ema_cross | tsmom | ma_regime                     (ema_cross v4, tsmom/ma_regime v5)
                       rsi: mean reversion, enter when RSI is stretched, exit when it comes back
                       The others are "state" signals: at every closed candle they say long, short
                       or flat, and the position follows (as entry.direction allows), exiting or
                       flipping when the state changes:
                       ema_cross: long while EMA(fast) > EMA(slow), short while below
                       tsmom (time-series momentum): long while the close is above the close
                         `lookback` candles ago, short while below
                       ma_regime: long while the close is above its simple moving average of `ma`
                         candles, short (or flat, for direction long) while below
  entry.direction      long | short | both (both: state signals only)
  entry.threshold      rsi: RSI level; long enters below it, short above it
  entry.fast/slow      ema_cross: EMA periods                                           (default 50 / 200)
  entry.lookback       tsmom: candles back to compare with                              (default 60)
  entry.ma             ma_regime: moving average length in candles                      (default 100)
  entry.target_vol     state signals: scale the position down when the annualised volatility of the
                       last 30 candles is above this (0.5 = 50%/year); 0 = off         (default 0)
  entry.timeframe      candle size the signal and ATR use: 1m | 5m | 15m | 1h | 4h | 1d  (since v2, default 1m)
  exit_rsi             rsi: long exits when RSI rises to it; short when it falls to 100 - it  (since v2, default 70)
  trend_filter         off | 1h | 4h: only enter with the trend, close above (long) or
                       below (short) the EMA(50) of that timeframe                      (since v2, default off)
  stop_loss_pct        stop distance in % of the entry price, used when stop_atr_mult is 0
  stop_atr_mult        stop distance in ATR(14) multiples; 0 = use stop_loss_pct         (since v2, default 0)
  take_profit_r        target distance in multiples of the stop distance; 0 = no target (default 2)
  position_size_r      % of the account lost if the stop is hit
  position_pct         fixed exposure: % of the account each position uses; 0 = size by
                       position_size_r and the stop distance                            (since v3, default 0)
  max_hold_min         close a position after this many minutes; 0 = no limit          (since v2, default 0)

Stops are never closer than goal.yaml costs.min_stop_x_costs times the round-trip cost (see
min_stop_frac): a stop inside the fees loses on every trade. After a stop, the same direction
isn't re-entered until the signal changes.

Indicators use closed candles only: the forming candle is dropped, so live and backtest agree.
"""
from __future__ import annotations

import time

import numpy as np

RSI_PERIOD = 14
ATR_PERIOD = 14
TREND_EMA = 50
TF_SECONDS = {"1m": 60, "5m": 300, "15m": 900, "1h": 3600, "4h": 14400, "1d": 86400}
ENTRY_TIMEFRAMES = ("1m", "5m", "15m", "1h", "4h", "1d")
INDICATORS = ("rsi", "ema_cross", "tsmom", "ma_regime")
STATE_INDICATORS = ("ema_cross", "tsmom", "ma_regime")  # long / short / flat at every candle
VOL_BARS = 30  # candles of realised volatility for entry.target_vol
DIRECTIONS = ("long", "short", "both")
TREND_FILTERS = ("off", "1h", "4h")
ENTRY_BARS = 100          # candles fetched for the entry timeframe (RSI/ATR warm-up)
TREND_BARS = TREND_EMA * 2 + 20

# Keep file-edited strategies inside the same safe envelope used by reflection.py.
# stop_atr_mult may be 0 to select the percentage stop fallback.
PARAM_BOUNDS = {
    "threshold": (5.0, 95.0),
    "exit_rsi": (55.0, 90.0),
    "stop_loss_pct": (0.2, 10.0),
    "stop_atr_mult": (0.0, 6.0),
    "take_profit_r": (0.0, 10.0),
    "position_size_r": (0.1, 2.0),
    "max_hold_min": (0.0, 2880.0),
    "position_pct": (0.0, 100.0),
    "fast": (2.0, 200.0),
    "slow": (5.0, 400.0),
    "lookback": (5.0, 365.0),
    "ma": (10.0, 400.0),
    "target_vol": (0.0, 3.0),
}

# Cost model for simulated fills: % per side. Stocks fill at Alpaca (commission-free, real spread).
DEFAULT_COSTS = {"crypto": {"fee_pct": 0.1, "slippage_pct": 0.02}, "stock": {"fee_pct": 0.0, "slippage_pct": 0.0}}
DEFAULT_MIN_STOP_X_COSTS = 3.0


def params(strategy: dict) -> dict:
    """The strategy as flat numbers, with the original behaviour for fields it doesn't set."""
    entry = strategy["entry"]
    indicator = str(entry.get("indicator", "rsi"))
    if indicator not in INDICATORS:
        raise ValueError(f"unsupported indicator {indicator!r} ({', '.join(INDICATORS)})")
    if entry["direction"] not in DIRECTIONS or (indicator not in STATE_INDICATORS and entry["direction"] == "both"):
        raise ValueError(f"unsupported direction {entry['direction']!r} for {indicator}")
    p = {
        "indicator": indicator,
        "direction": entry["direction"],
        "threshold": float(entry.get("threshold", 30)),
        "fast": float(entry.get("fast", 50)),
        "slow": float(entry.get("slow", 200)),
        "lookback": float(entry.get("lookback", 60)),
        "ma": float(entry.get("ma", 100)),
        "target_vol": float(entry.get("target_vol", 0) or 0),
        "timeframe": str(entry.get("timeframe", "1m")),
        "exit_rsi": float(strategy.get("exit_rsi", 70)),
        "trend_filter": str(strategy.get("trend_filter", "off")),
        "stop_loss_pct": float(strategy["stop_loss_pct"]),
        "stop_atr_mult": float(strategy.get("stop_atr_mult", 0) or 0),
        "take_profit_r": float(strategy.get("take_profit_r", 2.0)),
        "position_size_r": float(strategy["position_size_r"]),
        "max_hold_min": float(strategy.get("max_hold_min", 0) or 0),
        "position_pct": float(strategy.get("position_pct", 0) or 0),
    }
    if p["timeframe"] not in ENTRY_TIMEFRAMES:
        raise ValueError(f"unsupported entry timeframe {p['timeframe']!r} ({', '.join(ENTRY_TIMEFRAMES)})")
    if p["trend_filter"] not in TREND_FILTERS:
        raise ValueError(f"unsupported trend filter {p['trend_filter']!r} ({', '.join(TREND_FILTERS)})")
    for name, (lo, hi) in PARAM_BOUNDS.items():
        value = p[name]
        if not lo <= value <= hi:
            raise ValueError(f"strategy {name!r} must be between {lo:g} and {hi:g}, got {value:g}")
    if indicator == "ema_cross" and p["fast"] >= p["slow"]:
        raise ValueError(f"entry.fast ({p['fast']:g}) must be below entry.slow ({p['slow']:g})")
    return p


def bars_needed(p: dict) -> int:
    """Closed candles the signal needs (with room for an EMA to settle)."""
    need = {"ema_cross": int(p["slow"] * 3), "tsmom": int(p["lookback"]) + 5, "ma_regime": int(p["ma"]) + 5}.get(p["indicator"], 0)
    if p["target_vol"] > 0:
        need = max(need, VOL_BARS + 5)
    return max(ENTRY_BARS, need)


def _cost_spec(goal: dict, stock: bool, side: str) -> dict:
    """goal.yaml `costs.<kind>`: flat {fee_pct, slippage_pct}, or per side of the book
    {long: {...}, short: {..., funding: true}} (e.g. longs on spot, shorts on a perpetual)."""
    kind = "stock" if stock else "crypto"
    c = (goal.get("costs") or {}).get(kind) or {}
    if side in c and isinstance(c[side], dict):
        c = c[side]
    return {**DEFAULT_COSTS[kind], **{k: v for k, v in c.items() if not isinstance(v, dict)}}


def costs(goal: dict, stock: bool, side: str = "long") -> tuple[float, float]:
    """(fee, slippage) as fractions per side of a trade, for a long or a short position."""
    c = _cost_spec(goal, stock, side)
    return float(c["fee_pct"]) / 100, float(c["slippage_pct"]) / 100


def pays_funding(goal: dict, stock: bool, side: str) -> bool:
    """Whether positions on this side are perpetual futures that pay/receive the funding rate."""
    return bool(_cost_spec(goal, stock, side).get("funding", False))


def funding_venue(goal: dict) -> str:
    return str(((goal.get("costs") or {}).get("crypto") or {}).get("venue", "okx"))


def min_stop_frac(goal: dict, stock: bool) -> float:
    """Closest a stop may be, as a fraction of the price: a multiple of the round-trip cost."""
    fee, slippage = costs(goal, stock)
    k = float((goal.get("costs") or {}).get("min_stop_x_costs", DEFAULT_MIN_STOP_X_COSTS))
    return k * 2 * (fee + slippage)


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


def clip_wicks(candles: dict, max_frac: float) -> dict:
    """Bad prints: a high or low more than `max_frac` beyond the bar's open, close and the previous
    close is cut back to that range (SPY's 2 Feb 2026 daily bar has a low of 68 with SPY near 690).
    Such wicks would trigger simulated stops that never happened."""
    o, h, lo, c = (np.asarray(candles[k], dtype=float) for k in ("open", "high", "low", "close"))
    if len(c) < 2:
        return candles
    prev = np.concatenate([[c[0]], c[:-1]])
    top, bottom = np.maximum.reduce([o, c, prev]), np.minimum.reduce([o, c, prev])
    h = np.where(h > top * (1 + max_frac), np.maximum(o, c), h)
    lo = np.where(lo < bottom * (1 - max_frac), np.minimum(o, c), lo)
    return {**candles, "high": h.tolist(), "low": lo.tolist()}


STOCK_MAX_WICK = {"1d": 0.2}  # beyond the bar's open, close and previous close; intraday: 8%


def stock_max_wick(tf: str) -> float:
    return STOCK_MAX_WICK.get(tf, 0.08)


# --- rules ---------------------------------------------------------------------------------


def trend_ok(direction: str, trend_closes) -> bool | None:
    """With the trend? None when there isn't enough history for the EMA."""
    ema = ema_series(trend_closes)
    if not len(ema) or np.isnan(ema[-1]):
        return None
    last = float(trend_closes[-1])
    return last > ema[-1] if direction == "long" else last < ema[-1]


def cross_state(closes, fast: float, slow: float) -> int | None:
    """+1 while EMA(fast) is above EMA(slow), -1 while below; None without enough history."""
    f, s = ema_series(closes, int(fast)), ema_series(closes, int(slow))
    if not len(s) or np.isnan(s[-1]) or np.isnan(f[-1]):
        return None
    return 1 if f[-1] > s[-1] else -1


def cross_states(closes, fast: float, slow: float) -> np.ndarray:
    """cross_state at every bar (0 where there isn't enough history), for the backtester."""
    f, s = ema_series(closes, int(fast)), ema_series(closes, int(slow))
    out = np.zeros(len(s))
    ok = ~np.isnan(s) & ~np.isnan(f)
    out[ok] = np.where(f[ok] > s[ok], 1, -1)
    return out


def signal_states(p: dict, closes) -> np.ndarray:
    """State signal (+1 long, -1 short, 0 not enough history) at every candle, for the backtester."""
    c = np.asarray(closes, dtype=float)
    if p["indicator"] == "ema_cross":
        return cross_states(c, p["fast"], p["slow"])
    out = np.zeros(len(c))
    if p["indicator"] == "tsmom":
        n = int(p["lookback"])
        if len(c) > n:
            out[n:] = np.sign(c[n:] - c[:-n])
    elif p["indicator"] == "ma_regime":
        n = int(p["ma"])
        if len(c) >= n:
            sma = np.convolve(c, np.ones(n) / n, mode="valid")  # sma[k] covers c[k .. k+n-1]
            out[n - 1:] = np.where(c[n - 1:] > sma, 1, -1)
    return out


def signal_state(p: dict, closes) -> int | None:
    """The state signal now (see signal_states); None without enough history."""
    states = signal_states(p, closes)
    return int(states[-1]) if len(states) and states[-1] != 0 else None


def realized_vol_series(closes, tf: str, n: int = VOL_BARS, t=None) -> np.ndarray:
    """Annualised volatility of the last n candle returns, at every candle (NaN before). With the
    candle times `t`, candles per year are counted from them: a stock trades ~252 days a year, not 365."""
    c = np.asarray(closes, dtype=float)
    out = np.full(len(c), np.nan)
    if len(c) <= n:
        return out
    r = np.diff(c) / c[:-1]
    per_year = 365 * 86400 / TF_SECONDS[tf]
    if t is not None and len(t) > 1 and t[-1] > t[0]:
        per_year = min(per_year, (len(t) - 1) / ((t[-1] - t[0]) / (365 * 86400000)))
    for i in range(n, len(c)):
        out[i] = float(np.std(r[i - n:i], ddof=1)) * np.sqrt(per_year)
    return out


def vol_scale(p: dict, vol: float | None) -> float:
    """entry.target_vol: the share of the normal size to take at this volatility (never above 1)."""
    if p["target_vol"] <= 0 or vol is None or not np.isfinite(vol) or vol <= 0:
        return 1.0
    return min(1.0, p["target_vol"] / vol)


def target_direction(p: dict, state: int | float | None) -> str | None:
    """State signals: the side the strategy wants to be on now, or None to be flat."""
    if not state:
        return None
    side = "long" if state > 0 else "short"
    return side if p["direction"] in ("both", side) else None


def entry_fires(p: dict, rsi_value: float | None, trend: bool | None) -> bool:
    if rsi_value is None or np.isnan(rsi_value):
        return False
    if p["trend_filter"] != "off" and trend is not True:
        return False
    return rsi_value < p["threshold"] if p["direction"] == "long" else rsi_value > p["threshold"]


def stop_distance(p: dict, price: float, atr: float | None, floor_frac: float = 0.0) -> float:
    if p["stop_atr_mult"] > 0 and atr is not None and not np.isnan(atr) and atr > 0:
        dist = p["stop_atr_mult"] * atr
    else:
        dist = price * p["stop_loss_pct"] / 100
    return max(dist, price * floor_frac)


def levels(p: dict, direction: str, fill: float, dist: float) -> tuple[float, float]:
    sign = 1 if direction == "long" else -1
    target = fill + sign * dist * p["take_profit_r"] if p["take_profit_r"] > 0 else sign * float("inf")
    return fill - sign * dist, target


def size(p: dict, equity: float, price: float, dist: float, scale: float = 1.0) -> float:
    """Units for a new position, never leveraged: position_pct % of equity when set, otherwise
    such that hitting the stop loses position_size_r % of equity; times `scale` (vol_scale)."""
    if equity <= 0:
        return 0.0
    if price <= 0:
        raise ValueError(f"price must be positive, got {price:g}")
    if dist <= 0:
        raise ValueError(f"stop distance must be positive, got {dist:g}")
    scale = min(1.0, max(0.0, scale))
    if p.get("position_pct", 0) > 0:
        return equity * p["position_pct"] / 100 / price * scale
    return min((equity * p["position_size_r"] / 100) / dist, equity / price) * scale


def exit_reason(pos: dict, p: dict, last: float, rsi_value: float | None, now_ms: float,
                target: str | None = None) -> str | None:
    """Why to close now. `target` is the side a state signal wants now (target_direction)."""
    is_long = pos["direction"] == "long"
    if (last <= pos["stop"]) if is_long else (last >= pos["stop"]):
        return "stop_loss"
    if pos.get("target") is not None and ((last >= pos["target"]) if is_long else (last <= pos["target"])):
        return "take_profit"
    if p["indicator"] in STATE_INDICATORS:
        if target != pos["direction"]:
            return "signal_exit"
    elif rsi_value is not None and not np.isnan(rsi_value):
        if (rsi_value >= p["exit_rsi"]) if is_long else (rsi_value <= 100 - p["exit_rsi"]):
            return "rsi_exit"
    if p["max_hold_min"] > 0 and pos.get("opened_ms") and now_ms - pos["opened_ms"] >= p["max_hold_min"] * 60000:
        return "time_exit"
    return None


def fill(price: float, side: str, slippage: float) -> float:
    """Simulated fill: buys pay up, sells give up the slippage."""
    return price * (1 + slippage) if side == "buy" else price * (1 - slippage)
