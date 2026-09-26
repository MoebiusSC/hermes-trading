"""Backtest a strategy on historical candles with the worker's own rules (strategy.py) and costs.

  python -m hermes_trading.backtest --asset BTC/USDT                     current strategy, last 30 days
  python -m hermes_trading.backtest --asset BTC/USDT --days 60 --set exit_rsi=60 --set entry.timeframe=5m
  python -m hermes_trading.backtest --all --set trend_filter=1h          every asset in goal.yaml

The period is split walk-forward: the first 70% is "in sample", the last 30% "out of sample".
A change that only looks good in sample is probably fitted to noise; the reflection cycle only
accepts changes that also hold out of sample.

Simulation, per bar of the strategy's timeframe:
  • signals are read at the bar's close (RSI, ATR, trend filter from closed candles only);
  • entries and RSI/time exits fill at the next bar's open, with slippage;
  • stops and targets fill inside the bar they're touched (at the level, or the open if it gapped
    through); if one bar touches both, the stop is assumed first;
  • fees are charged on both sides.
History is cached under cache/candles/ and topped up on each run.
"""
from __future__ import annotations

import argparse
import copy
import datetime as dt
import json
import sys
import time

import ccxt
import httpx
import numpy as np

from . import config
from . import strategy as rules
from .score import metrics, score
from .storage import atomic_write, load_yaml

CACHE_DIR = config.ROOT / "cache" / "candles"
DEFAULT_DAYS = 30
OOS_FRACTION = 0.3
# Exchanges with deep public history (Kraken only serves the last 720 candles)
HISTORY_EXCHANGES = ("binance", "okx", "binanceus", "kraken")
_clients: dict = {}


# --- history ---------------------------------------------------------------------------------


def _fetch_crypto(asset: str, tf: str, since_ms: int) -> dict:
    errors = []
    for exchange_id in HISTORY_EXCHANGES:
        try:
            client = _clients.setdefault(exchange_id, getattr(ccxt, exchange_id)({"enableRateLimit": True}))
            rows, cursor, step = [], since_ms, rules.TF_SECONDS[tf] * 1000
            while True:
                page = client.fetch_ohlcv(asset, timeframe=tf, since=cursor, limit=1000)
                page = [r for r in page if r[0] >= cursor]
                if not page:
                    break
                rows += page
                cursor = page[-1][0] + step
                if cursor > time.time() * 1000 or len(page) < 50:
                    break
        except Exception as e:  # geo-block, symbol not listed, network
            errors.append(f"{exchange_id}: {type(e).__name__}"[:80])
            continue
        if rows:
            return {"source": exchange_id, **_columns(rows)}
        errors.append(f"{exchange_id}: no data")
    raise RuntimeError(f"no history for {asset} {tf} — " + "; ".join(errors))


def _columns(rows: list) -> dict:
    return {"t": [int(r[0]) for r in rows], "open": [float(r[1]) for r in rows], "high": [float(r[2]) for r in rows],
            "low": [float(r[3]) for r in rows], "close": [float(r[4]) for r in rows]}


_ALPACA_TF = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "1h": "1Hour", "4h": "4Hour", "1d": "1Day"}


def _fetch_stock(symbol: str, tf: str, since_ms: int) -> dict:
    key, secret = config.env("ALPACA_API_KEY"), config.env("ALPACA_API_SECRET")
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_API_SECRET not set")
    rows, token = [], None
    start = dt.datetime.fromtimestamp(since_ms / 1000, dt.timezone.utc).isoformat()
    while True:
        params = {"timeframe": _ALPACA_TF[tf], "limit": 10000, "feed": config.env("ALPACA_DATA_FEED", "iex"), "start": start}
        if token:
            params["page_token"] = token
        r = httpx.get(f"https://data.alpaca.markets/v2/stocks/{symbol}/bars", params=params, timeout=30,
                      headers={"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret})
        r.raise_for_status()
        data = r.json()
        for b in data.get("bars") or []:
            t = int(dt.datetime.fromisoformat(b["t"].replace("Z", "+00:00")).timestamp() * 1000)
            rows.append([t, b["o"], b["h"], b["l"], b["c"]])
        token = data.get("next_page_token")
        if not token:
            break
    return {"source": "alpaca", **_columns(rows)}


def history(asset: str, tf: str, days: float) -> dict:
    """Closed candles covering the last `days`, from the disk cache topped up with new bars."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    path = CACHE_DIR / f"{config.asset_slug(asset)}_{tf}.json"
    now = time.time() * 1000
    want_from = now - days * 86400000
    cached = json.loads(path.read_text()) if path.exists() else None
    if cached and cached["t"] and cached["t"][0] <= want_from + rules.TF_SECONDS[tf] * 1000 * 2:
        since = cached["t"][-1] + rules.TF_SECONDS[tf] * 1000
        fresh = (_fetch_stock if config.is_stock(asset) else _fetch_crypto)(asset, tf, int(since)) if since < now else None
        data = cached
        if fresh and fresh["t"]:
            data = {k: cached[k] + fresh[k] for k in ("t", "open", "high", "low", "close")}
            data["source"] = cached.get("source")
    else:
        data = (_fetch_stock if config.is_stock(asset) else _fetch_crypto)(asset, tf, int(want_from))
    # keep at most twice the window on disk
    keep = [i for i, t in enumerate(data["t"]) if t >= now - 2 * days * 86400000]
    data = {k: ([data[k][i] for i in keep] if isinstance(data[k], list) else data[k]) for k in data}
    atomic_write(path, json.dumps(data))
    data = rules.closed({k: data[k] for k in ("t", "open", "high", "low", "close")}, tf, now) | {"source": data.get("source")}
    keep = [i for i, t in enumerate(data["t"]) if t >= want_from]
    return {k: ([data[k][i] for i in keep] if isinstance(data[k], list) else data[k]) for k in data}


# --- simulation ------------------------------------------------------------------------------


def _iso(ms: float) -> str:
    return dt.datetime.fromtimestamp(ms / 1000, dt.timezone.utc).isoformat(timespec="seconds")


def simulate(strategy: dict, entry: dict, trend: dict | None, start_equity: float, fee: float, slippage: float) -> dict:
    p = rules.params(strategy)
    tfe = rules.TF_SECONDS[p["timeframe"]] * 1000
    t, o, h, lo, c = (np.asarray(entry[k], dtype=float) for k in ("t", "open", "high", "low", "close"))
    n = len(t)
    rsi = rules.rsi_series(c)
    atr = rules.atr_series(h, lo, c)
    # trend at each entry bar's close: the last trend bar that had closed by then
    trend_at = [None] * n
    if p["trend_filter"] != "off" and trend and trend["t"]:
        ttf = rules.TF_SECONDS[p["trend_filter"]] * 1000
        tt, tc = np.asarray(trend["t"], dtype=float), np.asarray(trend["close"], dtype=float)
        ema = rules.ema_series(tc)
        j = -1
        for i in range(n):
            close_time = t[i] + tfe
            while j + 1 < len(tt) and tt[j + 1] + ttf <= close_time:
                j += 1
            if j >= 0 and not np.isnan(ema[j]):
                trend_at[i] = bool(tc[j] > ema[j]) if p["direction"] == "long" else bool(tc[j] < ema[j])

    equity, pos, pending, trades, in_market = start_equity, None, None, [], 0
    equity_curve = [{"ts": _iso(t[warmup - 1] if warmup < n else t[0]), "equity": float(start_equity)}]
    sign = 1 if p["direction"] == "long" else -1
    buy_side, sell_side = ("buy", "sell") if sign > 0 else ("sell", "buy")

    def close_at(price: float, i_ms: float, reason: str) -> None:
        nonlocal equity, pos
        gross = sign * pos["qty"] * (price - pos["entry_price"])
        fees = pos["fees"] + pos["qty"] * price * fee
        pnl = gross - fees
        trades.append({"opened_at": _iso(pos["opened_ms"]), "closed_at": _iso(i_ms), "direction": p["direction"],
                       "entry_price": pos["entry_price"], "exit_price": price, "exit_reason": reason,
                       "gross_pnl": gross, "fees": fees, "pnl": pnl, "pnl_pct": pnl / equity})
        equity += pnl
        pos = None

    warmup = max(rules.RSI_PERIOD, rules.ATR_PERIOD) + 2
    for i in range(warmup, n):
        # 1) orders decided at the previous close fill at this bar's open
        if pending == "exit" and pos:
            close_at(rules.fill(o[i], sell_side, slippage), t[i], pos.pop("exit_reason"))
        elif pending == "entry" and not pos:
            fill_price = rules.fill(o[i], buy_side, slippage)
            dist = rules.stop_distance(p, fill_price, atr[i - 1])
            qty = rules.size(p, equity, fill_price, dist)
            stop, target = rules.levels(p, p["direction"], fill_price, dist)
            pos = {"direction": p["direction"], "entry_price": fill_price, "qty": qty, "stop": stop, "target": target,
                   "opened_ms": t[i], "fees": qty * fill_price * fee}
        pending = None
        # 2) stops and targets inside this bar
        if pos:
            in_market += 1
            hit_stop = lo[i] <= pos["stop"] if sign > 0 else h[i] >= pos["stop"]
            hit_target = h[i] >= pos["target"] if sign > 0 else lo[i] <= pos["target"]
            if hit_stop:
                level = min(o[i], pos["stop"]) if sign > 0 else max(o[i], pos["stop"])
                close_at(rules.fill(level, sell_side, slippage), t[i] + tfe, "stop_loss")
            elif hit_target:
                level = max(o[i], pos["target"]) if sign > 0 else min(o[i], pos["target"])
                close_at(rules.fill(level, sell_side, slippage), t[i] + tfe, "take_profit")
        # Mark the portfolio to market after fills/stops for drawdown and time-series Sharpe.
        mark_price = c[i]
        marked = equity
        if pos:
            gross = sign * pos["qty"] * (mark_price - pos["entry_price"])
            estimated_exit_fee = pos["qty"] * mark_price * fee
            marked += gross - pos["fees"] - estimated_exit_fee
        equity_curve.append({"ts": _iso(t[i] + tfe), "equity": float(marked)})

        # 3) decisions at this bar's close
        if i == n - 1:
            break
        if pos:
            reason = rules.exit_reason({**pos, "stop": -np.inf * sign, "target": np.inf * sign}, p, c[i], rsi[i], t[i] + tfe)
            if reason:
                pos["exit_reason"] = reason
                pending = "exit"
        elif rules.entry_fires(p, rsi[i], trend_at[i]):
            pending = "entry"
    if pos:  # mark an open position to the last close, as a trade, so its loss or gain counts
        close_at(rules.fill(c[-1], sell_side, slippage), t[-1] + tfe, "end_of_test")
    return {"trades": trades, "bars": n, "in_market_pct": in_market / max(1, n - warmup) * 100,
            "final_equity": equity, "start_equity": start_equity, "equity_curve": equity_curve}


def summarize(trades: list[dict], goal: dict, equity_curve: list[dict] | None = None) -> dict:
    wins = [x["pnl"] for x in trades if x["pnl"] > 0]
    losses = [-x["pnl"] for x in trades if x["pnl"] < 0]
    m = metrics(trades, equity_curve)
    return {
        "n": len(trades),
        "score": score(trades, goal),
        "return_pct": round(m["realised_return"] * 100, 4),
        "max_drawdown_pct": round(m["max_drawdown"] * 100, 4),
        "sharpe": m["sharpe"],
        "win_rate": m["win_rate"],
        "profit_factor": round(sum(wins) / sum(losses), 3) if losses else (None if not wins else 99.0),
        "fees": round(sum(x["fees"] for x in trades), 2),
        "exits": {r: sum(1 for x in trades if x["exit_reason"] == r) for r in sorted({x["exit_reason"] for x in trades})},
    }


def run(asset: str, strategy: dict, goal: dict, days: float = DEFAULT_DAYS) -> dict:
    """Walk-forward backtest: whole period, in sample (first 70%) and out of sample (last 30%)."""
    p = rules.params(strategy)
    entry = history(asset, p["timeframe"], days)
    if len(entry["t"]) < 100:
        raise RuntimeError(f"only {len(entry['t'])} {p['timeframe']} candles of history for {asset}")
    trend = None
    if p["trend_filter"] != "off":
        warm = rules.TREND_BARS * rules.TF_SECONDS[p["trend_filter"]] / 86400
        trend = history(asset, p["trend_filter"], days + warm)
    fee, slippage = rules.costs(goal, config.is_stock(asset))
    sim = simulate(strategy, entry, trend, config.start_equity(asset, goal), fee, slippage)
    split_ms = entry["t"][0] + (entry["t"][-1] - entry["t"][0]) * (1 - OOS_FRACTION)
    split = _iso(split_ms)
    first, last = entry["close"][0], entry["close"][-1]
    all_trades = sim["trades"]
    in_trades = [x for x in all_trades if x["opened_at"] < split]
    oos_trades = [x for x in all_trades if x["opened_at"] >= split]
    curve = sim["equity_curve"]
    in_curve = [p for p in curve if p["ts"] < split]
    oos_curve = [p for p in curve if p["ts"] >= split]
    return {
        "asset": asset,
        "from": _iso(entry["t"][0]), "to": _iso(entry["t"][-1]), "split": split, "days": days,
        "timeframe": p["timeframe"], "source": entry.get("source"), "fee_pct": fee * 100, "slippage_pct": slippage * 100,
        "in_market_pct": round(sim["in_market_pct"], 1),
        "buy_hold_pct": round((last / first - 1) * 100, 3),
        "all": summarize(all_trades, goal, curve),
        "in_sample": summarize(in_trades, goal, in_curve),
        "out_of_sample": summarize(oos_trades, goal, oos_curve),
    }


def with_changes(strategy: dict, changes: dict) -> dict:
    """A copy of the strategy with dotted-key changes (entry.threshold=25, exit_rsi=60…)."""
    out = copy.deepcopy(strategy)
    for key, value in changes.items():
        *parents, leaf = key.split(".")
        d = out
        for part in parents:
            d = d.setdefault(part, {})
        d[leaf] = value
    return out


def _parse_set(items: list[str]) -> dict:
    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            out[key.strip()] = json.loads(raw)
        except json.JSONDecodeError:
            out[key.strip()] = raw.strip()
    return out


def _line(label: str, s: dict) -> str:
    pf = "—" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
    return (f"  {label:<14} n={s['n']:<4} return={s['return_pct']:+.2f}%  dd={s['max_drawdown_pct']:.2f}%  "
            f"pf={pf:<5} win={s['win_rate'] * 100:.0f}%  score={s['score']:+.3f}  fees=${s['fees']:.0f}")


def main(argv: list[str] | None = None) -> int:
    config.load_env()
    ap = argparse.ArgumentParser(description="Backtest strategies on historical candles.")
    who = ap.add_mutually_exclusive_group(required=True)
    who.add_argument("--asset")
    who.add_argument("--all", action="store_true")
    ap.add_argument("--days", type=float, default=DEFAULT_DAYS)
    ap.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a strategy field")
    ap.add_argument("--json", action="store_true")
    args = ap.parse_args(argv)
    goal = load_yaml(config.GOAL_FILE)
    assets = config.goal_assets(goal) if args.all else [args.asset]
    changes = _parse_set(args.set)
    results = []
    for asset in assets:
        path = config.asset_paths(asset).strategy
        base = load_yaml(path) if path.exists() else load_yaml(config.STRATEGY_TEMPLATE)
        try:
            r = run(asset, with_changes(base, changes), goal, args.days)
        except Exception as e:
            print(f"{asset}: FAILED — {type(e).__name__}: {e}", flush=True)
            continue
        results.append(r)
        if not args.json:
            print(f"{asset} · {r['timeframe']} · {r['from'][:16]} → {r['to'][:16]} · {r['source']} · "
                  f"costs {r['fee_pct']:.2f}%+{r['slippage_pct']:.2f}%/side · buy&hold {r['buy_hold_pct']:+.2f}% · in market {r['in_market_pct']:.0f}%")
            for label in ("all", "in_sample", "out_of_sample"):
                print(_line(label, r[label]))
            print(f"  exits: {r['all']['exits']}", flush=True)
    if args.json:
        print(json.dumps(results, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
