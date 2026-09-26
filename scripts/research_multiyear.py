"""Multi-year research: the live-engine EMA cross vs five strategies from the literature, 12 crypto pairs.

  python scripts/research_multiyear.py out.json

Everything is measured the same way: an equal-weight portfolio of the 12 pairs (each pair's strategy
return averaged daily), per year (Y1, Y2, Y3 = the last three 365-day windows), with costs.
Costs: spot 0.1% fee + 0.02% slippage per side; shorts are perpetual futures at 0.05% + 0.02% and pay
or receive the real Binance funding rate (cache/funding, from scripts/fetch_history.py).
"""
import datetime as dt
import itertools
import json
import math
import sys

import numpy as np
import pandas as pd

from hermes_trading import backtest as b, config
from hermes_trading import strategy as rules
from hermes_trading.storage import load_yaml

goal = load_yaml(config.GOAL_FILE)
ASSETS = [a for a in config.goal_assets(goal) if not config.is_stock(a)]
SPOT = (0.001, 0.0002)
PERP = (0.0005, 0.0002)
DAY = 86400000
END = None


def load(asset, tf):
    d = json.load(open(config.ROOT / "cache" / "candles" / f"{config.asset_slug(asset)}_{tf}.json"))
    return {k: np.asarray(d[k], dtype=float) for k in ("t", "open", "high", "low", "close")}


def load_funding(asset):
    path = config.ROOT / "cache" / "funding" / f"{config.asset_slug(asset)}.json"
    return np.asarray(json.load(open(path)), dtype=float) if path.exists() else np.zeros((0, 2))


D = {a: load(a, "1d") for a in ASSETS}
END = max(D[a]["t"][-1] for a in ASSETS) + DAY
YEARS = [(END - (3 - k) * 365 * DAY, END - (2 - k) * 365 * DAY) for k in range(3)]  # Y1, Y2, Y3
DATES = np.arange(YEARS[0][0] - 400 * DAY, END, DAY)  # daily grid (bar start), with warm-up
F = {a: load_funding(a) for a in ASSETS}


def daily_funding(asset):
    """Sum of the 8h funding rates paid by longs during each day of DATES (shorts receive it)."""
    f = F[asset]
    out = np.zeros(len(DATES))
    if len(f):
        idx = ((f[:, 0] - DATES[0]) // DAY).astype(int)
        ok = (idx >= 0) & (idx < len(DATES))
        np.add.at(out, idx[ok], f[ok, 1])
    return out


FUND = {a: daily_funding(a) for a in ASSETS}


def on_grid(asset):
    """Daily closes on DATES (NaN before listing)."""
    d = D[asset]
    close = np.full(len(DATES), np.nan)
    idx = ((d["t"] - DATES[0]) // DAY).astype(int)
    ok = (idx >= 0) & (idx < len(DATES))
    close[idx[ok]] = d["close"][ok]
    return close


CLOSE = {a: on_grid(a) for a in ASSETS}


def weights_to_returns(asset, w):
    """Daily strategy return from target weights decided at each close (applied next day),
    with trading costs on weight changes and funding on the short side."""
    c = CLOSE[asset]
    r = np.zeros(len(c))
    r[1:] = c[1:] / c[:-1] - 1
    r[np.isnan(r)] = 0
    held = np.zeros(len(c))
    held[1:] = w[:-1]  # decided at yesterday's close
    held[np.isnan(held)] = 0
    turn = np.abs(np.diff(np.concatenate([[0], held])))
    cost = turn * np.where(held < 0, sum(PERP), sum(SPOT))
    fund = np.where(held < 0, -held * FUND[asset], 0)  # a short receives positive funding
    return held * r - cost + fund


def ann_stats(daily):
    """Return per year window: total return %, annualised vol %, Sharpe, max drawdown %."""
    out = []
    for lo, hi in YEARS:
        m = (DATES >= lo) & (DATES < hi)
        x = daily[m]
        eq = np.cumprod(1 + x)
        dd = float(np.max(1 - eq / np.maximum.accumulate(eq))) * 100
        vol = float(np.std(x, ddof=1) * math.sqrt(365)) * 100
        sharpe = float(np.mean(x) / np.std(x, ddof=1) * math.sqrt(365)) if np.std(x) > 0 else 0
        out.append({"ret": float(eq[-1] - 1) * 100, "vol": vol, "sharpe": sharpe, "dd": dd})
    return out


def portfolio(per_asset_daily):
    """Equal weight across the pairs that are listed on each day."""
    m = np.vstack([per_asset_daily[a] for a in ASSETS])
    listed = np.vstack([~np.isnan(CLOSE[a]) for a in ASSETS])
    return np.where(listed.sum(0) > 0, (m * listed).sum(0) / np.maximum(listed.sum(0), 1), 0)


results = {}


def record(name, params, per_asset):
    results.setdefault(name, []).append({"params": params, "years": ann_stats(portfolio(per_asset))})


# --- benchmark: buy & hold (equal weight, daily rebalanced) ------------------------------------
record("buy_hold", {}, {a: weights_to_returns(a, np.where(np.isnan(CLOSE[a]), 0, 1.0)) for a in ASSETS})


# --- 0. the live engine's EMA cross (backtest.simulate), resampled to daily ---------------------
def engine_daily(asset, strategy):
    tf = strategy["entry"]["timeframe"]
    e = load(asset, tf)
    sim = b.simulate(strategy, {k: list(v) for k, v in e.items()}, None, 10000, *SPOT, rules.min_stop_frac(goal, False))
    t = np.asarray([dt.datetime.fromisoformat(q["ts"]).timestamp() * 1000 for q in sim["equity_curve"]])
    v = np.asarray([q["equity"] for q in sim["equity_curve"]])
    eq = np.interp(DATES + DAY, t, v, left=v[0])
    out = np.zeros(len(DATES))
    out[1:] = eq[1:] / eq[:-1] - 1
    return out


for tf, (fa, sl), direction, stop in itertools.product(("1h", "4h"), ((20, 50), (20, 100), (50, 200)), ("both", "long"), (3.0, 5.0)):
    s = {"version": "01", "entry": {"indicator": "ema_cross", "direction": direction, "fast": fa, "slow": sl, "timeframe": tf},
         "stop_loss_pct": 2.0, "stop_atr_mult": stop, "take_profit_r": 0, "position_size_r": 0.5, "position_pct": 100, "max_hold_min": 0}
    record("ema_cross_engine", {"tf": tf, "fast": fa, "slow": sl, "dir": direction, "atr_stop": stop},
           {a: engine_daily(a, s) for a in ASSETS})
print("engine done", file=sys.stderr, flush=True)


def realized_vol(c, n=30):
    r = np.zeros(len(c))
    r[1:] = c[1:] / c[:-1] - 1
    out = np.full(len(c), np.nan)
    for i in range(n, len(c)):
        w = r[i - n + 1:i + 1]
        if not np.isnan(w).any():
            out[i] = np.std(w, ddof=1) * math.sqrt(365)
    return out


VOL = {a: realized_vol(CLOSE[a]) for a in ASSETS}

# --- 1. time-series momentum, volatility targeted (Moskowitz-Ooi-Pedersen; crypto TSMOM) --------
for look, mode, tv in itertools.product((30, 60, 90), ("both", "long"), (0.5, 1.0)):
    per = {}
    for a in ASSETS:
        c = CLOSE[a]
        past = np.full(len(c), np.nan)
        past[look:] = c[look:] / c[:-look] - 1
        sig = np.sign(past)
        if mode == "long":
            sig = np.maximum(sig, 0)
        w = sig * np.minimum(1.0, tv / VOL[a])  # scale down in volatile times, never leverage
        w[np.isnan(w)] = 0
        per[a] = weights_to_returns(a, w)
    record("tsmom_voltarget", {"lookback": look, "mode": mode, "target_vol": tv}, per)

# --- 2. cross-sectional momentum (Liu-Tsyvinski-Wu): weekly, top-K long (optionally bottom-K short)
for look, k, short in itertools.product((14, 30, 60), (3, 4), (False, True)):
    W = {a: np.zeros(len(DATES)) for a in ASSETS}
    cur = {a: 0.0 for a in ASSETS}
    for i in range(look, len(DATES)):
        if i % 7 == 0:
            score = {a: CLOSE[a][i] / CLOSE[a][i - look] - 1 for a in ASSETS
                     if not np.isnan(CLOSE[a][i]) and not np.isnan(CLOSE[a][i - look]) and a != "PAXG/USDT"}
            ranked = sorted(score, key=score.get, reverse=True)
            n_assets = len(ASSETS)
            cur = {a: 0.0 for a in ASSETS}
            for a in ranked[:k]:
                cur[a] = n_assets / k * (0.5 if short else 1.0)  # portfolio() divides by the pair count
            if short:
                for a in ranked[-k:]:
                    cur[a] = -n_assets / k * 0.5
        for a in ASSETS:
            W[a][i] = cur[a]
    record("xs_momentum", {"lookback": look, "top_k": k, "long_short": short}, {a: weights_to_returns(a, W[a]) for a in ASSETS})

# --- 3. funding carry: long spot + short perp while funding is positive (Schmeling-Schrimpf-Todorov)
for avg_days, thr in itertools.product((3, 7), (0.0, 0.0001)):
    per = {}
    for a in ASSETS:
        f = FUND[a]
        avg = np.convolve(f, np.ones(avg_days) / avg_days, mode="full")[:len(f)]
        on = (avg > thr * 3).astype(float)  # thr per 8h period -> per day
        on[np.isnan(CLOSE[a])] = 0
        held = np.zeros(len(on))
        held[1:] = on[:-1]
        turn = np.abs(np.diff(np.concatenate([[0], held])))
        # capital = the spot notional (the perp is margined with it); both legs pay costs
        per[a] = held * f - turn * (sum(SPOT) + sum(PERP))
    record("funding_carry", {"avg_days": avg_days, "min_rate_8h": thr}, per)


# --- 4. Donchian breakout on daily candles (Turtle-style), long+short or long ---------------------
def donchian_weights(a, n, m, mode):
    d = D[a]
    c_full, h_full, l_full = d["close"], d["high"], d["low"]
    idx = ((d["t"] - DATES[0]) // DAY).astype(int)
    w = np.zeros(len(DATES))
    state = 0
    for j in range(max(n, m), len(c_full)):
        hi_n, lo_n = h_full[j - n:j].max(), l_full[j - n:j].min()
        hi_m, lo_m = h_full[j - m:j].max(), l_full[j - m:j].min()
        if c_full[j] > hi_n:
            state = 1
        elif c_full[j] < lo_n:
            state = -1 if mode == "both" else 0
        elif state == 1 and c_full[j] < lo_m:
            state = 0
        elif state == -1 and c_full[j] > hi_m:
            state = 0
        if 0 <= idx[j] < len(DATES):
            w[idx[j]] = state
    return w


for (n, m), mode in itertools.product(((20, 10), (55, 20)), ("both", "long")):
    per = {}
    for a in ASSETS:
        w = donchian_weights(a, n, m, mode) * np.nan_to_num(np.minimum(1.0, 0.8 / VOL[a]), nan=0)
        per[a] = weights_to_returns(a, w)
    record("donchian_daily", {"entry": n, "exit": m, "mode": mode}, per)

# --- 5. price vs its moving average (Detzel et al.): long above the MA, cash below ----------------
for ma, gate in itertools.product((50, 100, 200), ("own", "btc")):
    per = {}
    btc = CLOSE["BTC/USDT"]
    btc_ma = pd.Series(btc).rolling(ma).mean().to_numpy()  # NaN until a full window exists
    for a in ASSETS:
        c = CLOSE[a]
        own_ma = pd.Series(c).rolling(ma).mean().to_numpy()
        above = (c > own_ma) if gate == "own" else (btc > btc_ma)  # NaN comparisons are False: cash
        w = above.astype(float)
        w[np.isnan(c)] = 0
        per[a] = weights_to_returns(a, w)
    record("ma_regime", {"ma_days": ma, "gate": gate}, per)

out = {"years": [[dt.datetime.fromtimestamp(lo / 1000, dt.timezone.utc).date().isoformat(),
                  dt.datetime.fromtimestamp(hi / 1000, dt.timezone.utc).date().isoformat()] for lo, hi in YEARS],
       "results": results}
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("ok", file=sys.stderr)
