"""Five years, 12 crypto pairs: refine the combined operation (trend sleeve + momentum sleeve) by
anchored walk-forward, all with the live engine (backtest.simulate) and its costs.

  python scripts/research_5y.py out.json

Years Y1..Y5 are the last five 365-day windows (Y1 starts Sep 2021). For each test year from Y3 on,
the parameters (single strategy or combination, and its weights) are chosen on the years before it
only, then measured on the test year: that chain of out-of-sample years is the honest estimate of
"refine with the data you have, then trade". Each sleeve runs its pairs at 100% exposure; returns
and drawdowns scale roughly with the exposure used live, Sharpe doesn't.
"""
import datetime as dt
import itertools
import json
import math
import sys

import numpy as np

from hermes_trading import backtest as b, config
from hermes_trading import strategy as rules
from hermes_trading.storage import load_yaml

goal = load_yaml(config.GOAL_FILE)
ASSETS = [a for a in config.goal_assets(goal) if not config.is_stock(a) and not config.sleeve(a)]
DAY = 86400000
END = dt.datetime(2026, 9, 27, tzinfo=dt.timezone.utc).timestamp() * 1000
NY = 5
YEARS = [(END - (NY - k) * 365 * DAY, END - (NY - 1 - k) * 365 * DAY) for k in range(NY)]
GRID = np.arange(YEARS[0][0], END, DAY)
HIST_DAYS = 2050
COSTS = rules.costs(goal, False)
FLOOR = rules.min_stop_frac(goal, False)
H = {}


def strat(indicator, tf, direction, stop, **entry):
    return {"version": "01", "entry": {"indicator": indicator, "direction": direction, "timeframe": tf, **entry},
            "stop_loss_pct": 10.0, "stop_atr_mult": stop, "take_profit_r": 0, "position_size_r": 0.5,
            "position_pct": 100, "max_hold_min": 0}


def candles(asset, tf):
    if (asset, tf) not in H:
        H[(asset, tf)] = b.history(asset, tf, HIST_DAYS)
    return H[(asset, tf)]


def sleeve_daily(s):
    """Equal-weight daily return of the 12 pairs (each pair counts once it is listed)."""
    rets, listed = [], []
    for a in ASSETS:
        sim = b.simulate(s, candles(a, s["entry"]["timeframe"]), None, 10000, *COSTS, FLOOR)
        t = np.asarray([dt.datetime.fromisoformat(q["ts"]).timestamp() * 1000 for q in sim["equity_curve"]])
        v = np.asarray([q["equity"] for q in sim["equity_curve"]])
        eq = np.interp(GRID, t, v, left=np.nan)
        r = np.zeros(len(GRID))
        r[1:] = eq[1:] / eq[:-1] - 1
        ok = ~np.isnan(eq) & ~np.isnan(np.concatenate([[np.nan], eq[:-1]]))
        rets.append(np.where(ok, r, 0))
        listed.append(ok)
    rets, listed = np.vstack(rets), np.vstack(listed)
    return (rets * listed).sum(0) / np.maximum(listed.sum(0), 1)


def year_stats(daily, k):
    lo, hi = YEARS[k]
    x = daily[(GRID >= lo) & (GRID < hi)]
    eq = np.cumprod(1 + x)
    sd = np.std(x, ddof=1)
    return {"ret": float(eq[-1] - 1) * 100, "dd": float(np.max(1 - eq / np.maximum.accumulate(eq))) * 100,
            "sharpe": float(np.mean(x) / sd * math.sqrt(365)) if sd > 0 else 0.0}


def span_sharpe(daily, ks):
    x = np.concatenate([daily[(GRID >= YEARS[k][0]) & (GRID < YEARS[k][1])] for k in ks])
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * math.sqrt(365)) if sd > 0 else 0.0


def chain(daily_by_year):
    """Stats of a daily series stitched from the out-of-sample years."""
    x = np.asarray(daily_by_year)
    eq = np.cumprod(1 + x)
    sd = np.std(x, ddof=1)
    years = len(x) / 365
    return {"cagr": float(eq[-1] ** (1 / years) - 1) * 100, "total": float(eq[-1] - 1) * 100,
            "dd": float(np.max(1 - eq / np.maximum.accumulate(eq))) * 100, "sharpe": float(np.mean(x) / sd * math.sqrt(365))}


# --- candidate sleeves --------------------------------------------------------------------------
trend = {f"ema4h {f}/{s} stop{st:g}": strat("ema_cross", "4h", "both", st, fast=f, slow=s)
         for (f, s), st in itertools.product(((20, 100), (30, 150), (50, 200), (50, 250)), (3.0, 5.0))}
momentum = {f"tsmom1d {lb} {d} vol{tv}": strat("tsmom", "1d", d, 3.0, lookback=lb, target_vol=tv)
            for lb, d, tv in itertools.product((30, 60, 90, 120), ("both", "long"), (0.0, 0.5))}
D = {}
for name, s in {**trend, **momentum}.items():
    D[name] = sleeve_daily(s)
    print(name, [round(year_stats(D[name], k)["ret"], 1) for k in range(NY)], file=sys.stderr, flush=True)

bh = np.mean([np.nan_to_num(np.concatenate([[0], np.diff(np.interp(GRID, np.asarray(candles(a, "1d")["t"]) + DAY,
             np.asarray(candles(a, "1d")["close"]), left=np.nan)) / np.interp(GRID, np.asarray(candles(a, "1d")["t"]) + DAY,
             np.asarray(candles(a, "1d")["close"]), left=np.nan)[:-1]])) for a in ASSETS], axis=0)

# --- combinations: trend weight w, momentum 1-w (separate accounts, daily-rebalanced mix) ----------
WEIGHTS = (1.0, 0.7, 0.5, 0.3, 0.0)
combos = {}
for tn, mn, w in itertools.product(trend, momentum, WEIGHTS):
    if w == 1.0:
        key = (tn, "-", 1.0)
    elif w == 0.0:
        key = ("-", mn, 0.0)
    else:
        key = (tn, mn, w)
    if key not in combos:
        combos[key] = (D[tn] if w > 0 else 0) * w + (D[mn] if w < 1 else 0) * (1 - w)

LIVE = ("ema4h 50/200 stop3", "tsmom1d 60 both vol0.5", 0.5)  # the combination proposed before this study

# --- anchored walk-forward: choose on years < k (best Sharpe), test on year k ------------------
wf = {"trend_only": [], "momentum_only": [], "combo": [], "proposed_fixed": []}
wf_daily = {k: [] for k in wf}
for k in range(2, NY):
    train = list(range(k))
    for label, pool in (("trend_only", [c for c in combos if c[1] == "-"]),
                        ("momentum_only", [c for c in combos if c[0] == "-"]),
                        ("combo", list(combos)), ("proposed_fixed", [LIVE])):
        best = max(pool, key=lambda c: span_sharpe(combos[c], train))
        ys = year_stats(combos[best], k)
        wf[label].append({"test_year": k + 1, "chosen": list(best), "train_sharpe": span_sharpe(combos[best], train), **ys})
        lo, hi = YEARS[k]
        wf_daily[label].extend(combos[best][(GRID >= lo) & (GRID < hi)].tolist())

final = max(combos, key=lambda c: span_sharpe(combos[c], list(range(NY))))  # for deployment: all five years
top = sorted(combos, key=lambda c: span_sharpe(combos[c], list(range(NY))), reverse=True)[:15]
out = {
    "years": [[b._iso(lo)[:10], b._iso(hi)[:10]] for lo, hi in YEARS],
    "buy_hold": [year_stats(bh, k) for k in range(NY)],
    "sleeves": {n: [year_stats(D[n], k) for k in range(NY)] + [{"sharpe5y": span_sharpe(D[n], list(range(NY)))}] for n in D},
    "walk_forward": wf, "walk_forward_chain": {k: chain(v) for k, v in wf_daily.items()},
    "final_choice": {"combo": list(final), "years": [year_stats(combos[final], k) for k in range(NY)],
                     "sharpe5y": span_sharpe(combos[final], list(range(NY)))},
    "top_by_5y_sharpe": [{"combo": list(c), "sharpe5y": span_sharpe(combos[c], list(range(NY))),
                          "years": [round(year_stats(combos[c], k)["ret"], 1) for k in range(NY)],
                          "worst_dd": max(year_stats(combos[c], k)["dd"] for k in range(NY))} for c in top],
    "proposed": {"combo": list(LIVE), "years": [year_stats(combos[LIVE], k) for k in range(NY)],
                 "sharpe5y": span_sharpe(combos[LIVE], list(range(NY)))},
    "trend_live_alone": {"years": [year_stats(D["ema4h 50/200 stop3"], k) for k in range(NY)],
                         "sharpe5y": span_sharpe(D["ema4h 50/200 stop3"], list(range(NY)))},
}
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("ok", file=sys.stderr)
