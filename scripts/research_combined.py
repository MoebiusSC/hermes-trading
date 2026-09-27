"""Three years, 12 crypto pairs: the live trend strategy alone and combined with time-series momentum
and the moving-average regime, all run by the live engine (backtest.simulate) with its costs.

  python scripts/research_combined.py out.json

Per strategy ("sleeve") each pair is simulated at 100% exposure; a portfolio averages the pairs, and
a combination gives each sleeve an equal share of the capital (rebalanced daily). Sharpe does not
depend on the exposure; returns and drawdowns scale roughly with it.
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
ASSETS = [a for a in config.goal_assets(goal) if not config.is_stock(a)]
DAY = 86400000
END = dt.datetime(2026, 9, 27, tzinfo=dt.timezone.utc).timestamp() * 1000
YEARS = [(END - (3 - k) * 365 * DAY, END - (2 - k) * 365 * DAY) for k in range(3)]
GRID = np.arange(YEARS[0][0], END, DAY)
HIST_DAYS = {"4h": 1250, "1d": 1400}  # three years plus warm-up for the slowest signal
COSTS = rules.costs(goal, False)
FLOOR = rules.min_stop_frac(goal, False)


def strat(indicator, tf, direction, stop, **entry):
    return {"version": "01", "entry": {"indicator": indicator, "direction": direction, "timeframe": tf, **entry},
            "stop_loss_pct": 10.0, "stop_atr_mult": stop, "take_profit_r": 0, "position_size_r": 0.5,
            "position_pct": 100, "max_hold_min": 0}


def pair_daily(asset, s):
    tf = s["entry"]["timeframe"]
    e = b.history(asset, tf, HIST_DAYS[tf])
    sim = b.simulate(s, e, None, 10000, *COSTS, FLOOR)
    t = np.asarray([dt.datetime.fromisoformat(q["ts"]).timestamp() * 1000 for q in sim["equity_curve"]])
    v = np.asarray([q["equity"] for q in sim["equity_curve"]])
    eq = np.interp(GRID, t, v, left=np.nan)
    r = np.zeros(len(GRID))
    r[1:] = eq[1:] / eq[:-1] - 1
    listed = ~np.isnan(eq)
    r[np.isnan(r)] = 0
    n = sum(1 for x in sim["trades"] if x["opened_at"] >= b._iso(YEARS[0][0]))
    return r, listed, n


def sleeve(s):
    rs, ls, n = zip(*(pair_daily(a, s) for a in ASSETS))
    rs, ls = np.vstack(rs), np.vstack(ls)
    return (rs * ls).sum(0) / np.maximum(ls.sum(0), 1), sum(n)


def stats(daily):
    out = []
    for lo, hi in YEARS + [(YEARS[0][0], END)]:
        x = daily[(GRID >= lo) & (GRID < hi)]
        eq = np.cumprod(1 + x)
        sd = np.std(x, ddof=1)
        out.append({"ret": float(eq[-1] - 1) * 100, "dd": float(np.max(1 - eq / np.maximum.accumulate(eq))) * 100,
                    "sharpe": float(np.mean(x) / sd * math.sqrt(365)) if sd > 0 else 0.0})
    return out  # Y1, Y2, Y3, whole period


LIVE = strat("ema_cross", "4h", "both", 3.0, fast=50, slow=200)
candidates = {"trend (live) ema 4h 50/200 both": LIVE}
for look, direction, tv, stop in itertools.product((30, 60, 90), ("long", "both"), (0.5, 0.0), (3.0, 6.0)):
    candidates[f"tsmom 1d {look} {direction} vol{tv} stop{stop:g}"] = strat("tsmom", "1d", direction, stop, lookback=look, target_vol=tv)
for ma, direction, stop in itertools.product((50, 100, 200), ("long", "both"), (3.0, 6.0)):
    candidates[f"ma_regime 1d {ma} {direction} stop{stop:g}"] = strat("ma_regime", "1d", direction, stop, ma=ma)

sleeves, rows = {}, []
for name, s in candidates.items():
    daily, n = sleeve(s)
    sleeves[name] = daily
    rows.append({"name": name, "trades": n, "years": stats(daily)})
    print(name, [round(y["ret"], 1) for y in rows[-1]["years"]], file=sys.stderr, flush=True)


def pick(prefix):
    """The variant of a family with the best Y1+Y2 Sharpe (the last year is kept out of the choice)."""
    fam = [r for r in rows if r["name"].startswith(prefix)]
    return max(fam, key=lambda r: r["years"][0]["sharpe"] + r["years"][1]["sharpe"])["name"]


chosen = {"trend": "trend (live) ema 4h 50/200 both", "tsmom": pick("tsmom"), "ma": pick("ma_regime"),
          # the variants proposed before this study (tsmom 60 long vol 0.5, ma 100 long)
          "tsmom_proposed": "tsmom 1d 60 long vol0.5 stop6", "ma_proposed": "ma_regime 1d 100 long stop6"}
combos = {}
for label, parts in {
    "trend only": ["trend"],
    "trend + tsmom": ["trend", "tsmom"], "trend + ma": ["trend", "ma"], "trend + tsmom + ma": ["trend", "tsmom", "ma"],
    "trend + tsmom (proposed)": ["trend", "tsmom_proposed"], "trend + ma (proposed)": ["trend", "ma_proposed"],
    "trend + tsmom + ma (proposed)": ["trend", "tsmom_proposed", "ma_proposed"],
}.items():
    daily = np.mean([sleeves[chosen[k]] for k in parts], axis=0)
    combos[label] = {"parts": [chosen[k] for k in parts], "years": stats(daily)}
names = [chosen[k] for k in ("trend", "tsmom", "ma", "tsmom_proposed", "ma_proposed")]
corr = np.corrcoef(np.vstack([sleeves[n] for n in names]))
json.dump({"years": [[b._iso(lo)[:10], b._iso(hi)[:10]] for lo, hi in YEARS], "rows": rows, "chosen": chosen,
           "combos": combos, "corr": {"names": names, "matrix": corr.tolist()}}, open(sys.argv[1], "w"), indent=1, default=float)
print("ok", file=sys.stderr)
