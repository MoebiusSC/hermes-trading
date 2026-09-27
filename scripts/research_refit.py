"""Does reviewing (re-choosing parameters and weights) more often make more money? Five years, 12 pairs.

  python scripts/research_refit.py out.json

Reuses the candidate sleeves and combinations of research_5y.py. From the start of year 3 to the
end, every `period` days the combination with the best Sharpe on the training data (all prior data,
or only the last 365 days) is chosen and traded until the next review; switching costs are charged.
The resulting out-of-sample chain is compared with never reviewing (the combination now live).
"""
import json
import math
import sys
from pathlib import Path

import numpy as np

src = (Path(__file__).parent / "research_5y.py").read_text(encoding="utf-8")
ns: dict = {"__name__": "research_5y_lib"}
exec(compile(src.split("LIVE = (")[0], "research_5y", "exec"), ns)  # sleeves D, combos, GRID, YEARS
GRID, YEARS, combos, DAY = ns["GRID"], ns["YEARS"], ns["combos"], ns["DAY"]
COST = 2 * sum(ns["COSTS"])  # a switch closes and reopens about half the book: ~one round trip per switch


def sharpe(x):
    sd = np.std(x, ddof=1)
    return float(np.mean(x) / sd * math.sqrt(365)) if len(x) > 30 and sd > 0 else -9.0


def run(period, window):
    start = np.searchsorted(GRID, YEARS[2][0])
    out, chosen, switches = np.zeros(len(GRID) - start), None, 0
    keys = list(combos)
    for i in range(start, len(GRID), period):
        lo = 0 if window is None else max(0, i - window)
        best = max(keys, key=lambda c: sharpe(combos[c][lo:i]))
        seg = combos[best][i:i + period].copy()
        if chosen is not None and best != chosen:
            switches += 1
            seg[0] -= COST * 0.5
        chosen = best
        out[i - start:i - start + len(seg)] = seg
    eq = np.cumprod(1 + out)
    years = len(out) / 365
    return {"cagr": (eq[-1] ** (1 / years) - 1) * 100, "sharpe": sharpe(out), "switches": switches,
            "dd": float(np.max(1 - eq / np.maximum.accumulate(eq))) * 100,
            "per_year": [float(np.prod(1 + out[(GRID[start:] >= lo) & (GRID[start:] < hi)]) - 1) * 100 for lo, hi in YEARS[2:]]}


live = ("ema4h 50/200 stop5", "tsmom1d 120 long vol0.5", 0.5)
start = np.searchsorted(GRID, YEARS[2][0])
fixed = combos[live][start:]
eq = np.cumprod(1 + fixed)
res = {"never_review (live combo)": {"cagr": (eq[-1] ** (365 / len(fixed)) - 1) * 100, "sharpe": sharpe(fixed), "switches": 0,
                                     "dd": float(np.max(1 - eq / np.maximum.accumulate(eq))) * 100,
                                     "per_year": [float(np.prod(1 + fixed[(GRID[start:] >= lo) & (GRID[start:] < hi)]) - 1) * 100 for lo, hi in YEARS[2:]]}}
for period in (365, 180, 90, 30):
    for window, wname in ((None, "all history"), (365, "last year")):
        res[f"every {period}d, {wname}"] = run(period, window)
        print(period, wname, {k: round(v, 2) if isinstance(v, float) else v for k, v in res[f'every {period}d, {wname}'].items()}, file=sys.stderr, flush=True)
json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
print("ok", file=sys.stderr)
