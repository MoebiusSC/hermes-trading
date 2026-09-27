"""How much do fees and slippage cost the live combination? Five years, 12 pairs, live engine.

  python scripts/research_costs.py out.json

Runs the trend sleeve (EMA 50/200 4h, 5 ATR stop) and the momentum sleeve (tsmom 120d long, 50% vol
target) under several per-side cost levels, 50/50, at 100% exposure.
"""
import json
import sys
from pathlib import Path

import numpy as np

src = (Path(__file__).parent / "research_5y.py").read_text(encoding="utf-8")
ns: dict = {"__name__": "research_5y_lib"}
exec(compile(src.split("# --- candidate sleeves")[0], "research_5y", "exec"), ns)
b, strat, YEARS, GRID, NY = ns["b"], ns["strat"], ns["YEARS"], ns["GRID"], ns["NY"]

LEVELS = {  # (fee, slippage) per side, as fractions
    "sin costos": (0.0, 0.0),
    "perpetuos maker (0.02% + 0.01%)": (0.0002, 0.0001),
    "perpetuos taker (0.05% + 0.02%)": (0.0005, 0.0002),
    "spot 0.1% + 0.02% (lo que simula hoy)": (0.001, 0.0002),
    "spot 0.25% + 0.05%": (0.0025, 0.0005),
    "spot 0.40% + 0.05%": (0.004, 0.0005),
}
TREND = strat("ema_cross", "4h", "both", 5.0, fast=50, slow=200)
MOM = strat("tsmom", "1d", "long", 3.0, lookback=120, target_vol=0.5)
out = {}
for label, costs in LEVELS.items():
    ns["COSTS"] = costs
    ns["FLOOR"] = 0.0
    daily = 0.5 * ns["sleeve_daily"](TREND) + 0.5 * ns["sleeve_daily"](MOM)
    ys = [ns["year_stats"](daily, k) for k in range(NY)]
    eq = np.cumprod(1 + daily)
    out[label] = {"years": ys, "cagr": float(eq[-1] ** (365 / len(daily)) - 1) * 100,
                  "sharpe5y": ns["span_sharpe"](daily, list(range(NY)))}
    print(label, round(out[label]["cagr"], 2), [round(y["ret"], 1) for y in ys], file=sys.stderr, flush=True)
trades = {}
for name, s in (("trend", TREND), ("momentum", MOM)):
    n = sum(len([x for x in b.simulate(s, ns["candles"](a, s["entry"]["timeframe"]), None, 10000, 0.001, 0.0002)["trades"]
                 if x["opened_at"] >= b._iso(YEARS[0][0])]) for a in ns["ASSETS"])
    trades[name] = n / len(ns["ASSETS"]) / NY
out["_trades_per_account_per_year"] = trades
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("ok", file=sys.stderr)
