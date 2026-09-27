"""The live combination under the OKX cost model, with perpetual funding. Five years, 12 pairs.

  python scripts/research_okx.py out.json

Scenarios (per side: fee + slippage):
  spot       both sides 0.10% + 0.02%, no funding (the model used until now)
  okx        longs on spot 0.10% + 0.02%; shorts on the USDT perpetual 0.05% + 0.02%, receiving/paying
             funding; PAXG (no OKX perpetual) trades long only
  okx_perps  longs too on the perpetual 0.05% + 0.02%, paying funding
Funding: Binance USDT-M history (cache/funding, scripts/fetch_history.py) stands in for OKX, which
serves only ~3 months; over the months both have, their yearly averages differ by under 1%.
Trend sleeve + momentum sleeve 50/50 at 100% exposure, as in research_costs.py.
"""
import json
import sys
from pathlib import Path

import numpy as np

src = (Path(__file__).parent / "research_5y.py").read_text(encoding="utf-8")
ns: dict = {"__name__": "research_5y_lib"}
exec(compile(src.split("# --- candidate sleeves")[0], "research_5y", "exec"), ns)
b, strat, YEARS, GRID, NY, ASSETS = ns["b"], ns["strat"], ns["YEARS"], ns["GRID"], ns["NY"], ns["ASSETS"]
FLOOR = ns["FLOOR"]
FUND = {a: json.loads((b.FUNDING_DIR / f"{b.config.asset_slug(a)}.json").read_text()) for a in ASSETS}
NO_PERP = {"PAXG/USDT"}
SPOT, PERP = (0.001, 0.0002), (0.0005, 0.0002)
SCENARIOS = {
    "spot": {"long": SPOT, "short": SPOT, "funded": ()},
    "okx": {"long": SPOT, "short": PERP, "funded": ("short",)},
    "okx_perps": {"long": PERP, "short": PERP, "funded": ("long", "short")},
}
TREND = strat("ema_cross", "4h", "both", 5.0, fast=50, slow=200)
MOM = strat("tsmom", "1d", "long", 3.0, lookback=120, target_vol=0.5)
since = b._iso(YEARS[0][0])


def sleeve(s, sc):
    rets, listed, flows = [], [], {"fees": 0.0, "funding_long": 0.0, "funding_short": 0.0, "trades": 0}
    for a in ASSETS:
        strategy = s
        if sc["funded"] and a in NO_PERP and s["entry"]["direction"] == "both":
            strategy = {**s, "entry": {**s["entry"], "direction": "long"}}
        sim = b.simulate(strategy, ns["candles"](a, s["entry"]["timeframe"]), None, 10000, *sc["long"], FLOOR,
                         sc["short"], FUND[a] if sc["funded"] else None, sc["funded"])
        for t in sim["trades"]:
            if t["opened_at"] >= since:
                flows["fees"] += t["fees"] / 10000
                flows["funding_" + t["direction"]] += t["funding"] / 10000
                flows["trades"] += 1
        t = np.asarray([ns["dt"].datetime.fromisoformat(q["ts"]).timestamp() * 1000 for q in sim["equity_curve"]])
        v = np.asarray([q["equity"] for q in sim["equity_curve"]])
        eq = np.interp(GRID, t, v, left=np.nan)
        r = np.zeros(len(GRID))
        r[1:] = eq[1:] / eq[:-1] - 1
        ok = ~np.isnan(eq) & ~np.isnan(np.concatenate([[np.nan], eq[:-1]]))
        rets.append(np.where(ok, r, 0))
        listed.append(ok)
    rets, listed = np.vstack(rets), np.vstack(listed)
    # fees and funding: % of one account's starting capital, per account and year
    flows = {k: v / len(ASSETS) / NY * (100 if k != "trades" else 1) for k, v in flows.items()}
    return (rets * listed).sum(0) / np.maximum(listed.sum(0), 1), flows


out = {}
for name, sc in SCENARIOS.items():
    dt_, ft = sleeve(TREND, sc)
    dm, fm = sleeve(MOM, sc)
    res = {}
    for label, daily in (("trend", dt_), ("momentum", dm), ("combo_50_50", 0.5 * dt_ + 0.5 * dm)):
        eq = np.cumprod(1 + daily[GRID >= YEARS[0][0]])
        res[label] = {"years": [ns["year_stats"](daily, k) for k in range(NY)],
                      "cagr": float(eq[-1] ** (365 / len(eq)) - 1) * 100,
                      "sharpe5y": ns["span_sharpe"](daily, list(range(NY)))}
    res["flows_per_account_year_pct"] = {"trend": ft, "momentum": fm}
    out[name] = res
    print(name, {k: (round(v["cagr"], 2), round(v["sharpe5y"], 2)) for k, v in res.items() if "cagr" in v}, ft, fm,
          file=sys.stderr, flush=True)
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("ok", file=sys.stderr)
