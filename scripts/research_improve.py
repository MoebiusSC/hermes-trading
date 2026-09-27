"""Can the stock rotations get a better risk-adjusted return? Two changes tested on the live portfolio
(2/3 US rotation top 5 + 1/3 Europe rotation top 10, as in goal.yaml), over 5 and 10 years:

  uv run python scripts/research_improve.py out.json

1. Staggered rebalancing: each rotation split into 4 tranches, each its own account with a quarter of
   the capital, rebalancing on the month's 1st, 6th, 11th and 16th trading day (each tranche keeps the
   12-1 monthly rule). Doesn't change the expected return; removes the luck of the rebalance day, which
   alone moved the US rotation between 22.6% and 27.8%/yr over five years (research_global.py).
2. Volatility scaling (Barroso & Santa-Clara 2015, "Momentum has its moments"): each rotation invests
   min(cap, target / realized vol) of its account, the vol measured on the rotation's own daily returns
   over the previous `window` days; the rest stays in cash (no interest). cap 1 = never margin, as the
   bot today; cap 1.5 borrows at 6%/yr. Exposure changes pay the same slippage as trades. Updated
   daily, weekly or only at the monthly rebalance (what the bot could do without new order flow).

Every variant runs with 5 starting offsets (the 1st..5th trading day for the single-date rotation and
the first tranche) so a result isn't the luck of one calendar; min/median/max are reported.
Data and universes: research_global.py (5y: Sep 2021 universes, 10y: Sep 2016).
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import research_global as R  # noqa: E402

BORROW = 0.06  # %/yr on margin above 100% exposure
OFFSETS = range(5)
WEIGHTS = {"EEUU (actual)": (2 / 3, 5, "SPY"), "Europa NYSE/NASDAQ": (1 / 3, 10, "VGK")}


def tranches(px, syms, grid, top, index, first: int, k: int) -> pd.Series:
    """k independent accounts, each 1/k of the capital, rebalancing `21 // k`... days apart."""
    step = 20 // k if k > 1 else 0
    curves = [(1 + R.rotation(px, syms, grid, top, "none", index, offset=first + i * step)[0]).cumprod() for i in range(k)]
    eq = sum(curves) / k
    return eq.pct_change().fillna(0.0)


def vol_scale(r: pd.Series, target: float, window: int, cap: float, every: str) -> pd.Series:
    """Invest min(cap, target / trailing vol) of the account; the vol known at the previous close."""
    vol = r.rolling(window).std(ddof=1).shift(1) * np.sqrt(252)
    raw = (target / vol).clip(upper=cap).fillna(1.0)
    if every == "daily":
        e = raw
    else:
        key = r.index.to_period("W" if every == "weekly" else "M")
        e = raw.groupby(key).transform("first")  # set on the period's first day, held to its end
    cost = e.diff().abs().fillna(0.0) * R.SLIP
    return e * r - (e - 1).clip(lower=0) * BORROW / 252 - cost, e


def variants(px, U, grid, first: int) -> dict[str, pd.Series]:
    sleeves = {}
    for name, (w, top, index) in WEIGHTS.items():
        syms = U[name][0]
        single = R.rotation(px, syms, grid, top, "none", index, offset=first)[0]
        four = tranches(px, syms, grid, top, index, first, 4)
        sleeves[name] = {"base": single, "tramos": four}
        for base_name, base in (("base", single), ("tramos", four)):
            for target in (0.15, 0.20):
                for window in (20, 60):
                    for every in ("daily", "weekly", "monthly"):
                        for cap in (1.0, 1.5):
                            if cap > 1 and (window != 60 or every != "weekly"):
                                continue  # leverage only on the one setting it would be run with
                            key = f"{base_name} + vol {int(target * 100)}% {window}d {every}{' x1.5' if cap > 1 else ''}"
                            sleeves[name][key] = vol_scale(base, target, window, cap, every)[0]
    return {k: sum(sleeves[n][k] * WEIGHTS[n][0] for n in WEIGHTS) for k in sleeves["EEUU (actual)"]}


def worst_12m(r: pd.Series) -> float:
    eq = (1 + r).cumprod()
    return float((eq / eq.shift(252) - 1).min()) * 100


def main() -> None:
    out = {}
    for period in ("10y", "5y"):
        start, src, U = R.PERIODS[period]
        px = R.load(src)
        grid = px.index[(px.index >= start) & (px.index < R.END)]
        runs = [variants(px, U, grid, o) for o in OFFSETS]
        res = {}
        for key in runs[0]:
            stats = [{**R.stats(run[key]), "worst12": worst_12m(run[key]),
                      "sharpe_h1": R.windows(run[key])["1a mitad"]["sharpe"], "sharpe_h2": R.windows(run[key])["2a mitad"]["sharpe"]}
                     for run in runs]
            res[key] = {m: [round(min(s[m] for s in stats), 2), round(float(np.median([s[m] for s in stats])), 2),
                            round(max(s[m] for s in stats), 2)] for m in stats[0]}
        out[period] = res
        print(f"\n== {period} ({grid[0].date()} - {grid[-1].date()}) == min/mediana/max en {len(OFFSETS)} calendarios")
        for key, v in sorted(res.items(), key=lambda kv: -kv[1]["sharpe"][1]):
            print(f"{key:42} CAGR {v['cagr'][1]:5.1f} [{v['cagr'][0]:5.1f}-{v['cagr'][2]:5.1f}]  DD {v['dd'][1]:4.1f}  "
                  f"peor 12m {v['worst12'][1]:6.1f}  Sharpe {v['sharpe'][1]:.2f} [{v['sharpe'][0]:.2f}-{v['sharpe'][2]:.2f}]  "
                  f"mitades {v['sharpe_h1'][1]:.2f}/{v['sharpe_h2'][1]:.2f}", flush=True)
    json.dump(out, open(sys.argv[1], "w"), indent=1)


if __name__ == "__main__":
    main()
