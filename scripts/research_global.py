"""Should the bot add European or Chinese stocks? The live stock rule (hermes_trading/rotation.py: each
month hold the `top` stocks with the best 12-1 month return) and holding, per region and combined with
the US rotation the bot already runs.

  uv run python scripts/research_global.py out.json [5y|10y]

5y (Sep 2021 - Sep 2026): universes of September 2021, incl. OTC ADRs and A-shares; US stocks from the
Alpaca cache (scripts/fetch_stocks.py), as in scripts/research_stocks.py.
10y (Sep 2016 - Sep 2026): NYSE/NASDAQ universes of September 2016; everything from Yahoo, since
Alpaca's SIP bars start in January 2016, a year short of the first 12-1 ranking.
Data: scripts/fetch_global.py (Yahoo, adjusted, USD). Universes are the largest companies of each
region at the start of the test with a line Alpaca can trade, so nothing is chosen with hindsight;
delisted Chinese ADRs are kept until their last NYSE session (fetch_global.py).
Costs per side: 0.03% on NYSE/NASDAQ (as research_stocks.py), 0.15% on OTC ADRs and A-shares (wider
spreads; Stock Connect fees and stamp duty). No dividend withholding: adjusted prices reinvest the
gross dividend, so foreign stocks are flattered by ~0.3-0.6%/yr against what an account would get.
"""
import json
import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_global import (A_SHARES, CN_2016, CN_LISTED, CN_OTC, DELISTED, EU_2016, EU_LISTED, EU_OTC,  # noqa: E402
                          OUT as GLOBAL)
from fetch_stocks import OUT as STOCKS, UNIVERSE_2016, UNIVERSE_2021  # noqa: E402

END = "2026-09-26"
SLIP, SLIP_WIDE = 0.0003, 0.0015
US = UNIVERSE_2021
PERIODS = {
    "5y": ("2021-09-27", "alpaca", {
        "EEUU (actual)": (US, "SPY"),
        "Europa NYSE/NASDAQ": (EU_LISTED, "VGK"),
        "Europa + OTC": (EU_LISTED + EU_OTC, "VGK"),
        "China NYSE/NASDAQ": (CN_LISTED, "MCHI"),
        "China + OTC": (CN_LISTED + CN_OTC, "MCHI"),
        "China A (Stock Connect)": (A_SHARES, "ASHR"),
        "Global NYSE/NASDAQ": (US + EU_LISTED + CN_LISTED, "SPY"),
        "Global + OTC": (US + EU_LISTED + EU_OTC + CN_LISTED + CN_OTC, "SPY"),
    }),
    "10y": ("2016-09-27", "yahoo", {
        "EEUU (actual)": (UNIVERSE_2016, "SPY"),
        "Europa NYSE/NASDAQ": (EU_2016, "VGK"),
        "China NYSE/NASDAQ": (CN_2016, "MCHI"),
        "Global NYSE/NASDAQ": (UNIVERSE_2016 + EU_2016 + CN_2016, "SPY"),
    }),
}
ETFS = ["SPY", "QQQ", "VGK", "EZU", "FEZ", "EWG", "EWU", "EWQ", "EWL", "EWP", "HEDJ", "MCHI", "FXI", "KWEB", "ASHR",
        "GXC", "CQQQ"]


def load(source: str) -> pd.DataFrame:
    """Daily closes on the US trading calendar; foreign closes carried over their holidays. source:
    "alpaca" takes the 2021 US universe, SPY and QQQ from the Alpaca cache, "yahoo" all from Yahoo."""
    g = pd.read_csv(GLOBAL / "closes.csv", index_col=0, parse_dates=True)
    if source == "yahoo":
        us = pd.DataFrame(index=g.index[g["SPY"].notna()])
    else:
        cols = {}
        for sym in US + ["SPY", "QQQ"]:
            d = json.loads((STOCKS / f"{sym}_1d.json").read_text())
            idx = pd.to_datetime(d["t"], unit="ms", utc=True).tz_localize(None).normalize()
            cols[sym] = pd.Series(d["close"], index=idx)
        us = pd.DataFrame(cols)
        g = g.drop(columns=[c for c in us if c in g])
    g = g.reindex(us.index.union(g.index)).ffill(limit=10).reindex(us.index)
    for sym, last in DELISTED.items():
        g.loc[g.index > last, sym] = np.nan
    return pd.concat([us, g], axis=1)


def stats(r: pd.Series) -> dict:
    eq = (1 + r).cumprod()
    sd = r.std(ddof=1)
    years = len(r) / 252
    return {"cagr": float(eq.iloc[-1] ** (1 / years) - 1) * 100, "ret": float(eq.iloc[-1] - 1) * 100,
            "vol": float(sd * math.sqrt(252)) * 100, "dd": float((1 - eq / eq.cummax()).max()) * 100,
            "sharpe": float(r.mean() / sd * math.sqrt(252)) if sd > 0 else 0.0}


def hold_ew(px: pd.DataFrame, syms: list[str], grid: pd.DatetimeIndex) -> pd.Series:
    """Equal weight across the universe, rebalanced daily (as 'comprar y mantener' in research_stocks.py)."""
    return px[syms].pct_change(fill_method=None).loc[grid].mean(axis=1).fillna(0.0)


def rotation(px: pd.DataFrame, syms: list[str], grid: pd.DatetimeIndex, top: int, filt: str, index: str,
             lookback: int = 252, skip: int = 21, offset: int = 0) -> tuple[pd.Series, list]:
    """research_stocks.rotation: at the close of each month's first trading day hold the `top` best 12-1
    month returns, equal weight. filt: none | abs (only positive scores) | idx200 (cash while the region's
    ETF is below its 200-day average) | ma200 (each pick only while above its own 200-day average).
    offset: rebalance on the month's offset-th trading day instead of the first (timing luck)."""
    p = px[syms].to_numpy()
    ix = px[index].to_numpy()
    cost = np.array([SLIP_WIDE if (s in EU_OTC or s in CN_OTC or s in A_SHARES) else SLIP for s in syms])
    pos = px.index.get_indexer(grid)
    r = np.zeros(len(grid))
    held, picks = np.zeros(len(syms)), np.zeros(len(syms))
    month, log = None, []
    day_of_month = pd.Series(1, index=grid).groupby([grid.year, grid.month]).cumcount().to_numpy()
    for k, i in enumerate(pos):
        if k > 0:
            day = p[i] / p[pos[k - 1]] - 1
            r[k] = float(np.nansum(held * np.where(np.isnan(day), 0.0, day)))
        if grid[k].month != month and day_of_month[k] >= offset:
            month = grid[k].month
            score = p[i - skip] / p[i - lookback] - 1
            score = np.where(np.isnan(p[i]), np.nan, score)  # delisted: not rankable
            order = [j for j in np.argsort(-np.nan_to_num(score, nan=-9)) if not np.isnan(score[j])][:top]
            picks = np.zeros(len(syms))
            for j in order:
                if filt == "abs" and score[j] <= 0:
                    continue
                picks[j] = 1 / top
            log.append((str(grid[k].date()), [syms[j] for j in order if picks[j] > 0]))
        new = np.where(np.isnan(p[i]), 0.0, picks)  # a delisted holding is sold at its last price
        if filt == "idx200" and ix[i] < np.nanmean(ix[i - 199:i + 1]):
            new[:] = 0
        if filt == "ma200":
            new = np.where(p[i] > np.nanmean(p[i - 199:i + 1], axis=0), new, 0)
        r[k] -= float(np.sum(np.abs(new - held) * cost))
        held = new
    return pd.Series(r, index=grid), log


def windows(r: pd.Series) -> dict:
    """Year by year (Sep-Sep) and the two halves."""
    start = r.index[0]
    out = {f"{start.year + k}-{start.year + k + 1}": stats(r[(r.index >= start + pd.DateOffset(years=k)) &
                                                              (r.index < start + pd.DateOffset(years=k + 1))])
           for k in range(round(len(r) / 252))}
    half = start + (r.index[-1] - start) / 2
    out["1a mitad"], out["2a mitad"] = stats(r[r.index < half]), stats(r[r.index >= half])
    return out


def main() -> None:
    period = sys.argv[2] if len(sys.argv) > 2 else "5y"
    start, source, UNIVERSES = PERIODS[period]
    px = load(source)
    grid = px.index[(px.index >= start) & (px.index < END)]
    series, picks = {}, {}
    for sym in [e for e in ETFS if px[e].loc[grid].notna().all()]:
        series[f"ETF {sym}"] = px[sym].pct_change(fill_method=None).loc[grid].fillna(0.0)
    for name, (syms, index) in UNIVERSES.items():
        series[f"{name} | mantener todas"] = hold_ew(px, syms, grid)
        for top in (5, 10):
            for filt in ("none", "abs", "idx200", "ma200"):
                key = f"{name} | rotacion top{top} {filt}"
                series[key], log = rotation(px, syms, grid, top, filt, index)
                if top == 5 and filt == "none":
                    picks[name] = log
    us5, eu5, eu10 = ("EEUU (actual) | rotacion top5 none", "Europa NYSE/NASDAQ | rotacion top5 none",
                      "Europa NYSE/NASDAQ | rotacion top10 none")
    cn5, euh = "China NYSE/NASDAQ | rotacion top5 none", "Europa NYSE/NASDAQ | mantener todas"
    mixes = {
        "Cartera: rotacion EEUU (actual)": {us5: 1},
        "Cartera: 1/2 EEUU + 1/2 Europa top5": {us5: .5, eu5: .5},
        "Cartera: 1/2 EEUU + 1/2 Europa top10": {us5: .5, eu10: .5},
        "Cartera: 2/3 EEUU + 1/3 Europa top10": {us5: 2 / 3, eu10: 1 / 3},
        "Cartera: 1/2 EEUU + 1/2 mantener Europa": {us5: .5, euh: .5},
        "Cartera: 1/2 EEUU + 1/2 VGK": {us5: .5, "ETF VGK": .5},
        "Cartera: 1/3 EEUU + 1/3 Europa top10 + 1/3 China": {us5: 1 / 3, eu10: 1 / 3, cn5: 1 / 3},
    }
    if "Europa + OTC" in UNIVERSES:
        mixes["Cartera: 1/2 EEUU + 1/2 Europa+OTC top5"] = {us5: .5, "Europa + OTC | rotacion top5 none": .5}
        mixes["Cartera: 1/2 EEUU + 1/4 Europa + 1/4 China A"] = {us5: .5, eu5: .25,
                                                                  "China A (Stock Connect) | rotacion top5 none": .25}
    for name, w in mixes.items():
        series[name] = sum(series[k] * v for k, v in w.items())
    res = {"period": [str(grid[0].date()), str(grid[-1].date()), len(grid)], "all": {}, "windows": {}, "picks": picks}
    for name, r in series.items():
        res["all"][name] = stats(r)
        res["windows"][name] = windows(r)
    weekly = pd.DataFrame({k: series[k] for k in [f"ETF {e}" for e in ("SPY", "QQQ", "VGK", "MCHI", "ASHR")] +
                           [f"{n} | rotacion top5 none" for n in UNIVERSES if not n.startswith("Global")] if k in series})
    weekly = (1 + weekly).resample("W-FRI").prod() - 1
    res["corr_weekly"] = weekly.corr().round(2).to_dict()
    res["weekly_curves"] = {k: ((1 + series[k]).cumprod().resample("W-FRI").last().round(4)).tolist()
                            for k in list(mixes) + [f"ETF {s}" for s in ("SPY", "VGK", "MCHI", "ASHR")] +
                            [f"{n} | rotacion top5 none" for n in UNIVERSES] if k in series}
    res["weekly_dates"] = [str(d.date()) for d in (1 + series["ETF SPY"]).cumprod().resample("W-FRI").last().index]
    # timing luck: the same rule rebalanced on each of the month's first 10 trading days
    res["offsets"] = {}
    for name, (syms, index) in UNIVERSES.items():
        for top in (5, 10):
            runs = [stats(rotation(px, syms, grid, top, "none", index, offset=o)[0]) for o in range(10)]
            res["offsets"][f"{name} top{top}"] = {k: [round(min(x[k] for x in runs), 2), round(float(np.median([x[k] for x in runs])), 2),
                                                      round(max(x[k] for x in runs), 2)] for k in ("cagr", "dd", "sharpe")}
    for name, (syms, _) in UNIVERSES.items():
        hold = stats(hold_ew(px, syms, grid))
        print(f"{name:26} mantener CAGR {hold['cagr']:5.1f} Sharpe {hold['sharpe']:.2f} | rotacion (min/mediana/max en 10 dias): "
              + "  ".join(f"top{t} CAGR {res['offsets'][f'{name} top{t}']['cagr']} Sharpe {res['offsets'][f'{name} top{t}']['sharpe']}"
                          for t in (5, 10)))
    json.dump(res, open(sys.argv[1], "w"), indent=1, default=float)
    rows = sorted(res["all"].items(), key=lambda kv: -kv[1]["sharpe"])
    for name, s in rows:
        print(f"{name:62} CAGR {s['cagr']:6.1f}%  DD {s['dd']:5.1f}%  vol {s['vol']:5.1f}%  Sharpe {s['sharpe']:5.2f}")


if __name__ == "__main__":
    main()
