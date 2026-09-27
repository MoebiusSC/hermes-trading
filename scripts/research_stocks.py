"""Which stocks to add, and does the bot's algorithm fit them? Live engine (backtest.simulate) on
split- and dividend-adjusted Alpaca SIP bars (scripts/fetch_stocks.py).

  python scripts/research_stocks.py out.json [5y] [8y] [etfs]

Universes are the largest US companies by market cap at the START of each test, plus AMD, so the
list isn't picked with hindsight (choosing NVDA today because it rose 10x would be):
  8 years (Sep 2018 - Sep 2026), daily strategies: top 25 of Sep 2016 (SIP bars start in 2016 and an
                                                    EMA 50/200 needs 600 days to settle)
  5 years (Sep 2021 - Sep 2026), all strategies:     top 25 of Sep 2021 + AMD
Stocks are long only (the Alpaca account can't short), no commission, 0.03% slippage per side.
Each strategy uses 100% of its account; the portfolio is the equal-weight average of the stocks.
Cross-sectional momentum (rotation) is a portfolio rule the per-asset engine can't express: each
month, hold the N stocks with the best 12-1 month return.
"""
import datetime as dt
import json
import math
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from fetch_stocks import OUT, UNIVERSE_2016, UNIVERSE_2021  # noqa: E402

from hermes_trading import backtest as b  # noqa: E402
from hermes_trading import strategy as rules  # noqa: E402

UTC = dt.timezone.utc
END = dt.datetime(2026, 9, 26, tzinfo=UTC).timestamp() * 1000
DAY = 86400000
SLIP = 0.0003
H: dict = {}


def load(sym, tf):
    if (sym, tf) not in H:
        d = json.loads((OUT / f"{sym}_{tf}.json").read_text())
        keep = [i for i, t in enumerate(d["t"]) if t < END - DAY]
        H[(sym, tf)] = rules.clip_wicks({k: [d[k][i] for i in keep] for k in ("t", "open", "high", "low", "close")},
                                        rules.stock_max_wick(tf))
    return H[(sym, tf)]


def strat(indicator, tf, stop=5.0, tp=0, **entry):
    return {"version": "01", "entry": {"indicator": indicator, "direction": "long", "timeframe": tf, **entry},
            "stop_loss_pct": 10.0, "stop_atr_mult": stop, "take_profit_r": tp, "position_size_r": 0.5,
            "position_pct": 100, "max_hold_min": 0, "trend_filter": "off", "exit_rsi": 75}


DAILY = {
    "ema1d 50/200": strat("ema_cross", "1d", fast=50, slow=200),
    "ema1d 20/100": strat("ema_cross", "1d", fast=20, slow=100),
    "ma200 1d": strat("ma_regime", "1d", ma=200),
    "tsmom 120 vol0.5 (sub-cuenta momentum)": strat("tsmom", "1d", 3.0, lookback=120, target_vol=0.5),
    "tsmom 250": strat("tsmom", "1d", lookback=250),
}
INTRADAY = {
    "rsi15m (vivo en ETFs)": strat("rsi", "15m", 3.0, 3.0, threshold=25),
    "ema4h 50/200 (tendencia cripto)": strat("ema_cross", "4h", fast=50, slow=200),
    "ema1h 50/200": strat("ema_cross", "1h", fast=50, slow=200),
}


def grid(start):
    t = np.asarray(load("SPY", "1d")["t"], dtype=float) + DAY  # mark each trading day after its close
    return t[(t >= start) & (t < END)]


def curve_on(sim, g):
    t = np.asarray([dt.datetime.fromisoformat(q["ts"]).timestamp() * 1000 for q in sim["equity_curve"]])
    v = np.asarray([q["equity"] for q in sim["equity_curve"]])
    return np.interp(g, t, v, left=np.nan)


def bh_curve(sym, g):
    d = load(sym, "1d")
    return np.interp(g, np.asarray(d["t"], dtype=float) + DAY, np.asarray(d["close"]), left=np.nan)


def rets(eq):
    r = np.zeros(len(eq))
    r[1:] = eq[1:] / eq[:-1] - 1
    return np.where(np.isnan(r), 0.0, r)


def stats(r, g, lo=None, hi=None):
    m = np.ones(len(g), bool) if lo is None else (g >= lo) & (g < hi)
    x = r[m]
    eq = np.cumprod(1 + x)
    sd = np.std(x, ddof=1)
    years = len(x) / 252
    return {"ret": float(eq[-1] - 1) * 100, "cagr": float(eq[-1] ** (1 / years) - 1) * 100 if years > 0.9 else None,
            "dd": float(np.max(1 - eq / np.maximum.accumulate(eq))) * 100,
            "sharpe": float(np.mean(x) / sd * math.sqrt(252)) if sd > 0 else 0.0}


def years_of(g, start, n):
    return [(start + k * 365 * DAY, start + (k + 1) * 365 * DAY) for k in range(n)]


def per_stock(universe, strategies, g):
    """{strategy: {sym: daily returns}} plus buy & hold, and trades per year."""
    out, trades = {"comprar y mantener": {}}, {}
    for sym in universe:
        out["comprar y mantener"][sym] = rets(bh_curve(sym, g))
    for name, s in strategies.items():
        out[name], n = {}, 0
        for sym in universe:
            sim = b.simulate(s, load(sym, s["entry"]["timeframe"]), None, 10000, 0.0, SLIP)
            out[name][sym] = rets(curve_on(sim, g))
            n += len([x for x in sim["trades"] if x["opened_at"] >= b._iso(g[0])])
        trades[name] = n / len(universe) / (len(g) / 252)
        print(name, "done", file=sys.stderr, flush=True)
    return out, trades


def rotation(universe, g, top, filt):
    """Monthly: hold the `top` stocks with the best 12-1 month return (equal weight). filt: "none",
    "abs" (only stocks with a positive 12-1 return; the rest in cash), "spy200" (all cash while SPY
    closes below its 200-day average) or "ma200" (each chosen stock only while it closes above its
    own 200-day average, checked daily: the engine's ma_regime rule on the rotation's picks)."""
    closes = np.vstack([bh_curve(s, g) for s in universe])  # (stocks, days), nan before listing
    spy_d = load("SPY", "1d")
    spy_all = np.asarray(spy_d["close"])
    spy_t = np.asarray(spy_d["t"], dtype=float) + DAY
    full = np.vstack([np.interp(spy_t, np.asarray(load(s, "1d")["t"], dtype=float) + DAY, np.asarray(load(s, "1d")["close"]),
                                left=np.nan) for s in universe])
    idx = np.searchsorted(spy_t, g)  # position of each grid day in the full daily history
    r = np.zeros(len(g))
    picks, held = np.zeros(len(universe)), np.zeros(len(universe))
    month = None
    for k in range(len(g)):
        if k > 0:
            day_r = closes[:, k] / closes[:, k - 1] - 1
            r[k] = float(np.nansum(held * np.where(np.isnan(day_r), 0, day_r)))
        i = idx[k]
        if i < 253:
            continue
        m = dt.datetime.fromtimestamp(g[k] / 1000, UTC).month
        if m != month:  # first trading day of the month: choose at this close
            month = m
            score = full[:, i - 21] / full[:, i - 252] - 1
            order = [j for j in np.argsort(-np.nan_to_num(score, nan=-9)) if not np.isnan(score[j])][:top]
            picks = np.zeros(len(universe))
            for j in order:
                if filt == "abs" and score[j] <= 0:
                    continue
                picks[j] = 1 / top
        new = picks.copy()
        if filt == "spy200" and spy_all[i] < np.mean(spy_all[i - 199:i + 1]):
            new[:] = 0
        if filt == "ma200":
            new = np.where(full[:, i] > np.nanmean(full[:, i - 199:i + 1], axis=1), new, 0)
        r[k] -= float(np.sum(np.abs(new - held))) * SLIP
        held = new
    return r


def study(label, universe, strategies, start, n_years):
    g = grid(start)
    series, trades = per_stock(universe, strategies, g)
    yrs = years_of(g, start, n_years)
    res = {"universe": universe, "trades_per_stock_year": trades, "portfolio": {}, "per_stock": {}, "persistence": {}}
    spy = rets(bh_curve("SPY", g))
    port = {name: np.mean(np.vstack([s[sym] for sym in universe]), axis=0) for name, s in series.items()}
    port["SPY (referencia)"] = spy
    for top in (5, 10):
        for filt in ("none", "abs", "spy200", "ma200"):
            port[f"rotacion top{top} {filt}"] = rotation(universe, g, top, filt)
    for name, r in port.items():
        res["portfolio"][name] = {"all": stats(r, g), "years": [stats(r, g, lo, hi) for lo, hi in yrs]}
        print(label, name, {k: round(v, 2) for k, v in res["portfolio"][name]["all"].items() if v is not None}, file=sys.stderr, flush=True)
    for name, s in series.items():
        res["per_stock"][name] = {sym: stats(s[sym], g) for sym in universe}
    # does "the algorithm worked on this stock" in the first half predict the second half?
    half = g[0] + (g[-1] - g[0]) / 2
    for name in series:
        if name == "comprar y mantener":
            continue
        a, c = [], []
        for sym in universe:
            bh = series["comprar y mantener"][sym]
            a.append(stats(series[name][sym], g, g[0], half)["sharpe"] - stats(bh, g, g[0], half)["sharpe"])
            c.append(stats(series[name][sym], g, half, g[-1] + 1)["sharpe"] - stats(bh, g, half, g[-1] + 1)["sharpe"])
        ra, rc = np.argsort(np.argsort(a)), np.argsort(np.argsort(c))
        res["persistence"][name] = {"spearman": float(np.corrcoef(ra, rc)[0, 1]),
                                    "first_half_edge": dict(zip(universe, a)), "second_half_edge": dict(zip(universe, c))}
    return res


def etfs(start):
    """The ETFs the bot trades, each on its own (SECZ has only traded since 2025: shown from then)."""
    g = grid(start)
    names = {**{k: INTRADAY[k] for k in ("rsi15m (vivo en ETFs)", "ema4h 50/200 (tendencia cripto)")},
             **{k: DAILY[k] for k in ("ema1d 50/200", "ma200 1d", "tsmom 250")}}
    out = {}
    for sym in ("SPY", "QQQ", "VOO", "GLD", "TLT", "USO", "SECZ"):
        gg = g[g >= load(sym, "1d")["t"][0] + 30 * DAY] if sym == "SECZ" else g
        series, _ = per_stock([sym], names if sym != "SECZ" else {k: names[k] for k in ("rsi15m (vivo en ETFs)",)}, gg)
        out[sym] = {name: stats(v[sym], gg) for name, v in series.items()}
        print(sym, {n[:10]: round(x["sharpe"], 2) for n, x in out[sym].items()}, file=sys.stderr, flush=True)
    return out


PARTS = {
    "5y": lambda: study("5y", UNIVERSE_2021, {**INTRADAY, **DAILY}, dt.datetime(2021, 9, 27, tzinfo=UTC).timestamp() * 1000, 5),
    "8y": lambda: study("8y", UNIVERSE_2016, DAILY, dt.datetime(2018, 9, 27, tzinfo=UTC).timestamp() * 1000, 8),
    "etfs": lambda: etfs(dt.datetime(2021, 9, 27, tzinfo=UTC).timestamp() * 1000),
}
out = {part: PARTS[part]() for part in (sys.argv[2:] or PARTS)}
json.dump(out, open(sys.argv[1], "w"), indent=1, default=float)
print("ok", file=sys.stderr)
