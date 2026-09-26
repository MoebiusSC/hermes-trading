"""Score closed trades against goal.yaml. `python -m hermes_trading.score` prints each asset's score."""
from __future__ import annotations

import datetime as dt
import json
import math

import numpy as np

from . import config
from .storage import load_yaml, read_jsonl

SHARPE_MIN_SPAN_DAYS = 30  # annualise over at least the goal horizon so tiny samples don't explode


def _clip(x: float, lo: float = -1.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, x))


def _curve_metrics(equity_curve: list[dict]) -> tuple[float, float, float]:
    """Return return, mark-to-market max drawdown and annualised Sharpe."""
    if len(equity_curve) < 2:
        return 0.0, 0.0, 0.0
    points = sorted((
        dt.datetime.fromisoformat(p["ts"]), float(p["equity"])
    ) for p in equity_curve)
    values = np.asarray([v for _, v in points], dtype=float)
    if not np.all(np.isfinite(values)) or values[0] <= 0:
        return 0.0, 0.0, 0.0
    peaks = np.maximum.accumulate(values)
    max_dd = float(np.max(np.maximum(0.0, 1.0 - values / peaks)))
    returns = values[1:] / values[:-1] - 1.0
    sharpe = 0.0
    if len(returns) >= 2:
        std = returns.std(ddof=1)
        # Annualise by the bars actually observed per calendar year, so stocks (bars only in market
        # hours) aren't scaled as if they traded around the clock like crypto.
        years = (points[-1][0] - points[0][0]).total_seconds() / (365.25 * 86400)
        if std > 0 and years > 0:
            periods_per_year = len(returns) / years
            sharpe = float(returns.mean() / std * math.sqrt(periods_per_year))
    return float(values[-1] / values[0] - 1.0), max_dd, sharpe

def metrics(trades: list[dict], equity_curve: list[dict] | None = None) -> dict:
    """Use mark-to-market equity when supplied; trade returns remain the win-rate source."""
    rets = np.array([float(t["pnl_pct"]) for t in trades], dtype=float)
    n = len(rets)
    if n == 0:
        return {"n": 0, "realised_return": 0.0, "max_drawdown": 0.0, "sharpe": 0.0, "win_rate": 0.0}
    if equity_curve:
        realised_return, max_dd, sharpe = _curve_metrics(equity_curve)
    else:
        curve = np.cumprod(1 + rets)
        peaks = np.maximum.accumulate(np.concatenate([[1.0], curve]))[1:]
        max_dd = float(max(0.0, np.max(1 - curve / peaks)))
        std = rets.std(ddof=1) if n >= 2 else 0.0
        sharpe = 0.0
        if std > 0:
            start = dt.datetime.fromisoformat(trades[0]["opened_at"])
            end = dt.datetime.fromisoformat(trades[-1]["closed_at"])
            span_days = (end - start).total_seconds() / 86400
            sharpe = float(rets.mean() / std * math.sqrt(n * 365 / max(span_days, SHARPE_MIN_SPAN_DAYS)))
        realised_return = float(curve[-1] - 1)
    return {"n": n, "realised_return": round(realised_return, 6), "max_drawdown": round(max_dd, 6),
            "sharpe": round(sharpe, 4), "win_rate": round(float((rets > 0).mean()), 4)}

def score(trades: list[dict], goal: dict) -> float:
    """Composite in [-1, +1]: 40% return vs target, 30% drawdown vs max, 30% Sharpe vs min."""
    m = metrics(trades)
    if m["n"] == 0:
        return 0.0
    ret_c = _clip(m["realised_return"] / float(goal["target_return_30d"]))
    dd_c = _clip(1 - m["max_drawdown"] / float(goal["max_drawdown"]))
    sharpe_c = _clip(m["sharpe"] / float(goal["min_sharpe"]))
    s = 0.4 * ret_c + 0.3 * dd_c + 0.3 * sharpe_c

    floor = float(goal.get("failure_below", -0.04))
    if m["realised_return"] < floor:
        s = min(s, -0.5) - 10 * (floor - m["realised_return"])
    return round(_clip(s), 4)


def main() -> None:
    goal = load_yaml(config.GOAL_FILE)
    report = {}
    for asset in config.goal_assets(goal):
        trades = read_jsonl(config.asset_paths(asset).trades)
        report[asset] = {**metrics(trades), "score": score(trades, goal)}
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
