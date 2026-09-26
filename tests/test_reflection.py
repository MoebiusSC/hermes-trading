"""Reflection validation: the backtest gate, the out-of-sample minimum and duplicate hypotheses."""
import datetime as dt
import json
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hermes_trading import backtest, config, reflect, score
from hermes_trading.storage import dump_yaml

GOAL = {"reflection_every": 5, "target_return_30d": 0.05, "max_drawdown": 0.08, "min_sharpe": 1.2, "failure_below": -0.04}
STRATEGY = {"version": "01", "entry": {"indicator": "rsi", "threshold": 25, "direction": "long", "timeframe": "15m"},
            "exit_rsi": 75, "trend_filter": "off", "stop_loss_pct": 2.0, "stop_atr_mult": 3.0,
            "take_profit_r": 3.0, "position_size_r": 0.5, "max_hold_min": 0}


def fake_result(n_all, oos_n, oos_score, all_score, ret=0.0):
    part = lambda n, s: {"n": n, "score": s, "return_pct": ret}  # noqa: E731
    return {"from": "2026-06-01T00:00:00+00:00", "to": "2026-09-01T00:00:00+00:00", "split": "2026-08-05T00:00:00+00:00",
            "buy_hold_pct": 0.0, "all": part(n_all, all_score), "in_sample": part(n_all - oos_n, all_score),
            "out_of_sample": part(oos_n, oos_score)}


class ReflectionGateTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.paths = config.AssetPaths("BTC/USDT", self.dir)
        dump_yaml(self.paths.strategy, STRATEGY)
        t0 = dt.datetime(2026, 9, 1, tzinfo=dt.timezone.utc)
        with open(self.paths.trades, "w") as f:  # losing trades, so the fallback rule proposes a change
            for i in range(6):
                o, c = t0 + dt.timedelta(hours=2 * i), t0 + dt.timedelta(hours=2 * i + 1)
                f.write(json.dumps({"opened_at": o.isoformat(), "closed_at": c.isoformat(), "pnl": -5.0, "pnl_pct": -0.0005}) + "\n")
        patcher = mock.patch.object(reflect.config, "asset_paths", return_value=self.paths)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.addCleanup(shutil.rmtree, self.dir)

    def run_with(self, candidate):
        baseline = fake_result(30, 10, 0.10, 0.10)

        def fake_run(asset, strategy, goal, days=backtest.DEFAULT_DAYS):
            self.assertEqual(days, reflect.VALIDATION_DAYS)
            return baseline if strategy["entry"]["threshold"] == STRATEGY["entry"]["threshold"] else candidate

        with mock.patch.object(backtest, "run", side_effect=fake_run):
            return reflect.propose("BTC/USDT", GOAL, "fallback", force=False)

    def test_too_few_out_of_sample_trades_is_rejected(self):
        p = self.run_with(fake_result(30, reflect.MIN_OOS_TRADES - 1, 0.9, 0.9))
        self.assertTrue(p.rejected)
        self.assertEqual(p.backtest["reason"], "insufficient_oos_trades")
        self.assertIn("fewer than", reflect.apply_proposal(p, "fallback"))

    def test_better_candidate_with_enough_trades_is_accepted(self):
        p = self.run_with(fake_result(30, reflect.MIN_OOS_TRADES, 0.2, 0.3))
        self.assertFalse(p.rejected)
        self.assertEqual(p.backtest["verdict"], "accepted")

    def test_worse_out_of_sample_is_rejected(self):
        p = self.run_with(fake_result(30, 12, 0.05, 0.3))
        self.assertTrue(p.rejected)

    def test_recently_tried_value_is_not_tested_again(self):
        first = self.run_with(fake_result(30, 12, 0.05, 0.3))
        reflect.apply_proposal(first, "fallback")  # logged as rejected
        with open(self.paths.trades, "a") as f:  # enough trades after that rejection for the next cycle
            for i in range(6):
                o = dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=i + 1)
                f.write(json.dumps({"opened_at": o.isoformat(), "closed_at": (o + dt.timedelta(minutes=30)).isoformat(),
                                    "pnl": -5.0, "pnl_pct": -0.0005}) + "\n")
        again = self.run_with(fake_result(30, 12, 0.9, 0.9))
        self.assertTrue(again.rejected)
        self.assertEqual(again.backtest["reason"], "duplicate")


class SharpeAnnualisationTests(unittest.TestCase):
    def test_uses_observed_bars_per_year(self):
        # Two bars a day (like a market that's only open part of the day) for 20 days.
        start, curve, v = dt.datetime(2026, 1, 1, 15, tzinfo=dt.timezone.utc), [], 10000.0
        for d in range(20):
            for h in (0, 6):
                v *= 1.001 if (d + h) % 3 else 0.9995
                curve.append({"ts": (start + dt.timedelta(days=d, hours=h)).isoformat(), "equity": v})
        _, _, sharpe = score._curve_metrics(curve)
        values = [p["equity"] for p in curve]
        rets = [b / a - 1 for a, b in zip(values, values[1:])]
        mean = sum(rets) / len(rets)
        std = math.sqrt(sum((r - mean) ** 2 for r in rets) / (len(rets) - 1))
        years = (19 * 86400 + 6 * 3600) / (365.25 * 86400)
        self.assertAlmostEqual(sharpe, mean / std * math.sqrt(len(rets) / years), places=6)


if __name__ == "__main__":
    unittest.main()
