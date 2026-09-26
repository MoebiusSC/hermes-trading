"""Trend following (ema_cross): flipping at each cross, re-entry after a stop, no target, stop floor."""
import asyncio
import math
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from hermes_trading import backtest, config, strategy
from hermes_trading.storage import dump_yaml, read_jsonl

EMA = {"version": "01", "entry": {"indicator": "ema_cross", "direction": "both", "fast": 5, "slow": 20, "timeframe": "4h"},
       "stop_loss_pct": 2.0, "stop_atr_mult": 3.0, "take_profit_r": 0, "position_size_r": 0.5, "position_pct": 50,
       "max_hold_min": 0}


def candles(closes, start=1_700_000_000_000, step=4 * 3600 * 1000):
    c = np.asarray(closes, dtype=float)
    return {"t": [start + i * step for i in range(len(c))], "open": list(c), "high": list(c * 1.002),
            "low": list(c * 0.998), "close": list(c)}


class RuleTests(unittest.TestCase):
    def test_both_follows_the_cross(self):
        p = strategy.params(EMA)
        self.assertEqual(strategy.target_direction(p, 1), "long")
        self.assertEqual(strategy.target_direction(p, -1), "short")
        long_only = strategy.params({**EMA, "entry": {**EMA["entry"], "direction": "long"}})
        self.assertIsNone(strategy.target_direction(long_only, -1))

    def test_rsi_cannot_be_both_and_fast_must_be_below_slow(self):
        with self.assertRaises(ValueError):
            strategy.params({**EMA, "entry": {"indicator": "rsi", "direction": "both", "threshold": 30}})
        with self.assertRaises(ValueError):
            strategy.params({**EMA, "entry": {**EMA["entry"], "fast": 30, "slow": 20}})

    def test_no_target_and_stop_floor(self):
        p = strategy.params(EMA)
        stop, target = strategy.levels(p, "long", 100, 2)
        self.assertEqual((stop, target), (98, math.inf))
        pos = {"direction": "long", "stop": 98, "target": None}
        self.assertEqual(strategy.exit_reason(pos, p, 150, 50, 0, "long"), None)
        self.assertEqual(strategy.exit_reason(pos, p, 150, 50, 0, "short"), "signal_exit")
        frac = strategy.min_stop_frac({"costs": {"crypto": {"fee_pct": 0.1, "slippage_pct": 0.02}}}, False)
        self.assertAlmostEqual(frac, 3 * 2 * 0.0012)
        self.assertAlmostEqual(strategy.stop_distance(p, 100, 0.01, frac), 0.72)


class BacktestTests(unittest.TestCase):
    def test_goes_long_in_the_rise_and_short_in_the_fall(self):
        closes = [100 + i for i in range(60)] + [160 - 2 * i for i in range(60)]
        sim = backtest.simulate(EMA, candles(closes), None, 10000, 0.001, 0.0002)
        sides = [t["direction"] for t in sim["trades"]]
        self.assertEqual(sides[:2], ["long", "short"])
        self.assertEqual(sim["trades"][0]["exit_reason"], "signal_exit")
        self.assertGreater(sim["trades"][1]["pnl"], 0)  # the short made money in the fall

    def test_no_reentry_on_the_side_just_stopped_out(self):
        # a rise, a crash through the long stop while EMAs still point up, then more rise
        closes = [100 + i for i in range(40)] + [100] + [101 + i * 0.1 for i in range(10)]
        sim = backtest.simulate({**EMA, "stop_atr_mult": 0.5}, candles(closes), None, 10000, 0.001, 0.0002)
        reasons = [t["exit_reason"] for t in sim["trades"]]
        self.assertIn("stop_loss", reasons)
        after = sim["trades"][reasons.index("stop_loss") + 1:]
        self.assertFalse([t for t in after if t["direction"] == "long" and t["exit_reason"] != "end_of_test"])


class WorkerFlipTests(unittest.TestCase):
    def test_closes_long_and_opens_short_in_one_tick(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)

        def ensure(asset):
            paths = config.AssetPaths(asset, root / config.asset_slug(asset))
            paths.history.mkdir(parents=True, exist_ok=True)
            dump_yaml(paths.strategy, EMA)
            return paths

        from hermes_trading import loop
        with mock.patch.object(config, "ensure_asset_state", side_effect=ensure):
            book = loop.make_book("BTC/USDT", {"costs": {"crypto": {"fee_pct": 0.1, "slippage_pct": 0.02}}})
        book._save_paper = lambda: None
        p = strategy.params(EMA)
        long_sig = {"p": p, "rsi": 60.0, "atr": 1.0, "trend": None, "state": 1, "target": "long"}
        self.assertEqual(asyncio.run(book.decide(EMA, 100.0, long_sig, {})), "opened long")
        self.assertIsNone(book.paper["position"]["target"])  # no target, stored as None (valid JSON)
        short_sig = {**long_sig, "state": -1, "target": "short"}
        self.assertEqual(asyncio.run(book.decide(EMA, 101.0, short_sig, {})), "closed long (signal_exit); opened short")
        self.assertEqual(book.paper["position"]["direction"], "short")
        self.assertEqual(read_jsonl(book.paths.trades)[-1]["exit_reason"], "signal_exit")


if __name__ == "__main__":
    unittest.main()
