"""Fixed exposure per position (position_pct) and applying it to many assets at once."""
import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hermes_trading import config, reflect, strategy
from hermes_trading.storage import dump_yaml, load_yaml, read_jsonl

BASE = {"version": "01", "entry": {"indicator": "rsi", "threshold": 25, "direction": "long", "timeframe": "15m"},
        "exit_rsi": 75, "trend_filter": "off", "stop_loss_pct": 2.0, "stop_atr_mult": 3.0,
        "take_profit_r": 3.0, "position_size_r": 0.5, "max_hold_min": 0}


class SizeTests(unittest.TestCase):
    def test_fixed_exposure_ignores_the_stop_distance(self):
        p = strategy.params({**BASE, "position_pct": 50})
        for dist in (0.1, 2.0, 30.0):
            self.assertAlmostEqual(strategy.size(p, 10000, 100, dist), 50)

    def test_zero_keeps_risk_based_sizing(self):
        p = strategy.params({**BASE, "position_pct": 0})
        self.assertAlmostEqual(strategy.size(p, 10000, 100, 2), 25)  # 0.5% of 10k / $2 stop

    def test_out_of_range_is_refused(self):
        with self.assertRaises(ValueError):
            strategy.params({**BASE, "position_pct": 150})


class ManualOnlyTests(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.dir)
        self.paths = config.AssetPaths("BTC/USDT", self.dir)
        dump_yaml(self.paths.strategy, BASE)

    def test_owner_can_set_it(self):
        records = reflect.apply_manual(self.paths, {"position_pct": 60}, stock=False)
        self.assertEqual(records[0]["old_value"], 0.0)
        self.assertEqual(load_yaml(self.paths.strategy)["position_pct"], 60)

    def test_bounds_apply_to_manual_edits(self):
        with self.assertRaises(ValueError):
            reflect.apply_manual(self.paths, {"position_pct": 120}, stock=False)

    def test_the_ai_cannot_change_it(self):
        self.assertNotIn("position_pct", reflect.TUNABLE)
        with self.assertRaises(ValueError):
            reflect._bounded(BASE, {"variable": "position_pct", "new_value": 80})


class BulkTests(unittest.TestCase):
    def test_applies_to_one_kind_only(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)

        def ensure(asset):
            paths = config.AssetPaths(asset, root / config.asset_slug(asset))
            paths.history.mkdir(parents=True, exist_ok=True)
            if not paths.strategy.exists():
                dump_yaml(paths.strategy, BASE)
            return paths

        from hermes_trading import loop
        with mock.patch.object(config, "ensure_asset_state", side_effect=ensure):
            w = loop.Worker(["BTC/USDT", "ETH/USDT", "SPY"], {"stock_equity_per_asset": 200})
            message = asyncio.run(w.set_strategy_all({"position_pct": 40}, "crypto"))
        self.assertIn("2 activos actualizados", message)
        self.assertEqual(load_yaml(root / "BTC-USDT" / "strategy.yaml")["position_pct"], 40)
        self.assertEqual(load_yaml(root / "ETH-USDT" / "strategy.yaml")["position_pct"], 40)
        self.assertNotIn("position_pct", load_yaml(root / "SPY" / "strategy.yaml"))
        self.assertEqual(read_jsonl(root / "BTC-USDT" / "hypotheses.jsonl")[-1]["mode"], "manual")


if __name__ == "__main__":
    unittest.main()
