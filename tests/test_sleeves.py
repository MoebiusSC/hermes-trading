"""Combined operation: extra strategies per pair ("BTC/USDT@momentum") with their own accounts."""
import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from hermes_trading import config, run
from hermes_trading.storage import dump_yaml, load_yaml, read_jsonl

TEMPLATE = {"version": "01", "entry": {"indicator": "ema_cross", "direction": "both", "fast": 50, "slow": 200, "timeframe": "4h"},
            "stop_loss_pct": 2.0, "stop_atr_mult": 3.0, "take_profit_r": 0, "position_size_r": 0.5, "position_pct": 30,
            "max_hold_min": 0}
GOAL = """assets:
  - "BTC/USDT"
  - "SPY"
reflection_every: 10
sleeves:
  momentum:
    kinds: [crypto]
    strategy:
      entry.indicator: tsmom
      entry.timeframe: "1d"
      entry.lookback: 60
      entry.target_vol: 0.5
    note: test
strategy_migrations:
  - id: main-only
    kinds: [crypto]
    changes: {stop_atr_mult: 4.0}
  - id: sleeve-only
    kinds: [crypto]
    sleeve: momentum
    changes: {entry.lookback: 90}
"""


class IdTests(unittest.TestCase):
    def test_symbol_and_sleeve(self):
        self.assertEqual(config.symbol("BTC/USDT@momentum"), "BTC/USDT")
        self.assertEqual(config.sleeve("BTC/USDT@momentum"), "momentum")
        self.assertIsNone(config.sleeve("BTC/USDT"))
        self.assertFalse(config.is_stock("BTC/USDT@momentum"))
        self.assertEqual(config.asset_slug("BTC/USDT@momentum"), "BTC-USDT@momentum")


class BootTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        (self.root / "goal.yaml").write_text(GOAL, encoding="utf-8")
        dump_yaml(self.root / "strategy.template.yaml", TEMPLATE)
        for name, value in (("STATE", self.root), ("GOAL_FILE", self.root / "goal.yaml"),
                            ("STRATEGY_TEMPLATE", self.root / "strategy.template.yaml"),
                            ("STRATEGY_TEMPLATE_CRYPTO", self.root / "strategy.template.crypto.yaml"),
                            ("ASSETS_DIR", self.root / "assets")):
            patcher = mock.patch.object(config, name, value)
            patcher.start()
            self.addCleanup(patcher.stop)
        for asset in ("BTC/USDT", "SPY"):
            config.ensure_asset_state(asset)

    def boot(self):
        goal = load_yaml(config.GOAL_FILE)
        run.apply_migrations(goal, config.goal_assets(goal))
        goal = run.ensure_sleeves(goal)
        run.apply_migrations(goal, config.goal_assets(goal))
        return goal

    def test_creates_the_sleeve_for_crypto_only_and_scopes_migrations(self):
        goal = self.boot()
        self.assertEqual(config.goal_assets(goal), ["BTC/USDT", "SPY", "BTC/USDT@momentum"])
        main = load_yaml(config.asset_paths("BTC/USDT").strategy)
        sleeve = load_yaml(config.asset_paths("BTC/USDT@momentum").strategy)
        self.assertEqual(main["entry"]["indicator"], "ema_cross")
        self.assertEqual(main["stop_atr_mult"], 4.0)            # main-only migration
        self.assertEqual(sleeve["entry"]["indicator"], "tsmom")
        self.assertEqual(sleeve["entry"]["lookback"], 90)       # sleeve-only migration
        self.assertEqual(sleeve["stop_atr_mult"], 3.0)          # untouched by the main migration
        modes = {h["mode"] for h in read_jsonl(config.asset_paths("BTC/USDT@momentum").hypotheses)}
        self.assertEqual(modes, {"migration"})

    def test_second_boot_changes_nothing(self):
        self.boot()
        before = config.asset_paths("BTC/USDT@momentum").strategy.read_text()
        goal = self.boot()
        self.assertEqual(config.goal_assets(goal).count("BTC/USDT@momentum"), 1)
        self.assertEqual(config.asset_paths("BTC/USDT@momentum").strategy.read_text(), before)

    def test_the_sleeve_book_fetches_its_market(self):
        self.boot()
        from hermes_trading import loop
        book = loop.make_book("BTC/USDT@momentum", {})
        seen = []

        async def ohlcv(symbol, tf, limit):
            seen.append(symbol)
            closes = [100 + i for i in range(limit)]
            return {"t": [i * 86400000 for i in range(limit)], "open": closes, "high": closes, "low": closes, "close": closes}

        book.OHLCV = ohlcv
        sig = asyncio.run(book.signals(load_yaml(book.paths.strategy)))
        self.assertEqual(seen, ["BTC/USDT"])
        self.assertEqual(sig["target"], "long")


if __name__ == "__main__":
    unittest.main()
