"""Costs per side of the book (longs on spot, shorts on a perpetual) and perpetual funding."""
import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from hermes_trading import backtest, config, strategy
from hermes_trading.storage import dump_yaml, read_jsonl

OKX = {"costs": {"crypto": {"venue": "okx",
                            "long": {"fee_pct": 0.1, "slippage_pct": 0.02},
                            "short": {"fee_pct": 0.05, "slippage_pct": 0.02, "funding": True}}}}

EMA = {"version": "01", "entry": {"indicator": "ema_cross", "direction": "both", "timeframe": "4h", "fast": 50, "slow": 200},
       "stop_loss_pct": 3.0, "stop_atr_mult": 5.0, "take_profit_r": 0, "position_size_r": 0.5,
       "position_pct": 100, "max_hold_min": 0}


def ma_short(ma=10):
    return {"version": "01", "entry": {"indicator": "ma_regime", "direction": "short", "timeframe": "1d", "ma": ma},
            "stop_loss_pct": 10.0, "stop_atr_mult": 0, "take_profit_r": 0, "position_size_r": 0.5,
            "position_pct": 100, "max_hold_min": 0}


def candles(closes, step=86400000):
    c = np.asarray(closes, dtype=float)
    return {"t": [1_700_000_000_000 + i * step for i in range(len(c))], "open": list(c), "high": list(c * 1.001),
            "low": list(c * 0.999), "close": list(c)}


class CostModelTests(unittest.TestCase):
    def test_costs_per_side_and_funding_flag(self):
        self.assertEqual(strategy.costs(OKX, False, "long"), (0.001, 0.0002))
        self.assertEqual(strategy.costs(OKX, False, "short"), (0.0005, 0.0002))
        self.assertTrue(strategy.pays_funding(OKX, False, "short"))
        self.assertFalse(strategy.pays_funding(OKX, False, "long"))
        self.assertEqual(strategy.funding_venue(OKX), "okx")

    def test_flat_costs_still_apply_to_both_sides(self):
        flat = {"costs": {"crypto": {"fee_pct": 0.2, "slippage_pct": 0.05}}}
        self.assertEqual(strategy.costs(flat, False, "long"), strategy.costs(flat, False, "short"))
        self.assertAlmostEqual(strategy.costs(flat, False)[0], 0.002)
        self.assertFalse(strategy.pays_funding(flat, False, "short"))
        self.assertEqual(strategy.costs({}, True), (0.0, 0.0))


class BacktestFundingTests(unittest.TestCase):
    closes = [100.0] * 15 + [100 - i for i in range(1, 40)]

    def test_short_pays_its_own_fee_and_receives_positive_funding(self):
        c = candles(self.closes)
        rates = [[t + 3600000, 0.001] for t in c["t"]]  # one settlement per bar
        plain = backtest.simulate(ma_short(), c, None, 10000, 0.001, 0.0002, 0.0, (0.0005, 0.0002))
        funded = backtest.simulate(ma_short(), c, None, 10000, 0.001, 0.0002, 0.0, (0.0005, 0.0002), rates)
        t0, t1 = plain["trades"][0], funded["trades"][0]
        self.assertEqual(t0["direction"], "short")
        qty = t0["gross_pnl"] / (t0["entry_price"] - t0["exit_price"])
        notional = qty * (t0["entry_price"] + t0["exit_price"])
        self.assertAlmostEqual(t0["fees"], notional * 0.0005, places=6)  # the short's (perp) fee, not the long's
        self.assertEqual(t0["funding"], 0.0)
        self.assertGreater(t1["funding"], 0)  # a short receives a positive rate
        self.assertAlmostEqual(t1["pnl"], t1["gross_pnl"] - t1["fees"] + t1["funding"], places=6)

    def test_negative_funding_costs_the_short_and_longs_are_unfunded_by_default(self):
        c = candles(self.closes)
        rates = [[t + 3600000, -0.001] for t in c["t"]]
        sim = backtest.simulate(ma_short(), c, None, 10000, 0.001, 0.0002, 0.0, (0.0005, 0.0002), rates)
        self.assertLess(sim["trades"][0]["funding"], 0)
        up = candles([100.0] * 15 + [100 + i for i in range(1, 40)])
        long = {**ma_short(), "entry": {**ma_short()["entry"], "direction": "long"}}
        sim = backtest.simulate(long, up, None, 10000, 0.001, 0.0002, 0.0, (0.0005, 0.0002), rates)
        self.assertEqual(sim["trades"][0]["funding"], 0.0)


class LiveFundingTests(unittest.TestCase):
    def setUp(self):
        root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, root)

        def ensure(asset):
            paths = config.AssetPaths(asset, root / config.asset_slug(asset))
            paths.history.mkdir(parents=True, exist_ok=True)
            dump_yaml(paths.strategy, EMA)
            return paths

        from hermes_trading import loop
        self.loop = loop
        with mock.patch.object(config, "ensure_asset_state", side_effect=ensure):
            self.book = loop.make_book("BTC/USDT", OKX)
        self.book._save_paper = lambda: None

    def test_short_accrues_each_settlement_once_and_books_it_on_close(self):
        book, p = self.book, strategy.params(EMA)
        sig = {"p": p, "rsi": 40.0, "atr": 1.0, "trend": None, "state": -1, "target": "short"}
        self.assertEqual(asyncio.run(book.decide(EMA, 100.0, sig, {})), "opened short")
        pos = book.paper["position"]
        self.assertAlmostEqual(pos["fees"], pos["qty"] * pos["entry_price"] * 0.0005, places=5)
        opened = pos["opened_ms"]
        rates = [[opened - 1000, 0.5], [opened + 1000, 0.0002], [opened + 2000, 0.0003]]

        async def fake(symbol, venue, since):
            self.assertEqual((symbol, venue), ("BTC/USDT", "okx"))
            return [r for r in rates if r[0] >= since]

        with mock.patch.object(self.loop.price, "funding", side_effect=fake):
            asyncio.run(book._accrue_funding(100.0))
            asyncio.run(book._accrue_funding(100.0))  # nothing new: not counted twice
        expected = pos["qty"] * 100.0 * 0.0005
        self.assertAlmostEqual(pos["funding"], expected, places=5)
        long_sig = {**sig, "state": 1, "target": "long"}
        asyncio.run(book.decide(EMA, 100.0, long_sig, {}))
        trade = read_jsonl(book.paths.trades)[-1]
        self.assertAlmostEqual(trade["funding"], expected, places=4)
        self.assertAlmostEqual(trade["pnl"], trade["gross_pnl"] - trade["fees"] + trade["funding"], places=3)
        self.assertEqual(book.paper["position"]["direction"], "long")
        with mock.patch.object(self.loop.price, "funding", side_effect=AssertionError("spot longs pay no funding")):
            asyncio.run(book._accrue_funding(100.0))


if __name__ == "__main__":
    unittest.main()
