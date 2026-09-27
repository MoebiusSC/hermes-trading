"""Stock candles: all pages of Alpaca's history, split-adjusted, and volatility on a trading-day calendar."""
import asyncio
import unittest
from unittest import mock

import numpy as np

from hermes_trading import strategy
from hermes_trading.adapters import alpaca


class PagingTests(unittest.TestCase):
    def test_follows_pages_until_the_limit_and_asks_for_adjusted_bars(self):
        with mock.patch.dict("os.environ", {"ALPACA_API_KEY": "k", "ALPACA_API_SECRET": "s"}):
            client = alpaca.Alpaca()
        # newest first, as with sort=desc; 3 pages of 50 bars
        bars = [{"t": f"2026-01-{1 + i // 24:02d}T{i % 24:02d}:00:00Z", "o": 1, "h": 1, "l": 1, "c": float(i)} for i in range(150)][::-1]
        calls = []

        async def req(method, url, params):
            calls.append(params)
            page = int(params.get("page_token", 0))
            chunk = bars[page * 50:(page + 1) * 50]
            return {"bars": chunk, "next_page_token": str(page + 1) if page < 2 else None}

        client._req = req
        candles = asyncio.run(client.ohlcv("MSFT", "1h", 120))
        self.assertEqual(len(candles["t"]), 120)
        self.assertEqual(candles["close"][-1], 149.0)  # oldest first, ending with the newest bar
        self.assertEqual(candles["close"][0], 30.0)
        self.assertTrue(all(c["adjustment"] == "all" for c in calls))
        self.assertEqual(len(calls), 3)


class VolatilityTests(unittest.TestCase):
    def test_counts_candles_per_year_from_their_times(self):
        rng = np.random.default_rng(1)
        closes = 100 * np.cumprod(1 + rng.normal(0, 0.01, 400))
        crypto_t = [i * 86400000 for i in range(400)]                          # every day
        stock_t = [(i // 5 * 7 + i % 5) * 86400000 for i in range(400)]        # weekdays only
        v24 = strategy.realized_vol_series(closes, "1d", t=crypto_t)[-1]
        v_stock = strategy.realized_vol_series(closes, "1d", t=stock_t)[-1]
        self.assertAlmostEqual(v24, strategy.realized_vol_series(closes, "1d")[-1], places=6)
        self.assertAlmostEqual(v_stock / v24, np.sqrt(5 / 7), places=2)


class BadPrintTests(unittest.TestCase):
    def test_impossible_wicks_are_cut_back_and_real_moves_kept(self):
        candles = {"t": [0, 1, 2, 3], "open": [690.0, 689.0, 690.0, 600.0], "high": [692.0, 691.5, 693.0, 612.0],
                   "low": [688.0, 68.47, 688.0, 590.0], "close": [689.0, 690.0, 691.0, 605.0]}
        out = strategy.clip_wicks(candles, strategy.stock_max_wick("1d"))
        self.assertEqual(out["low"][1], 689.0)   # SPY 2 Feb 2026: a low of 68 near 690
        self.assertEqual(out["low"][3], 590.0)   # a 13% gap down is a real move, kept
        self.assertEqual(out["high"], candles["high"])
        self.assertEqual(strategy.stock_max_wick("15m"), 0.08)


if __name__ == "__main__":
    unittest.main()
