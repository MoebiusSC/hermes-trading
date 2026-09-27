"""Monthly momentum rotation (rotation.py) and buy-and-hold ETFs (indicator "hold")."""
import asyncio
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from hermes_trading import backtest, rotation, strategy
from hermes_trading.storage import read_jsonl

DAY = 86400000


class FakeAlpaca:
    """Enough of adapters.alpaca.Alpaca for the rotation: daily bars, quotes, fills at the quote."""

    def __init__(self, growth: dict[str, float], buying_power: float = 1e9, minutes: float = 60, date: str = "2026-10-01"):
        self.growth, self.buying_power, self.minutes, self.date = growth, buying_power, minutes, date
        self.held: dict[str, float] = {}
        self.orders: list[tuple[str, str, float]] = []
        self.price = {s: 100.0 for s in growth}
        self.margin = 1

    async def session(self):
        return {"is_open": True, "date": self.date, "minutes_since_open": self.minutes, "next_open": "2026-10-02T09:30"}

    async def ohlcv(self, sym, tf, limit):
        g = self.growth[sym]
        closes = [100 * (1 + g) ** (i / 260) for i in range(300)]
        return {"t": [1_600_000_000_000 + i * DAY for i in range(300)], "open": closes, "high": closes, "low": closes, "close": closes}

    async def bars(self, sym, limit=1):
        return [(0, self.price[sym])]

    async def positions(self):
        return {s: {"symbol": s, "qty": q, "current_price": self.price[s]} for s, q in self.held.items() if q > 0}

    async def account(self):
        return {"buying_power": str(self.buying_power * self.margin), "cash": str(self.buying_power)}

    async def market_order(self, sym, qty, side, client_order_id):
        self.orders.append((sym, side, qty))
        cost = qty * self.price[sym]
        self.held[sym] = self.held.get(sym, 0) + (qty if side == "buy" else -qty)
        self.buying_power += -cost if side == "buy" else cost
        return {"id": f"o{len(self.orders)}", "qty": qty, "price": self.price[sym]}

    async def wait_filled(self, order_id):
        sym, side, qty = self.orders[int(order_id[1:]) - 1]
        return {"id": order_id, "filled_avg_price": str(self.price[sym]), "filled_qty": str(qty)}


GROWTH = {"AAA": 0.9, "BBB": 0.5, "CCC": 0.3, "DDD": 0.1, "EEE": -0.2}


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.root = Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, self.root)
        self.spec = rotation.settings({"rotation": {"universe": list(GROWTH), "top": 2, "capital": 1000}})

    def run_tick(self, book, broker):
        with mock.patch.object(rotation.alpaca, "client", return_value=broker):
            return asyncio.run(book.tick())

    def test_momentum_skips_the_last_month(self):
        closes = list(range(1, 300))
        self.assertAlmostEqual(rotation.momentum(closes, 252, 21), closes[-22] / closes[-253] - 1)
        self.assertIsNone(rotation.momentum(closes[:100], 252, 21))

    def test_buys_the_top_equal_weight_then_holds_until_next_month(self):
        book, broker = rotation.RotationBook(self.spec, self.root), FakeAlpaca(GROWTH)
        s = self.run_tick(book, broker)
        self.assertTrue(s["decision"].startswith("rebalanced (2026-10): picks AAA, BBB"), s["decision"])
        self.assertEqual(sorted(book.paper["positions"]), ["AAA", "BBB"])
        self.assertAlmostEqual(book.paper["positions"]["AAA"]["qty"] * 100, 500, delta=0.01)
        self.assertAlmostEqual(book.paper["cash"], 0, delta=0.01)
        self.assertEqual(read_jsonl(self.root / "rankings.jsonl")[-1]["picks"], ["AAA", "BBB"])
        orders = len(broker.orders)
        self.assertIn("holding", self.run_tick(book, broker)["decision"])
        self.assertEqual(len(broker.orders), orders)  # same month: no trades

    def test_next_month_sells_the_dropped_pick_and_books_its_pnl(self):
        book, broker = rotation.RotationBook(self.spec, self.root), FakeAlpaca(GROWTH)
        self.run_tick(book, broker)
        broker.date, broker.growth = "2026-11-02", {**GROWTH, "CCC": 2.0, "BBB": -0.5}
        broker.price["BBB"] = 110.0
        s = self.run_tick(book, broker)
        self.assertIn("sold BBB", s["decision"])
        self.assertEqual(sorted(book.paper["positions"]), ["AAA", "CCC"])
        trade = read_jsonl(self.root / "trades.jsonl")[-1]
        self.assertEqual((trade["asset"], trade["exit_reason"]), ("BBB@rotation", "rotation_exit"))
        self.assertAlmostEqual(trade["pnl"], 50, delta=0.01)  # 5 shares, 100 -> 110
        self.assertAlmostEqual(book.paper["equity"], 1050, delta=0.01)

    def test_waits_for_cash_and_buys_when_alpaca_has_it(self):
        book, broker = rotation.RotationBook(self.spec, self.root), FakeAlpaca(GROWTH, buying_power=300)
        s = self.run_tick(book, broker)
        self.assertIn("waiting for cash to buy BBB", s["decision"])
        self.assertAlmostEqual(book.paper["positions"]["AAA"]["qty"] * 100, 294, delta=0.01)  # 98% of what Alpaca had
        broker.buying_power = 5000  # the owner topped up the paper account
        s = self.run_tick(book, broker)
        self.assertIn("bought BBB", s["decision"])
        self.assertEqual(book.paper["pending"], [])

    def test_never_buys_on_margin(self):
        book, broker = rotation.RotationBook(self.spec, self.root), FakeAlpaca(GROWTH, buying_power=300)
        broker.margin = 4  # a margin account reports 4x its cash as buying power
        s = self.run_tick(book, broker)
        self.assertIn("waiting for cash to buy BBB", s["decision"])
        self.assertAlmostEqual(book.paper["positions"]["AAA"]["qty"] * 100, 294, delta=0.01)

    def test_waits_after_the_open_and_skips_symbols_other_books_trade(self):
        book, broker = rotation.RotationBook(self.spec, self.root), FakeAlpaca(GROWTH, minutes=5)
        self.assertIn("waiting", self.run_tick(book, broker)["decision"])
        broker.minutes = 60
        book.exclude = {"AAA"}
        self.run_tick(book, broker)
        self.assertEqual(sorted(book.paper["positions"]), ["BBB", "CCC"])

    def test_a_position_gone_from_alpaca_is_recorded_closed(self):
        book, broker = rotation.RotationBook(self.spec, self.root), FakeAlpaca(GROWTH)
        self.run_tick(book, broker)
        broker.held.pop("AAA")
        self.run_tick(book, broker)
        self.assertNotIn("AAA", book.paper["positions"])
        self.assertEqual(read_jsonl(self.root / "trades.jsonl")[-1]["exit_reason"], "missing_at_broker")

    def test_settings(self):
        self.assertIsNone(rotation.settings({}))
        self.assertIsNone(rotation.settings({"rotation": {"enabled": False, "universe": ["A"]}}))
        with self.assertRaises(ValueError):
            rotation.settings({"rotation": {"universe": ["A"], "top": 5}})

    def test_several_rotations_have_their_own_names_and_universes(self):
        goal = {"rotation": {"universe": ["AAA", "BBB"], "top": 1},
                "rotations": {"europe": {"label": "Europa", "universe": ["CCC", "DDD"], "top": 1},
                              "asia": {"enabled": False, "universe": ["EEE"], "top": 1}}}
        specs = rotation.all_settings(goal)
        self.assertEqual({n: s["label"] for n, s in specs.items()}, {"rotation": "EE. UU.", "rotation-europe": "Europa"})
        self.assertEqual(list(rotation.all_settings({"rotations": {"europe": goal["rotations"]["europe"]}})), ["rotation-europe"])
        with self.assertRaises(ValueError):  # one owner per Alpaca position
            rotation.all_settings({**goal, "rotations": {"europe": {"universe": ["BBB", "CCC"], "top": 1}}})

    def test_a_named_rotation_logs_its_trades_under_its_name(self):
        spec = rotation.all_settings({"rotations": {"europe": {"universe": list(GROWTH), "top": 2, "capital": 1000}}})["rotation-europe"]
        book, broker = rotation.RotationBook(spec, self.root, "rotation-europe"), FakeAlpaca(GROWTH)
        self.assertEqual(book.label, "rotación europe")
        self.run_tick(book, broker)
        broker.date = "2026-11-02"
        broker.growth = {**GROWTH, "CCC": 2.0, "BBB": -0.5}
        self.run_tick(book, broker)
        self.assertEqual(read_jsonl(self.root / "trades.jsonl")[-1]["asset"], "BBB@rotation-europe")

    def test_rotations_share_alpaca_cash_without_margin(self):
        from hermes_trading import loop
        us = rotation.RotationBook(self.spec, self.root / "us", "rotation")
        eu_spec = rotation.settings({"rotation": {"universe": ["X1", "X2"], "top": 2, "capital": 1000}})
        eu = rotation.RotationBook(eu_spec, self.root / "eu", "rotation-europe")
        broker = FakeAlpaca({**GROWTH, "X1": 0.4, "X2": 0.2}, buying_power=1500)
        broker.margin = 4
        worker = loop.Worker.__new__(loop.Worker)
        worker.rotations = [us, eu]
        with mock.patch.object(rotation.alpaca, "client", return_value=broker):
            first, second = asyncio.run(worker._tick_rotations())
        self.assertIn("bought AAA", first["decision"])
        self.assertIn("waiting for cash to buy X2", second["decision"])
        self.assertLessEqual(sum(q * broker.price[s] for s, q in broker.held.items()), 1500)


HOLD = {"version": "01", "entry": {"indicator": "hold", "direction": "long", "timeframe": "1d"}, "stop_loss_pct": 2.0,
        "stop_atr_mult": 3.0, "take_profit_r": 0, "position_size_r": 0.5, "position_pct": 100, "max_hold_min": 0}


class HoldTests(unittest.TestCase):
    def test_hold_is_always_long_without_stop_or_target(self):
        p = strategy.params(HOLD)
        self.assertEqual(strategy.signal_state(p, [1.0, 2.0, 0.5]), 1)
        self.assertEqual(strategy.levels(p, "long", 100.0, 5.0), (0.0, float("inf")))
        with self.assertRaises(ValueError):
            strategy.params({**HOLD, "position_pct": 0})
        with self.assertRaises(ValueError):
            strategy.params({**HOLD, "entry": {**HOLD["entry"], "direction": "both"}})

    def test_after_a_manual_sale_it_stays_out(self):
        from hermes_trading import loop
        book = loop.AssetBook.__new__(loop.AssetBook)
        book.paper = {"equity": 100.0, "position": None, "blocked": "long"}
        sig = {"p": strategy.params(HOLD), "target": "long", "rsi": 50.0, "trend": None}
        self.assertEqual(book._entry_side(sig), (None, "sold by hand: buys again with a manual buy"))
        book.paper.pop("blocked")
        self.assertEqual(book._entry_side(sig), ("long", ""))

    def test_backtest_buys_once_and_rides_a_crash(self):
        closes = np.concatenate([np.linspace(100, 150, 60), np.linspace(150, 60, 40), np.linspace(60, 120, 60)])
        c = {"t": [i * DAY for i in range(len(closes))], "open": list(closes), "high": list(closes * 1.01),
             "low": list(closes * 0.99), "close": list(closes)}
        sim = backtest.simulate(HOLD, c, None, 10000, 0.0, 0.0)
        self.assertEqual(len(sim["trades"]), 1)
        self.assertEqual(sim["trades"][0]["exit_reason"], "end_of_test")


if __name__ == "__main__":
    unittest.main()
