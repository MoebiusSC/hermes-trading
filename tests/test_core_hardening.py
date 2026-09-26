import unittest

import numpy as np

from hermes_trading import score, strategy


class StrategyTests(unittest.TestCase):
    def base(self):
        return {
            "version": "01",
            "entry": {"indicator": "rsi", "direction": "long", "threshold": 25, "timeframe": "1m"},
            "exit_rsi": 70, "trend_filter": "off", "stop_loss_pct": 2,
            "stop_atr_mult": 0, "take_profit_r": 2, "position_size_r": 1, "max_hold_min": 0,
        }

    def test_params_reject_invalid_risk_values(self):
        bad = self.base(); bad["position_size_r"] = 3
        with self.assertRaises(ValueError):
            strategy.params(bad)

    def test_size_is_never_leveraged_and_validates_inputs(self):
        p = strategy.params(self.base())
        self.assertAlmostEqual(strategy.size(p, 10000, 100, 2), 50)
        self.assertEqual(strategy.size(p, 0, 100, 2), 0)
        with self.assertRaises(ValueError):
            strategy.size(p, 10000, 0, 2)
        with self.assertRaises(ValueError):
            strategy.size(p, 10000, 100, 0)


class ScoreTests(unittest.TestCase):
    def trades(self):
        return [
            {"pnl_pct": 0.01, "opened_at": "2026-01-01T00:00:00+00:00", "closed_at": "2026-01-01T01:00:00+00:00"},
            {"pnl_pct": -0.005, "opened_at": "2026-01-01T02:00:00+00:00", "closed_at": "2026-01-01T03:00:00+00:00"},
        ]

    def test_equity_curve_drawdown_uses_intratrade_peak_to_trough(self):
        trades = self.trades()
        curve = [
            {"ts": "2026-01-01T00:00:00+00:00", "equity": 10000},
            {"ts": "2026-01-01T01:00:00+00:00", "equity": 11000},
            {"ts": "2026-01-01T02:00:00+00:00", "equity": 9900},
            {"ts": "2026-01-01T03:00:00+00:00", "equity": 10050},
        ]
        m = score.metrics(trades, curve)
        self.assertAlmostEqual(m["max_drawdown"], 1100 / 11000, places=6)
        self.assertAlmostEqual(m["realised_return"], 0.005, places=6)


if __name__ == "__main__":
    unittest.main()
