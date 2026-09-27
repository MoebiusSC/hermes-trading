"""Time-series momentum and moving-average regime signals, and volatility-scaled sizing."""
import unittest

import numpy as np

from hermes_trading import backtest, strategy


def strat(indicator, **entry):
    return {"version": "01", "entry": {"indicator": indicator, "direction": "long", "timeframe": "1d", **entry},
            "stop_loss_pct": 2.0, "stop_atr_mult": 3.0, "take_profit_r": 0, "position_size_r": 0.5,
            "position_pct": 100, "max_hold_min": 0}


def candles(closes, step=86400000):
    c = np.asarray(closes, dtype=float)
    return {"t": [1_700_000_000_000 + i * step for i in range(len(c))], "open": list(c), "high": list(c * 1.001),
            "low": list(c * 0.999), "close": list(c)}


class SignalTests(unittest.TestCase):
    def test_tsmom_compares_with_the_close_lookback_candles_ago(self):
        p = {**strategy.params(strat("tsmom")), "lookback": 3}  # short window to check the arithmetic
        states = strategy.signal_states(p, [10, 11, 12, 13, 12, 11, 10])
        self.assertEqual(list(states), [0, 0, 0, 1, 1, -1, -1])  # 13>10, 12>11, 11<12, 10<13
        self.assertEqual(strategy.signal_state(p, [10, 11]), None)  # not enough history

    def test_ma_regime_is_the_close_against_its_average(self):
        p = {**strategy.params(strat("ma_regime")), "ma": 3}
        states = strategy.signal_states(p, [1, 2, 3, 4, 1, 1])
        self.assertEqual(list(states), [0, 0, 1, 1, -1, -1])  # 3>2, 4>3, 1<2.67, 1<2
        long_only = strategy.params(strat("ma_regime"))
        self.assertIsNone(strategy.target_direction(long_only, -1))

    def test_volatility_scales_the_size_down_never_up(self):
        p = strategy.params(strat("tsmom", lookback=10, target_vol=0.5))
        self.assertAlmostEqual(strategy.vol_scale(p, 1.0), 0.5)
        self.assertEqual(strategy.vol_scale(p, 0.2), 1.0)
        self.assertAlmostEqual(strategy.size(p, 10000, 100, 5, strategy.vol_scale(p, 1.0)), 50)

    def test_backtest_follows_the_regime(self):
        closes = [100 + i for i in range(40)] + [140 - 2 * i for i in range(30)]
        sim = backtest.simulate(strat("ma_regime", ma=10), candles(closes), None, 10000, 0.001, 0.0002)
        self.assertEqual(sim["trades"][0]["direction"], "long")
        self.assertEqual(sim["trades"][0]["exit_reason"], "signal_exit")
        self.assertTrue(all(t["direction"] == "long" for t in sim["trades"]))  # long-only never shorts


if __name__ == "__main__":
    unittest.main()
