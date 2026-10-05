import numpy as np
import pandas as pd
import pytest

from src.backtest import backtest_pair, max_drawdown, sharpe_ratio


class TestSharpe:
    def test_known_value(self):
        r = pd.Series([0.01, -0.005, 0.02, 0.0, 0.007] * 100)
        expected = r.mean() / r.std(ddof=1) * np.sqrt(252)
        assert sharpe_ratio(r) == pytest.approx(expected)

    def test_zero_vol_does_not_crash(self):
        assert sharpe_ratio(pd.Series([0.0, 0.0, 0.0])) == 0.0


class TestMaxDrawdown:
    def test_known_curve(self):
        # peak 1.2, trough 0.9 -> drawdown = 0.9/1.2 - 1 = -0.25
        eq = pd.Series([1.0, 1.2, 1.0, 0.9, 1.1, 1.3])
        assert max_drawdown(eq) == pytest.approx(-0.25)

    def test_monotone_curve_has_zero_drawdown(self):
        eq = pd.Series([1.0, 1.1, 1.2, 1.3])
        assert max_drawdown(eq) == pytest.approx(0.0)


class TestBacktest:
    def make_inputs(self):
        idx = pd.bdate_range("2020-01-01", periods=6)
        y = pd.Series([100.0, 101.0, 102.0, 101.0, 100.0, 100.0], index=idx)
        x = pd.Series([100.0, 100.0, 100.0, 100.0, 100.0, 100.0], index=idx)
        pos = pd.Series([0, 1, 1, 0, 0, 0], index=idx)
        return y, x, pos

    def test_positions_are_lagged(self):
        # pos goes long on day 1, so day 1 must still earn nothing
        y, x, pos = self.make_inputs()
        res = backtest_pair(y, x, beta=1.0, positions=pos, cost_bps=0.0)
        assert res["ret"].iloc[1] == pytest.approx(0.0)
        # day 2: long spread, y up ~0.99%, x flat -> positive return
        assert res["ret"].iloc[2] > 0

    def test_costs_reduce_returns(self):
        y, x, pos = self.make_inputs()
        gross = backtest_pair(y, x, beta=1.0, positions=pos, cost_bps=0.0)
        net = backtest_pair(y, x, beta=1.0, positions=pos, cost_bps=20.0)
        assert net["equity"].iloc[-1] < gross["equity"].iloc[-1]

    def test_flat_strategy_flat_equity(self):
        y, x, _ = self.make_inputs()
        pos = pd.Series(0, index=y.index)
        res = backtest_pair(y, x, beta=1.0, positions=pos, cost_bps=5.0)
        assert res["equity"].iloc[-1] == pytest.approx(1.0)
