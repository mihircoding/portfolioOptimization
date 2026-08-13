import numpy as np
import pytest

from src.optimizer import (max_sharpe_weights, min_variance_weights,
                           portfolio_performance)


class TestPerformance:
    def test_formulas(self, three_asset):
        mu, cov = three_asset
        w = np.array([0.5, 0.3, 0.2])
        ret, vol, sharpe = portfolio_performance(w, mu, cov)
        assert ret == pytest.approx(w @ mu)
        assert vol == pytest.approx(np.sqrt(w @ cov @ w))
        assert sharpe == pytest.approx(ret / vol)


class TestMinVariance:
    def test_matches_two_asset_closed_form(self, two_asset_cov):
        # w1 = (s2^2 - s12) / (s1^2 + s2^2 - 2 s12)
        c = two_asset_cov
        expected_w1 = (c[1, 1] - c[0, 1]) / (c[0, 0] + c[1, 1] - 2 * c[0, 1])
        w = min_variance_weights(c)
        assert w[0] == pytest.approx(expected_w1, abs=1e-4)
        assert w.sum() == pytest.approx(1.0)

    def test_long_only_respected(self, three_asset):
        _, cov = three_asset
        w = min_variance_weights(cov, long_only=True)
        assert (w >= -1e-8).all()

    def test_beats_equal_weight(self, two_asset_cov):
        w = min_variance_weights(two_asset_cov)
        ew = np.array([0.5, 0.5])
        assert w @ two_asset_cov @ w <= ew @ two_asset_cov @ ew + 1e-12


class TestMaxSharpe:
    def test_finds_dominant_asset(self):
        # same vol, uncorrelated, asset 1 has double the return:
        # any weight on asset 0 only hurts risk-adjusted return... but
        # diversification still helps; asset 1 must get the LARGER weight.
        mu = np.array([0.05, 0.10])
        cov = np.diag([0.15**2, 0.15**2])
        w = max_sharpe_weights(mu, cov)
        assert w[1] > w[0]
        assert w.sum() == pytest.approx(1.0)

    def test_sharpe_at_least_equal_weight(self, three_asset):
        mu, cov = three_asset
        w = max_sharpe_weights(mu, cov)
        _, _, s_opt = portfolio_performance(w, mu, cov)
        _, _, s_ew = portfolio_performance(np.full(3, 1 / 3), mu, cov)
        assert s_opt >= s_ew - 1e-9
