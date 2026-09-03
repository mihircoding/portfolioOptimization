import numpy as np
import pytest

from src.optimizer import (max_sharpe_turnover_penalized, max_sharpe_weights,
                           min_variance_weights, portfolio_performance)


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


class TestMaxSharpeTurnoverPenalized:
    def test_zero_penalty_matches_plain_max_sharpe(self, three_asset):
        mu, cov = three_asset
        w_prev = np.full(3, 1 / 3)
        w_plain = max_sharpe_weights(mu, cov)
        w_penalized = max_sharpe_turnover_penalized(mu, cov, w_prev, penalty=0.0)
        assert w_penalized == pytest.approx(w_plain, abs=1e-4)

    def test_large_penalty_stays_near_previous_weights(self, three_asset):
        mu, cov = three_asset
        w_prev = np.array([0.5, 0.3, 0.2])
        w = max_sharpe_turnover_penalized(mu, cov, w_prev, penalty=1e6)
        assert w == pytest.approx(w_prev, abs=1e-3)

    def test_still_fully_invested(self, three_asset):
        mu, cov = three_asset
        w_prev = np.full(3, 1 / 3)
        w = max_sharpe_turnover_penalized(mu, cov, w_prev, penalty=25.0)
        assert w.sum() == pytest.approx(1.0)

    def test_moderate_penalty_trades_less_than_unpenalized(self, three_asset):
        mu, cov = three_asset
        w_prev = np.array([0.6, 0.1, 0.3])
        w_plain = max_sharpe_weights(mu, cov)
        w_penalized = max_sharpe_turnover_penalized(mu, cov, w_prev, penalty=10.0)
        turnover_plain = np.abs(w_plain - w_prev).sum()
        turnover_penalized = np.abs(w_penalized - w_prev).sum()
        assert turnover_penalized < turnover_plain
