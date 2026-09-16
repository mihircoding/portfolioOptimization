"""Black-Litterman, checked against the properties that define it.

The formula is easy to write down and easy to get subtly wrong - a transpose
in the wrong place still produces plausible-looking numbers. So these tests
are mostly about the limits, where the right answer is known exactly:

  - no views                -> the equilibrium, unchanged
  - a view stating exactly what equilibrium already says -> no change
  - a certain view          -> binds exactly
  - a worthless view        -> ignored entirely
  - the round trip          -> reverse then forward optimization is identity
"""

import numpy as np
import pytest

from src.black_litterman import (absolute_view, black_litterman_weights,
                                 default_view_uncertainty, equilibrium_returns,
                                 implied_risk_aversion, posterior, relative_view)

# three assets: a volatile one, a quiet one, and a middling one correlated
# with the volatile one - enough structure for a view to propagate
COV = np.array([
    [0.0400, 0.0060, 0.0180],
    [0.0060, 0.0025, 0.0020],
    [0.0180, 0.0020, 0.0225],
])
W_MKT = np.array([0.55, 0.30, 0.15])
DELTA = 2.5


class TestReverseOptimization:
    def test_equilibrium_is_delta_sigma_w(self):
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        assert pi == pytest.approx(DELTA * COV @ W_MKT)

    def test_weights_must_sum_to_one(self):
        with pytest.raises(ValueError, match="sum to 1"):
            equilibrium_returns(COV, np.array([0.5, 0.3, 0.1]), DELTA)

    def test_riskier_assets_get_higher_equilibrium_returns(self):
        # the volatile asset carries more of the market's risk, so equilibrium
        # has to pay more for holding it
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        assert pi[0] > pi[2] > pi[1]

    def test_implied_risk_aversion(self):
        # 6% excess return on 15% vol -> 6% / 2.25% = 2.67
        assert implied_risk_aversion(0.06, 0.15 ** 2) == pytest.approx(2.667, rel=1e-3)

    def test_zero_variance_is_rejected(self):
        with pytest.raises(ValueError):
            implied_risk_aversion(0.06, 0.0)


class TestPosterior:
    def test_no_views_returns_the_prior(self):
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        mu_bl, _ = posterior(COV, pi)
        assert mu_bl == pytest.approx(pi)

    def test_a_view_that_agrees_with_equilibrium_changes_nothing(self):
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        P, Q = absolute_view(3, 0, pi[0])          # "asset 0 returns exactly pi[0]"
        mu_bl, _ = posterior(COV, pi, P, Q)
        assert mu_bl == pytest.approx(pi, abs=1e-12)

    def test_a_certain_view_binds_exactly(self):
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        P, Q = absolute_view(3, 0, 0.20)
        tiny = np.array([[1e-12]])                 # near-zero view variance
        mu_bl, _ = posterior(COV, pi, P, Q, omega=tiny)
        assert mu_bl[0] == pytest.approx(0.20, abs=1e-5)

    def test_a_worthless_view_is_ignored(self):
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        P, Q = absolute_view(3, 0, 0.99)
        huge = np.array([[1e9]])                   # infinite view variance
        mu_bl, _ = posterior(COV, pi, P, Q, omega=huge)
        assert mu_bl == pytest.approx(pi, abs=1e-6)

    def test_a_bullish_view_pulls_that_asset_up(self):
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        P, Q = absolute_view(3, 0, pi[0] + 0.05)
        mu_bl, _ = posterior(COV, pi, P, Q)
        assert mu_bl[0] > pi[0]

    def test_a_view_propagates_to_correlated_assets(self):
        """The property that makes Black-Litterman worth the trouble.

        Nobody wrote a view about asset 2. It moves anyway, because Sigma says
        it moves with asset 0, and a coherent posterior can't raise the return
        on one without raising it on things that track it.
        """
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        P, Q = absolute_view(3, 0, pi[0] + 0.05)
        mu_bl, _ = posterior(COV, pi, P, Q)
        assert mu_bl[2] > pi[2]                    # correlated 0.6 with asset 0
        assert mu_bl[2] - pi[2] < mu_bl[0] - pi[0]  # but by less than asset 0

    def test_relative_view_moves_the_spread_toward_the_view(self):
        """A relative view constrains a difference, not two levels.

        Worth stating as a test because the intuitive expectation - "asset 1
        goes up, asset 0 goes down" - is wrong, and code written to that
        expectation would look fine. Both legs can move the same way. The
        only thing the view pins down is the gap between them, which lands
        between the prior gap and the view, never past it.
        """
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        P, Q = relative_view(3, 1, 0, 0.03)        # quiet asset beats volatile one
        mu_bl, _ = posterior(COV, pi, P, Q)

        prior_spread = pi[1] - pi[0]
        post_spread = mu_bl[1] - mu_bl[0]
        assert prior_spread < post_spread < 0.03

    def test_posterior_covariance_exceeds_the_prior(self):
        # cov_BL = Sigma + M: the estimate of the mean is itself uncertain,
        # and pretending otherwise is what makes optimizers overconfident
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        _, cov_bl = posterior(COV, pi)
        assert np.all(np.diag(cov_bl) > np.diag(COV))

    def test_mismatched_view_shapes_are_rejected(self):
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        with pytest.raises(ValueError, match="views"):
            posterior(COV, pi, np.eye(3)[:2], np.array([0.1]))
        with pytest.raises(ValueError, match="columns"):
            posterior(COV, pi, np.eye(2), np.array([0.1, 0.1]))


class TestRoundTrip:
    def test_no_views_recovers_the_market_portfolio_exactly(self):
        """The check the whole construction stands on.

        Reverse-optimize the market to get Pi, then forward-optimize Pi at the
        same risk aversion. If that isn't the market portfolio again, the two
        halves are solving different problems.
        """
        w = black_litterman_weights(COV, W_MKT, DELTA, use_posterior_cov=False)
        assert w == pytest.approx(W_MKT, abs=1e-4)

    def test_posterior_covariance_nudges_toward_the_quiet_assets(self):
        """Same thing with Sigma + M, which is the default and is not identity.

        The extra uncertainty makes the same risk aversion want less risk, so
        the unconstrained answer is w_market / (1 + tau). Being fully invested
        puts that missing 5% back along the minimum-variance direction, which
        is why the quiet asset ends up slightly overweight its market share.
        """
        w = black_litterman_weights(COV, W_MKT, DELTA)
        assert w == pytest.approx(W_MKT, abs=0.04)
        assert w[1] > W_MKT[1]          # the quiet asset picks up the slack

    def test_a_bullish_view_tilts_toward_that_asset(self):
        # uncorrelated, so the view can't leak into anything else
        cov = np.diag([0.04, 0.0225, 0.01])
        w_mkt = np.array([0.4, 0.35, 0.25])
        pi = equilibrium_returns(cov, w_mkt, DELTA)
        P, Q = absolute_view(3, 0, pi[0] + 0.05)
        w = black_litterman_weights(cov, w_mkt, DELTA, P, Q)
        assert w[0] > w_mkt[0]

    def test_a_view_on_a_quiet_asset_moves_the_book_into_a_volatile_one(self):
        """The behavior that surprises people, and the reason it is correct.

        The view is about asset 1, whose variance is 0.0025. Asset 0 is
        correlated 0.6 with it and eight times as volatile, so the regression
        of asset 0's mean on asset 1's is 0.006 / 0.0025 = 2.4. Raise asset
        1's expected return by 2 points and the posterior has to raise asset
        0's by nearly 5, and the optimizer buys asset 0 rather than the asset
        the view was about.

        This is Black-Litterman working, not failing - it is refusing to hold
        a belief about one asset that is incoherent with what Sigma says about
        the others. It is also the reason views on low-variance assets need
        care: the quieter the asset, the more leverage a view about it exerts
        on everything correlated with it.
        """
        pi = equilibrium_returns(COV, W_MKT, DELTA)
        P, Q = absolute_view(3, 1, pi[1] + 0.04)
        mu_bl, _ = posterior(COV, pi, P, Q)
        w = black_litterman_weights(COV, W_MKT, DELTA, P, Q)

        assert mu_bl[0] - pi[0] > mu_bl[1] - pi[1]   # the spillover is larger
        assert w[0] > W_MKT[0]

    def test_weights_stay_long_only_and_fully_invested(self):
        P, Q = relative_view(3, 1, 0, 0.10)        # a large view
        w = black_litterman_weights(COV, W_MKT, DELTA, P, Q)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert np.all(w >= -1e-9)


class TestViewUncertainty:
    def test_omega_is_diagonal(self):
        P = np.vstack([absolute_view(3, 0, 0.1)[0], relative_view(3, 1, 2, 0.02)[0]])
        omega = default_view_uncertainty(COV, P)
        assert omega.shape == (2, 2)
        assert omega[0, 1] == 0.0 and omega[1, 0] == 0.0

    def test_views_about_volatile_combinations_are_trusted_less(self):
        P = np.vstack([absolute_view(3, 0, 0.1)[0],   # the volatile asset
                       absolute_view(3, 1, 0.1)[0]])  # the quiet one
        omega = default_view_uncertainty(COV, P)
        assert omega[0, 0] > omega[1, 1]
