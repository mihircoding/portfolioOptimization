import numpy as np
import pytest

from src.risk_parity import (equal_risk_contribution_weights,
                             inverse_vol_weights, ledoit_wolf_alpha,
                             risk_contributions, shrink_covariance)


class TestShrinkage:
    def test_endpoints(self, two_asset_cov):
        c = two_asset_cov
        np.testing.assert_allclose(shrink_covariance(c, 0.0), c)
        np.testing.assert_allclose(shrink_covariance(c, 1.0), np.diag(np.diag(c)))

    def test_halfway_offdiagonal(self, two_asset_cov):
        c = two_asset_cov
        shrunk = shrink_covariance(c, 0.5)
        assert shrunk[0, 1] == pytest.approx(0.5 * c[0, 1])
        assert shrunk[0, 0] == pytest.approx(c[0, 0])  # variances untouched


class TestInverseVol:
    def test_diagonal_case(self):
        cov = np.diag([0.10**2, 0.20**2])  # vols 10% and 20%
        w = inverse_vol_weights(cov)
        # 1/0.1 : 1/0.2 = 2 : 1 -> weights 2/3, 1/3
        np.testing.assert_allclose(w, [2 / 3, 1 / 3], atol=1e-9)


class TestRiskContributions:
    def test_sums_to_one(self, two_asset_cov):
        w = np.array([0.6, 0.4])
        rc = risk_contributions(w, two_asset_cov)
        assert rc.sum() == pytest.approx(1.0)

    def test_single_asset_owns_all_risk(self, two_asset_cov):
        rc = risk_contributions(np.array([1.0, 0.0]), two_asset_cov)
        np.testing.assert_allclose(rc, [1.0, 0.0], atol=1e-12)


class TestERC:
    def test_contributions_equal(self):
        # correlated, unequal vols — the non-trivial case
        s = np.array([0.10, 0.18, 0.25])
        rho = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]])
        cov = np.outer(s, s) * rho
        w = equal_risk_contribution_weights(cov)
        rc = risk_contributions(w, cov)
        np.testing.assert_allclose(rc, np.full(3, 1 / 3), atol=1e-3)
        assert w.sum() == pytest.approx(1.0)
        assert (w > 0).all()

    def test_reduces_to_inverse_vol_when_uncorrelated(self):
        cov = np.diag([0.10**2, 0.20**2])
        w = equal_risk_contribution_weights(cov)
        np.testing.assert_allclose(w, inverse_vol_weights(cov), atol=1e-3)

    def test_equal_correlations_also_reduce_to_inverse_vol(self):
        """The one case with a closed form: equal pairwise correlation."""
        s = np.array([0.08, 0.14, 0.22, 0.31])
        cov = np.outer(s, s) * (np.full((4, 4), 0.45) + np.eye(4) * 0.55)
        np.testing.assert_allclose(equal_risk_contribution_weights(cov),
                                   inverse_vol_weights(cov), atol=1e-6)

    def test_solves_a_fifty_asset_covariance_exactly(self):
        """The case the old dispersion-minimizing solver used to give up on.

        Fifty correlated assets from a factor structure - the same shape as
        the walk-forward's awkward windows. The convex formulation's optimum
        satisfies (Sigma w)_i * w_i = 1/n exactly, so the risk contributions
        should be equal to numerical precision, not to a solver tolerance.
        """
        rng = np.random.default_rng(0)
        n = 50
        loadings = rng.normal(0, 1, size=(n, 3))
        specific = np.diag(rng.uniform(0.02, 0.25, n) ** 2)
        cov = loadings @ loadings.T * 0.01 + specific

        w = equal_risk_contribution_weights(cov)
        rc = risk_contributions(w, cov)
        np.testing.assert_allclose(rc, np.full(n, 1 / n), atol=1e-9)
        assert w.sum() == pytest.approx(1.0)
        assert (w > 0).all()

    def test_the_answer_does_not_depend_on_the_scale_of_the_covariance(self):
        """Weights are scale-free; risk contributions are shares, not sizes."""
        s = np.array([0.10, 0.18, 0.25])
        rho = np.array([[1.0, 0.4, 0.2], [0.4, 1.0, 0.5], [0.2, 0.5, 1.0]])
        cov = np.outer(s, s) * rho
        np.testing.assert_allclose(equal_risk_contribution_weights(cov),
                                   equal_risk_contribution_weights(cov * 10_000),
                                   atol=1e-10)

    def test_a_single_asset_holds_all_of_it(self):
        np.testing.assert_allclose(
            equal_risk_contribution_weights(np.array([[0.04]])), [1.0])


class TestLedoitWolfAlpha:
    def _correlated_returns(self, rng, T, N=5):
        # a single common factor plus idiosyncratic noise -> genuinely
        # correlated assets, not the degenerate all-independent case
        factor = rng.normal(0, 0.01, T)
        idio = rng.normal(0, 0.01, (T, N))
        loadings = np.array([0.8, 0.6, 1.0, 0.4, 0.7])
        return factor[:, None] * loadings[None, :] + idio

    def test_alpha_is_a_valid_shrinkage_intensity(self):
        rng = np.random.default_rng(0)
        returns = self._correlated_returns(rng, T=750)
        alpha = ledoit_wolf_alpha(returns)
        assert 0.0 <= alpha <= 1.0

    def test_less_data_means_more_shrinkage(self):
        # same generating process, just truncated - the short sample should
        # need MORE help from the target, not less. This is the property
        # the whole estimator exists for; if this direction were backwards
        # the formula would be wrong, not just imprecise.
        rng = np.random.default_rng(1)
        returns = self._correlated_returns(rng, T=1500)
        alpha_short = ledoit_wolf_alpha(returns[:60])
        alpha_long = ledoit_wolf_alpha(returns)
        assert alpha_short > alpha_long

    def test_exactly_diagonal_sample_shrinks_fully(self):
        # a 2x2 Hadamard block, tiled: both columns are exactly mean-zero
        # and exactly orthogonal by construction (not approximately, by
        # cancellation), so the sample covariance IS the diagonal target
        # already. alpha should saturate at 1.0 instead of dividing by a
        # near-zero denominator and blowing up.
        block = np.array([[1.0, 1.0], [1.0, -1.0], [-1.0, 1.0], [-1.0, -1.0]])
        returns = np.tile(block, (25, 1))  # 100 x 2, still exactly orthogonal
        assert ledoit_wolf_alpha(returns) == 1.0
