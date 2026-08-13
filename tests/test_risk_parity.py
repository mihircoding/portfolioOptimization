import numpy as np
import pytest

from src.risk_parity import (equal_risk_contribution_weights,
                             inverse_vol_weights, risk_contributions,
                             shrink_covariance)


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
