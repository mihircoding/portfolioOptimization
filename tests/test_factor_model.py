"""Tests for the factor-model covariance estimator.

The interesting ones here are not "does the function run" but the two
structural guarantees the estimator is bought for: the diagonal is preserved
exactly, and the result is positive semi-definite even when the sample
covariance is singular. If either of those breaks, the min-variance optimizer
downstream produces garbage rather than an error.
"""

import numpy as np
import pytest

from src.factor_model import (factor_variance_share, marchenko_pastur_edge,
                              num_significant_factors, pca_factor_covariance)
from src.optimizer import min_variance_weights_analytic


def one_factor_returns(t_obs=1000, n_assets=20, seed=0):
    """Returns that genuinely have one factor plus independent noise."""
    rng = np.random.default_rng(seed)
    factor = rng.normal(0, 0.01, size=t_obs)
    betas = rng.uniform(0.5, 1.5, size=n_assets)
    noise = rng.normal(0, 0.008, size=(t_obs, n_assets))
    return np.outer(factor, betas) + noise


def test_k_equals_n_reproduces_the_sample_covariance():
    """With as many factors as assets there is no residual left, so the model
    is an identity. Anything else means the eigendecomposition is wrong."""
    r = one_factor_returns(n_assets=10)
    assert np.allclose(pca_factor_covariance(r, 10), np.cov(r, rowvar=False))


def test_k_zero_is_the_diagonal():
    r = one_factor_returns(n_assets=10)
    sample = np.cov(r, rowvar=False)
    assert np.allclose(pca_factor_covariance(r, 0), np.diag(np.diag(sample)))


@pytest.mark.parametrize("k", [1, 2, 5, 9])
def test_diagonal_is_preserved_exactly(k):
    """Asset volatilities are the one thing a sample estimate gets right, so
    the model must not touch them - it only restructures correlations."""
    r = one_factor_returns(n_assets=10)
    sample = np.cov(r, rowvar=False)
    model = pca_factor_covariance(r, k)
    assert np.allclose(np.diag(model), np.diag(sample))


@pytest.mark.parametrize("k", [1, 2, 3])
def test_positive_semidefinite_when_the_sample_covariance_is_singular(k):
    """30 assets from 10 observations: the sample covariance has rank 9 and
    twenty-one eigenvalues that are numerically zero or negative. That matrix
    is what makes an unconstrained optimizer produce absurd weights. The factor
    model's smallest eigenvalue must stay positive."""
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, size=(10, 30))
    assert np.linalg.eigvalsh(np.cov(r, rowvar=False))[0] < 1e-12
    assert np.linalg.eigvalsh(pca_factor_covariance(r, k))[0] > 0


def test_it_recovers_a_one_factor_structure():
    """Built from one factor, so a one-factor model should be close to the
    truth and much closer than assuming no correlation at all."""
    r = one_factor_returns()
    sample = np.cov(r, rowvar=False)
    err_factor = np.abs(pca_factor_covariance(r, 1) - sample).sum()
    err_diag = np.abs(pca_factor_covariance(r, 0) - sample).sum()
    assert err_factor < 0.2 * err_diag


def test_marchenko_pastur_edge_shrinks_toward_one_with_more_data():
    assert marchenko_pastur_edge(50, 252) > marchenko_pastur_edge(50, 1008)
    assert marchenko_pastur_edge(50, 10 ** 8) == pytest.approx(1.0, abs=1e-2)


def test_pure_noise_produces_no_significant_factors():
    """The whole point of the cutoff: independent series must not look like
    they share anything, however many of them there are."""
    rng = np.random.default_rng(7)
    r = rng.normal(0, 0.01, size=(500, 40))
    assert num_significant_factors(r, min_factors=0) == 0


def test_a_real_factor_clears_the_cutoff():
    assert num_significant_factors(one_factor_returns()) >= 1


def test_variance_share_is_monotone_and_ends_at_one():
    r = one_factor_returns(n_assets=10)
    shares = [factor_variance_share(r, k) for k in range(11)]
    assert shares == sorted(shares)
    assert shares[-1] == pytest.approx(1.0)


def test_factor_covariance_gives_saner_weights_than_a_singular_sample():
    """The practical payoff, stated as a test. A near-singular covariance sends
    the closed-form min-variance solution to enormous offsetting long and short
    positions; the factor model cannot, because it has no near-zero eigenvalues
    to exploit."""
    rng = np.random.default_rng(1)
    r = rng.normal(0, 0.01, size=(60, 55))
    gross_sample = np.abs(min_variance_weights_analytic(np.cov(r, rowvar=False))).sum()
    gross_factor = np.abs(min_variance_weights_analytic(pca_factor_covariance(r, 3))).sum()
    # Gross exposure of 1.0 is fully invested with no shorts; anything above
    # that is offsetting longs and shorts the data never justified.
    assert gross_sample > 3.0
    assert gross_factor < 1.2


def test_rejects_impossible_k():
    r = one_factor_returns(n_assets=5)
    with pytest.raises(ValueError):
        pca_factor_covariance(r, 6)
    with pytest.raises(ValueError):
        pca_factor_covariance(r, -1)
