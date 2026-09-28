"""Hierarchical risk parity, checked against things that are known.

Three kinds of test here. The clustering is checked against scipy, because
the linkage is written out by hand in src/hrp.py and "I reimplemented a
standard algorithm" is only a good idea if it agrees with the standard one.
The allocation is checked against cases where the right answer is forced by
symmetry - equal, uncorrelated, equal-variance assets must get equal weight,
and a block of near-duplicates must not get more money for being numerous.
And the structural guarantees are checked directly: long only, sums to one,
no matrix inverse anywhere, which is the reason to use the method at all.
"""
import numpy as np
import pytest
from scipy.cluster.hierarchy import linkage
from scipy.spatial.distance import squareform

from src.hrp import (cluster_variance, correlation_distance,
                     correlation_from_covariance, hrp_weights,
                     inverse_variance_weights, quasi_diagonal_order,
                     recursive_bisection, single_linkage)


def random_cov(n, seed, t=600):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(t, n))
    return np.cov(x, rowvar=False)


def test_linkage_agrees_with_scipy():
    """Same merge heights, in the same order, on random correlation
    distances. If this ever fails the hand-written clustering is wrong, not
    scipy."""
    for seed in range(5):
        cov = random_cov(9, seed)
        d = correlation_distance(correlation_from_covariance(cov))
        mine = single_linkage(d)
        theirs = linkage(squareform(d, checks=False), method="single")
        assert np.allclose(mine[:, 2], theirs[:, 2], atol=1e-12)
        assert mine.shape == theirs.shape


def test_the_leaf_order_is_a_permutation_of_the_assets():
    cov = random_cov(12, seed=7)
    d = correlation_distance(correlation_from_covariance(cov))
    order = quasi_diagonal_order(single_linkage(d))
    assert sorted(order) == list(range(12))


def test_correlated_assets_end_up_adjacent():
    """The point of the reordering. Two near-copies must be neighbours in
    the leaf order, whatever their column positions were."""
    rng = np.random.default_rng(1)
    base = rng.normal(size=(800, 1))
    x = np.hstack([base, rng.normal(size=(800, 1)),
                   base + 0.05 * rng.normal(size=(800, 1)),
                   rng.normal(size=(800, 1))])
    cov = np.cov(x, rowvar=False)
    order = quasi_diagonal_order(
        single_linkage(correlation_distance(correlation_from_covariance(cov))))
    assert abs(order.index(0) - order.index(2)) == 1


def test_distance_is_zero_for_identical_and_one_for_opposite():
    corr = np.array([[1.0, 1.0, -1.0], [1.0, 1.0, -1.0], [-1.0, -1.0, 1.0]])
    d = correlation_distance(corr)
    assert d[0, 1] == pytest.approx(0.0)
    assert d[0, 2] == pytest.approx(1.0)
    assert np.allclose(d, d.T)


def test_weights_are_long_only_and_sum_to_one():
    """Not enforced by a constraint anywhere - it falls out of multiplying
    fractions down the tree, which is the structural claim the method
    makes."""
    for seed in range(6):
        w = hrp_weights(random_cov(15, seed))
        assert w.min() >= 0
        assert w.sum() == pytest.approx(1.0)


def test_identical_uncorrelated_assets_get_equal_weight():
    w = hrp_weights(np.eye(8) * 0.04)
    assert np.allclose(w, 1 / 8)


def test_a_riskier_asset_gets_less():
    cov = np.diag([0.01, 0.04, 0.09])
    w = hrp_weights(cov)
    assert w[0] > w[1] > w[2]


def test_duplicates_get_less_than_inverse_variance_gives_them_but_not_half():
    """Three near-copies of one bet and one independent bet.

    Inverse-variance weighting hands the copies 3/4 of the book purely for
    being numerous. HRP is supposed to fix that, and it only half does: the
    recursive bisection splits the ORDERED LIST down the middle by count,
    not the dendrogram at its cluster boundary, so a 3-1 split gets cut 2-2
    and one copy is scored against the independent asset.

    This is pinned as a test rather than smoothed over, because it is the
    method's real behaviour and the number is quotable: the copies keep
    about two thirds instead of three quarters. Splitting on the tree
    instead of on the midpoint is the obvious fix and is not what the
    published algorithm does.
    """
    rng = np.random.default_rng(4)
    a = rng.normal(size=(2000, 1))
    b = rng.normal(size=(2000, 1))
    x = np.hstack([a, a + 1e-3 * rng.normal(size=(2000, 1)),
                   a + 1e-3 * rng.normal(size=(2000, 1)), b])
    cov = np.cov(x, rowvar=False)

    hrp = hrp_weights(cov)[:3].sum()
    ivp = inverse_variance_weights(cov)[:3].sum()
    assert ivp > 0.70                       # naive: pays for the duplicates
    assert hrp < ivp                        # HRP does discount them
    assert 0.60 < hrp < 0.70                # but nowhere near down to 1/2


def test_it_never_touches_the_inverse():
    """A singular covariance kills every optimizer in optimizer.py and must
    not bother this one - which is the entire argument for using it when
    there are more assets than observations."""
    rng = np.random.default_rng(2)
    x = rng.normal(size=(30, 6))          # 6 assets, 30 days: rank deficient
    x = np.hstack([x, x[:, :2]])          # and exactly collinear as well
    cov = np.cov(x, rowvar=False)
    with pytest.raises(np.linalg.LinAlgError):
        np.linalg.inv(np.linalg.cholesky(cov))
    w = hrp_weights(cov)
    assert np.isfinite(w).all() and w.sum() == pytest.approx(1.0)


def test_a_constant_series_is_handled_rather_than_producing_nans():
    cov = np.diag([0.04, 0.0, 0.09])
    corr = correlation_from_covariance(cov)
    assert np.isfinite(corr).all()


def test_cluster_variance_uses_inverse_variance_weights():
    cov = np.diag([0.01, 0.04])
    w = inverse_variance_weights(cov)
    assert cluster_variance(cov, [0, 1]) == pytest.approx(w @ cov @ w)


def test_bisection_splits_money_away_from_the_riskier_branch():
    cov = np.diag([0.01, 0.01, 0.25, 0.25])
    w = recursive_bisection(cov, [0, 1, 2, 3])
    assert w[:2].sum() > w[2:].sum()


def test_one_asset_gets_everything():
    assert hrp_weights(np.array([[0.04]])) == pytest.approx(np.array([1.0]))
