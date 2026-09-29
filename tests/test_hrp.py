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


# ---------------------------------------------------------------------------
# Splitting at the tree's joins instead of the ordered list's midpoint, and
# the reason that turns out to be a worse idea on real correlation matrices.
# ---------------------------------------------------------------------------

# Aliased: the module-level `linkage` above is scipy's, and the tests that
# check this implementation against it need that name to keep meaning scipy.
from src.hrp import linkage as hrp_linkage
from src.hrp import cluster_leaves, tree_bisection, tree_shape


def _duplicate_cov(rho: float = 0.99, var: float = 0.04) -> np.ndarray:
    """Three near-identical bets and one independent one, all the same vol."""
    corr = np.eye(4)
    for i in range(3):
        for j in range(3):
            if i != j:
                corr[i, j] = rho
    return corr * var


def _block_cov(n_blocks: int = 6, per_block: int = 4,
               seed: int = 3) -> np.ndarray:
    """A covariance that really does have blocks: market factor + block factors."""
    rng = np.random.default_rng(seed)
    market = rng.normal(size=600)
    cols = []
    for _ in range(n_blocks):
        block = rng.normal(size=600)
        for _ in range(per_block):
            cols.append(0.5 * market + 0.7 * block + 0.5 * rng.normal(size=600))
    return np.cov(np.array(cols).T, rowvar=False)


class TestClusterLeaves:
    def test_a_leaf_is_its_own_only_leaf(self):
        link = hrp_linkage(np.array([[0.0, 0.3, 0.9], [0.3, 0.0, 0.8],
                                 [0.9, 0.8, 0.0]]))
        assert cluster_leaves(link, 0, 3) == [0]

    def test_the_root_holds_every_asset_once(self):
        cov = _block_cov()
        n = cov.shape[0]
        link = hrp_linkage(correlation_distance(correlation_from_covariance(cov)))
        leaves = cluster_leaves(link, n + len(link) - 1, n)
        assert sorted(leaves) == list(range(n))

    def test_it_agrees_with_the_leaf_order_the_published_step_uses(self):
        cov = _block_cov()
        n = cov.shape[0]
        link = hrp_linkage(correlation_distance(correlation_from_covariance(cov)))
        root = n + len(link) - 1
        assert sorted(cluster_leaves(link, root, n)) == \
               sorted(quasi_diagonal_order(link))


class TestAverageLinkage:
    def test_it_is_a_valid_linkage_of_the_right_shape(self):
        cov = _block_cov()
        n = cov.shape[0]
        dist = correlation_distance(correlation_from_covariance(cov))
        link = hrp_linkage(dist, method="average")
        assert link.shape == (n - 1, 4)
        assert link[-1, 3] == n                      # the root holds everything
        assert np.all(np.diff(link[:, 2]) >= -1e-12)  # heights never decrease

    def test_single_linkage_is_still_the_default_and_the_old_name_works(self):
        dist = correlation_distance(correlation_from_covariance(_block_cov()))
        assert np.allclose(single_linkage(dist), hrp_linkage(dist))
        assert np.allclose(hrp_linkage(dist), hrp_linkage(dist, method="single"))

    def test_an_unknown_method_is_refused_rather_than_guessed(self):
        dist = correlation_distance(correlation_from_covariance(_duplicate_cov()))
        with pytest.raises(ValueError):
            hrp_linkage(dist, method="ward")


class TestTreeBisection:
    def test_the_weights_are_still_positive_and_sum_to_one(self):
        for cov in (_duplicate_cov(), _block_cov()):
            w = hrp_weights(cov, split="tree")
            assert w.min() > 0
            assert w.sum() == pytest.approx(1.0)

    def test_it_splits_the_duplicate_cluster_in_half(self):
        """The whole reason the fix exists.

        The published midpoint cut gives three copies of one bet about two
        thirds of the book. Cutting at the cluster boundary gives them a half,
        which is what "one bet is one bet" means.
        """
        cov = _duplicate_cov()
        published = hrp_weights(cov, split="list")[:3].sum()
        at_the_join = hrp_weights(cov, split="tree")[:3].sum()
        assert published == pytest.approx(0.667, abs=0.01)
        assert at_the_join == pytest.approx(0.50, abs=0.01)

    def test_the_fix_does_not_depend_on_which_linkage_built_the_tree(self):
        cov = _duplicate_cov()
        for method in ("single", "average"):
            trio = hrp_weights(cov, split="tree", method=method)[:3].sum()
            assert trio == pytest.approx(0.50, abs=0.01)

    def test_an_unknown_split_is_refused(self):
        with pytest.raises(ValueError):
            hrp_weights(_duplicate_cov(), split="midpoint")

    def test_both_splits_agree_when_the_tree_is_already_balanced(self):
        """Two independent assets have one join, and it is the midpoint."""
        cov = np.diag([0.04, 0.09])
        assert np.allclose(hrp_weights(cov, split="list"),
                           hrp_weights(cov, split="tree"))


class TestTreeShape:
    def test_a_real_correlation_matrix_gives_a_ladder(self):
        """The measured reason the fix loses: there is no balanced tree there.

        Twenty assets driven by one dominant factor plus noise - which is what
        a stock universe is - have no block structure to find, so agglomeration
        peels off one name at a time whichever linkage is used.
        """
        rng = np.random.default_rng(5)
        market = rng.normal(size=800)
        cols = [0.8 * market + 0.4 * rng.normal(size=800) for _ in range(20)]
        cov = np.cov(np.array(cols).T, rowvar=False)
        dist = correlation_distance(correlation_from_covariance(cov))
        for method in ("single", "average"):
            shape = tree_shape(hrp_linkage(dist, method=method))
            assert min(shape["root"]) <= 2
            assert shape["single_asset_joins"] >= 0.6 * shape["joins"]
            assert shape["balance"] < 0.5

    def test_real_blocks_give_a_balanced_tree(self):
        shape = tree_shape(hrp_linkage(
            correlation_distance(correlation_from_covariance(_block_cov())),
            method="average"))
        assert shape["balance"] > 0.5
        assert min(shape["root"]) >= 4

    def test_balance_weights_the_big_joins_over_the_small_ones(self):
        """A ladder is full of balanced (1,1) joins at the bottom, and an
        unweighted average of balance is misleadingly high because of them."""
        rng = np.random.default_rng(6)
        market = rng.normal(size=800)
        cols = [0.8 * market + 0.4 * rng.normal(size=800) for _ in range(30)]
        cov = np.cov(np.array(cols).T, rowvar=False)
        link = hrp_linkage(correlation_distance(correlation_from_covariance(cov)))
        shape = tree_shape(link)
        n = cov.shape[0]
        raw = [min(len(cluster_leaves(link, int(link[k, 0]), n)),
                   len(cluster_leaves(link, int(link[k, 1]), n)))
               / max(len(cluster_leaves(link, int(link[k, 0]), n)),
                     len(cluster_leaves(link, int(link[k, 1]), n)))
               for k in range(len(link))]
        assert shape["balance"] < float(np.mean(raw))


class TestDescendingALadderConcentrates:
    def test_the_tree_split_holds_fewer_names_on_a_one_factor_universe(self):
        """The finding, as a test: the fix concentrates the book.

        Effective positions, 1/sum(w^2), falls when the money is split down a
        ladder, because the first join hands one asset a branch of its own.
        """
        rng = np.random.default_rng(7)
        market = rng.normal(size=900)
        cols = [0.8 * market + 0.4 * rng.normal(size=900) for _ in range(40)]
        cov = np.cov(np.array(cols).T, rowvar=False)

        eff = lambda w: 1.0 / float(np.sum(w ** 2))
        published = eff(hrp_weights(cov, split="list"))
        at_the_join = eff(hrp_weights(cov, split="tree"))
        assert at_the_join < 0.6 * published

    def test_and_does_not_when_the_blocks_are_real(self):
        cov = _block_cov(n_blocks=6, per_block=4)
        eff = lambda w: 1.0 / float(np.sum(w ** 2))
        published = eff(hrp_weights(cov, split="list", method="average"))
        at_the_join = eff(hrp_weights(cov, split="tree", method="average"))
        assert at_the_join > 0.75 * published


class TestBisectionIsIndependentOfTheClustering:
    def test_a_random_order_bisected_is_not_much_worse_on_one_factor_data(self):
        """What the study's last table is, as an assertion.

        If the balance rather than the clustering is doing the work, then
        bisecting a random order should land close to bisecting the clustered
        one. On a single-factor universe it does, within a few percent.
        """
        rng = np.random.default_rng(8)
        market = rng.normal(size=900)
        cols = [0.8 * market + 0.4 * rng.normal(size=900) for _ in range(40)]
        cov = np.cov(np.array(cols).T, rowvar=False)

        clustered = hrp_weights(cov)
        vol = lambda w: float(np.sqrt(w @ cov @ w))
        draws = []
        for _ in range(50):
            w = recursive_bisection(cov, list(rng.permutation(len(cov))))
            draws.append(vol(w / w.sum()))
        assert vol(clustered) < np.mean(draws)                  # it does help
        assert vol(clustered) > 0.9 * float(np.mean(draws))     # but barely
