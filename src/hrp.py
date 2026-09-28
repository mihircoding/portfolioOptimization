"""Hierarchical risk parity: allocate down a tree, never invert a matrix.

Every optimizer in optimizer.py has the same shape underneath: it needs
Sigma inverse. Min-variance is Sigma^-1 1 over 1' Sigma^-1 1, max-Sharpe is
Sigma^-1 mu normalized, and section 9 of RESULTS.md is a long account of
what happens to those when the covariance is estimated from not much data -
the smallest eigenvalues are the worst estimated, inversion divides by them,
and the optimizer puts its largest bets exactly there.

Lopez de Prado's answer (2016) is to stop inverting. HRP uses the covariance
only through pairwise correlations, which are the part of the matrix that
survives a short sample, and turns the allocation into a sequence of
two-way splits:

  1. Turn the correlation matrix into a distance: d_ij = sqrt((1 - rho)/2).
     This is a real metric - identical assets are 0 apart, perfectly
     opposite ones are 1 - so "close together" means what it sounds like.
  2. Cluster the assets with single linkage, building a tree of which
     things move together.
  3. Quasi-diagonalize: reorder the assets into the tree's leaf order, so
     correlated assets end up adjacent and the covariance matrix is close
     to block diagonal along its diagonal.
  4. Bisect recursively. Split the ordered list in half, compute each half's
     variance as if it were held at inverse-variance weights, and split the
     money between the halves inversely to those variances. Recurse.

Step 4 is the whole method in one sentence: at every level, the less risky
branch gets more. No matrix inverse, no constraint solver, no optimizer that
can fail to converge - the weights are positive and sum to one by
construction, because they are a product of fractions.

What it is NOT is a free lunch. HRP never looks at expected returns, so it
cannot express a view; and step 4's "variance of a cluster held at
inverse-variance weights" is a heuristic, not the solution to a stated
problem. It buys robustness by refusing to solve the harder problem at all.
Whether that is a good trade is an empirical question, which is what
run_optimization.py's walk-forward table and hrp_study.py are for.

One more piece of small print, because it is easy to miss and it is in the
published algorithm rather than in this implementation: step 4 bisects the
ORDERED LIST by count, not the tree at its cluster boundary. Three copies of
one bet and one independent bet get cut 2-2 rather than 3-1, so one copy is
weighed against the independent asset instead of against its own siblings.
The effect is real and measured in tests/test_hrp.py: the duplicate cluster
keeps about two thirds of the book where inverse-variance weighting gives it
three quarters and "one bet is one bet" would give it a half. HRP discounts
redundancy; it does not remove it.

The linkage is implemented here rather than imported from
scipy.cluster.hierarchy, for the same reason the Kalman filter in the pairs
project is written out: the clustering step is where the method's behaviour
comes from, and reading it is the point. tests/test_hrp.py checks it agrees
with scipy's on random inputs.
"""

import numpy as np


def correlation_from_covariance(cov: np.ndarray) -> np.ndarray:
    """Correlation matrix, with zero-variance assets treated as uncorrelated.

    A constant series has no correlation with anything - the quantity is 0/0,
    not 1 - and letting a NaN through here would silently poison the whole
    tree three functions later.
    """
    sd = np.sqrt(np.diag(cov))
    safe = np.where(sd > 0, sd, 1.0)
    corr = cov / np.outer(safe, safe)
    dead = sd <= 0
    if dead.any():
        corr[dead, :] = 0.0
        corr[:, dead] = 0.0
    np.fill_diagonal(corr, 1.0)
    return np.clip(corr, -1.0, 1.0)


def correlation_distance(corr: np.ndarray) -> np.ndarray:
    """d_ij = sqrt((1 - rho_ij) / 2), the standard correlation metric.

    Worth checking it really is a metric rather than just a monotone
    transform: it satisfies the triangle inequality, which is what lets a
    clustering algorithm mean anything by "these two are closer to each
    other than to that one".
    """
    return np.sqrt(np.clip((1.0 - corr) / 2.0, 0.0, 1.0))


def single_linkage(dist: np.ndarray) -> np.ndarray:
    """Agglomerative single-linkage clustering, in scipy's linkage format.

    Rows are [left, right, height, size]. Leaves are 0..n-1 and the cluster
    formed at row k is numbered n+k, which is the convention every tree
    utility expects.

    Single linkage means the distance between two clusters is the distance
    between their nearest members. It is the most permissive rule - it will
    happily chain a long thin cluster together - and it is what the original
    HRP paper uses. For correlation distances that is defensible: two stocks
    belong in the same block if either one is tightly linked to it, not only
    if the whole block is.

    O(n^3) as written, which is nothing at the sizes here (50 assets is 50
    merges over a 50x50 matrix) and far clearer than the O(n^2) nearest-
    neighbour-chain version.
    """
    n = dist.shape[0]
    active = {i: [i] for i in range(n)}      # cluster id -> member leaves
    d = dist.astype(float).copy()
    np.fill_diagonal(d, np.inf)
    ids = list(range(n))
    out = []

    for step in range(n - 1):
        best = None
        for a_idx, a in enumerate(ids):
            for b in ids[a_idx + 1:]:
                if d[a, b] < (np.inf if best is None else best[0]):
                    best = (d[a, b], a, b)
        height, a, b = best
        new_id = n + step
        out.append([a, b, height, len(active[a]) + len(active[b])])

        # Grow the distance matrix by one row/column for the new cluster and
        # fill it with the single-linkage rule: nearest member wins.
        d = np.pad(d, ((0, 1), (0, 1)), constant_values=np.inf)
        for c in ids:
            if c in (a, b):
                continue
            nearest = min(d[a, c], d[b, c])
            d[new_id, c] = d[c, new_id] = nearest
        active[new_id] = active[a] + active[b]
        ids = [c for c in ids if c not in (a, b)] + [new_id]

    return np.array(out, dtype=float)


def quasi_diagonal_order(link: np.ndarray) -> list[int]:
    """Leaf order of the tree: correlated assets end up next to each other.

    This is the step that gives the method its name. Reordering the
    covariance matrix this way pushes the large entries towards the diagonal,
    so the recursive bisection below is splitting between blocks that really
    are less related to each other than to themselves.
    """
    n = int(link[-1, 3])
    order = [int(link[-1, 0]), int(link[-1, 1])]
    while max(order) >= n:
        expanded = []
        for item in order:
            if item < n:
                expanded.append(item)
            else:
                row = link[item - n]
                expanded.extend([int(row[0]), int(row[1])])
        order = expanded
    return order


def inverse_variance_weights(cov: np.ndarray) -> np.ndarray:
    """1/sigma^2, normalized. The allocation inside a cluster."""
    ivp = 1.0 / np.diag(cov)
    return ivp / ivp.sum()


def cluster_variance(cov: np.ndarray, members: list[int]) -> float:
    """Variance of a sub-portfolio held at inverse-variance weights.

    This is the heuristic at the centre of HRP and the part most worth being
    honest about: it is not the minimum variance of that cluster, and it is
    not the variance of what HRP will eventually hold there either, because
    deeper splits will change those weights. It is a cheap, monotone,
    inversion-free stand-in for "how risky is this branch", which is all the
    split below needs it to be.
    """
    sub = cov[np.ix_(members, members)]
    w = inverse_variance_weights(sub)
    return float(w @ sub @ w)


def recursive_bisection(cov: np.ndarray, order: list[int]) -> np.ndarray:
    """Split the money down the tree, inversely to each branch's variance."""
    weights = np.ones(len(cov))
    clusters = [order]
    while clusters:
        nxt = []
        for cluster in clusters:
            if len(cluster) <= 1:
                continue
            half = len(cluster) // 2
            left, right = cluster[:half], cluster[half:]
            v_left = cluster_variance(cov, left)
            v_right = cluster_variance(cov, right)
            # More variance, less money. The fraction going left is the
            # right branch's share of total variance.
            alpha = 1.0 - v_left / (v_left + v_right)
            weights[left] *= alpha
            weights[right] *= 1.0 - alpha
            nxt.extend([left, right])
        clusters = nxt
    return weights


def hrp_weights(cov: np.ndarray) -> np.ndarray:
    """Hierarchical risk parity weights: long only, summing to one.

    No expected returns, no matrix inverse, no solver that can fail. Compare
    with min_variance_weights() in optimizer.py, which needs both a Sigma
    inverse and a constrained solve to produce something with the same two
    properties.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape[0] == 1:
        return np.ones(1)
    corr = correlation_from_covariance(cov)
    link = single_linkage(correlation_distance(corr))
    order = quasi_diagonal_order(link)
    w = recursive_bisection(cov, order)
    return w / w.sum()
