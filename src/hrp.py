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

hrp_weights(cov, split="tree") is that cut made at the cluster boundary
instead, and it does fix the 2-2 split exactly - the duplicates drop from two
thirds of the book to a half. On real data it is also much worse, and the
reason is worth more than the fix: single linkage builds LADDERS, not
balanced trees, so descending one puts a large share of the book on whichever
asset the root happened to peel off first. The midpoint cut is the wrong cut
and is protected from this by being balanced whatever the tree looks like.

The textbook cure for chaining is average linkage (method="average"), and it
does not work here either - the tree is still a ladder, because a correlation
matrix with one dominant factor has no blocks to find. hrp_split_study.py
runs all four combinations and measures the tree's shape directly.

Which leaves a conclusion about the method rather than about the fix. If the
midpoint cut is carrying the balance, then most of what HRP delivers is the
balance and not the clustering - and replacing the clustered order with a RANDOM
one, keeping the midpoint cut, should barely hurt. It barely does. Over 47
quarterly rebalances the clustered order beats the average random one by 0.19
points of predicted volatility on 50 large caps - two standard deviations of the
random distribution, 1.3% in relative terms - and by 0.05 points, a fifth of a
standard deviation, on a 20-asset multi-asset book.

So three of the four steps here are worth about a percent, and the step usually
written up as the wart is carrying the rest.

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


def linkage(dist: np.ndarray, method: str = "single") -> np.ndarray:
    """Agglomerative clustering, in scipy's linkage format.

    Rows are [left, right, height, size]. Leaves are 0..n-1 and the cluster
    formed at row k is numbered n+k, which is the convention every tree
    utility expects.

    Two update rules, and the difference between them decides the SHAPE of
    the tree, which turns out to matter more than anything else in this file:

      single    the distance between two clusters is the distance between
                their nearest members. The most permissive rule, and the one
                the original HRP paper uses. Its failure mode is chaining: a
                cluster grows by absorbing whichever asset is nearest to any
                one of its members, so on real correlations it builds a
                LADDER, gluing on one asset at a time. On the 50 large caps
                in hrp_study.py, 37 of the 49 joins attach exactly one asset
                and the root splits 1 against 49.
      average   the distance between two clusters is the mean distance over
                all pairs across them, weighted by cluster size. An asset has
                to be close to the cluster as a whole rather than to one
                member of it, which is the textbook cure for chaining.

    It is worth being clear that on financial correlations the cure does not
    work. Average linkage on the same 50 large caps still splits 1 against 49
    at the root and still attaches a single asset at 36 of its 49 joins. The
    reason is not the update rule: it is that a correlation matrix dominated
    by one common factor has no block structure to find, so every asset is
    roughly equidistant from every other and greedy agglomeration snowballs
    whichever way it is told to measure. hrp_split_study.py measures the shape
    of both trees on both universes and they are both ladders.

    Which one you have does not matter at all to the published bisection,
    which cuts the ordered list at its midpoint and never looks at the tree -
    and matters enormously to tree_bisection(), which descends it.

    O(n^3) as written, which is nothing at the sizes here (50 assets is 50
    merges over a 50x50 matrix) and far clearer than the O(n^2) nearest-
    neighbour-chain version.
    """
    if method not in ("single", "average"):
        raise ValueError(f"method must be 'single' or 'average', got {method!r}")

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
        size_a, size_b = len(active[a]), len(active[b])
        out.append([a, b, height, size_a + size_b])

        # Grow the distance matrix by one row/column for the new cluster and
        # fill it with whichever update rule is in force.
        d = np.pad(d, ((0, 1), (0, 1)), constant_values=np.inf)
        for c in ids:
            if c in (a, b):
                continue
            if method == "single":
                merged = min(d[a, c], d[b, c])
            else:
                merged = ((size_a * d[a, c] + size_b * d[b, c])
                          / (size_a + size_b))
            d[new_id, c] = d[c, new_id] = merged
        active[new_id] = active[a] + active[b]
        ids = [c for c in ids if c not in (a, b)] + [new_id]

    return np.array(out, dtype=float)


def single_linkage(dist: np.ndarray) -> np.ndarray:
    """linkage(dist, "single"). Kept because it is the name the tests use."""
    return linkage(dist, method="single")


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


def cluster_leaves(link: np.ndarray, node: int, n: int) -> list[int]:
    """Every original asset underneath one node of the tree.

    Leaves are 0..n-1 and internal node n+k is the cluster formed at link row
    k, which is scipy's convention and the one single_linkage() writes.
    """
    if node < n:
        return [node]
    stack = [node]
    leaves = []
    while stack:
        item = stack.pop()
        if item < n:
            leaves.append(item)
        else:
            row = link[item - n]
            stack.extend([int(row[1]), int(row[0])])
    return leaves


def tree_shape(link: np.ndarray) -> dict:
    """How balanced the dendrogram is, which is what tree_bisection() rides on.

    Returns the root's two branch sizes, the share of joins that attach exactly
    one asset, and a balance score. Balance at one join is the smaller branch
    over the larger, and the score is the average of those WEIGHTED BY CLUSTER
    SIZE - because a lopsided cut at the root decides where most of the money
    goes, and a lopsided cut between two leaves decides almost nothing. The
    unweighted average is misleadingly high on a ladder for exactly that
    reason: a ladder is full of balanced (1,1) joins at the bottom.

    1.0 is a perfectly balanced binary tree. A pure ladder tends to 2/n.

    Worth having as a number rather than a picture, because it is the whole
    difference between the published bisection and the one that descends the
    tree, and on real correlation matrices it is not close.
    """
    n = int(link[-1, 3])
    sizes = []
    for k in range(len(link)):
        a, b = int(link[k, 0]), int(link[k, 1])
        sizes.append((len(cluster_leaves(link, a, n)),
                      len(cluster_leaves(link, b, n))))
    weights = np.array([a + b for a, b in sizes], dtype=float)
    balances = np.array([min(a, b) / max(a, b) for a, b in sizes])
    return {
        "n": n,
        "root": sizes[-1],
        "single_asset_joins": sum(1 for a, b in sizes if min(a, b) == 1),
        "joins": len(sizes),
        "balance": float((balances * weights).sum() / weights.sum()),
    }


def tree_bisection(cov: np.ndarray, link: np.ndarray) -> np.ndarray:
    """Split at the tree's own joins instead of at the ordered list's midpoint.

    recursive_bisection() above implements the published algorithm, which cuts
    the quasi-diagonal ORDER in half by count at every level. That is not the
    same thing as cutting the tree, and where the two disagree the published
    version splits a cluster down the middle and weighs half of it against
    something it is not related to. The docstring at the top of this file
    measures the consequence: three copies of one bet plus one independent bet
    get cut 2-2, and the duplicates keep about two thirds of the book.

    This version descends the dendrogram. At each internal node the two
    branches are the two things the clustering actually decided were separate,
    and the money is split between them by the same inverse-variance rule. It
    is the obvious fix and several follow-up papers make it.

    What it is not is a better method, and the reason is the tree rather than
    the cut. Correlation dendrograms of financial assets are LADDERS: on the
    50 large caps in hrp_study.py the root splits 1 against 49 and 37 of the
    49 joins attach a single asset, and switching to average linkage barely
    moves either number. Descending a ladder hands one asset a large share of
    the book before anything else is considered - effective positions fall
    from 37 to 9 on that universe. The midpoint cut is protected from this by
    being balanced whatever the tree looks like, which is not a property
    anyone claimed for it.

    So this function is here to be measured, not to be used, and
    hrp_split_study.py is the measurement. Its value is what it says about the
    published algorithm: the step most often written up as a wart is the step
    holding the method together.
    """
    n = cov.shape[0]
    weights = np.ones(n)
    stack = [n + len(link) - 1]           # the root

    while stack:
        node = stack.pop()
        if node < n:
            continue
        row = link[node - n]
        a, b = int(row[0]), int(row[1])
        left, right = cluster_leaves(link, a, n), cluster_leaves(link, b, n)
        v_left = cluster_variance(cov, left)
        v_right = cluster_variance(cov, right)
        alpha = 1.0 - v_left / (v_left + v_right)
        weights[left] *= alpha
        weights[right] *= 1.0 - alpha
        stack.extend([a, b])

    return weights


def hrp_weights(cov: np.ndarray, split: str = "list",
                method: str = "single") -> np.ndarray:
    """Hierarchical risk parity weights: long only, summing to one.

    No expected returns, no matrix inverse, no solver that can fail. Compare
    with min_variance_weights() in optimizer.py, which needs both a Sigma
    inverse and a constrained solve to produce something with the same two
    properties.

    split="list" is the published algorithm: bisect the quasi-diagonal order
    by count. method="single" is the published linkage. Both are the defaults
    so that every number already in RESULTS.md still refers to the same
    calculation.
    split="tree" descends the dendrogram instead (see tree_bisection()), and
    method="average" builds a tree worth descending (see linkage()). They are
    two halves of one change: either alone is worse than the published
    algorithm on real data, and hrp_study.py shows all four combinations.
    """
    cov = np.asarray(cov, dtype=float)
    if cov.shape[0] == 1:
        return np.ones(1)
    corr = correlation_from_covariance(cov)
    link = linkage(correlation_distance(corr), method=method)

    if split == "tree":
        w = tree_bisection(cov, link)
    elif split == "list":
        w = recursive_bisection(cov, quasi_diagonal_order(link))
    else:
        raise ValueError(f"split must be 'list' or 'tree', got {split!r}")

    return w / w.sum()
