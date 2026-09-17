"""Factor-model covariance: Sigma = B Omega B' + D.

Verify with:  pytest tests/test_factor_model.py

The problem this solves is counting. A sample covariance of N assets has
N(N+1)/2 free parameters - 1,275 for 50 assets - and a one-year estimation
window gives you 252 observations per asset. You are asking for more numbers
than the data contains, and the matrix tells you so: when N is an appreciable
fraction of T, its smallest eigenvalues collapse toward zero. Those directions
are pure sampling noise, and a min-variance optimizer, which is hunting for
low-variance directions, walks straight into them. It is not a bug in the
optimizer. The optimizer is doing exactly what it was told with a matrix that
lied about how little risk was available.

A factor model refuses to estimate the full matrix. It says returns are driven
by k common factors plus asset-specific noise:

    r = B f + e,     cov(f) = Omega,   cov(e) = D (diagonal)
    Sigma = B Omega B' + D

That is Nk + N parameters instead of N(N+1)/2 - for N=50, k=3, it is 200
instead of 1,275. The residual covariances are not estimated badly; they are
asserted to be zero. That assertion is wrong, and it is wrong in a direction
that costs far less than the noise it removes.

Two ways to get B are standard. Fundamental factor models (Barra, Axioma) hand
you the factors - market, size, value, momentum, industry - and regress. A
statistical factor model, which is what is here, extracts them from the returns
themselves by eigendecomposition. The tradeoff is the usual one: fundamental
factors are interpretable and stable, statistical factors need no outside data
and can't be wrong about which factors this particular universe actually has.
"""

import numpy as np

__all__ = [
    "pca_factor_covariance",
    "marchenko_pastur_edge",
    "num_significant_factors",
    "factor_variance_share",
]


def pca_factor_covariance(returns: np.ndarray, k: int) -> np.ndarray:
    """Statistical factor covariance from the top k principal components.

    `returns` is T observations x N assets, raw (not annualized) - the returned
    matrix carries the same units as np.cov of the input, so annualize it the
    same way you would a sample covariance.

    Mechanically: eigendecompose the sample covariance S, keep the k largest
    eigenpairs as the common part, and put whatever variance that leaves over
    into a diagonal.

        B      = V_k sqrt(Lambda_k)       loadings, factors scaled to unit variance
        common = B B'                     rank k by construction
        D      = diag(diag(S) - diag(common))
        Sigma  = common + D

    Two properties worth knowing, because they are what make this usable rather
    than merely clever:

    - diag(Sigma) equals diag(S) exactly. Individual asset volatilities are the
      one thing a sample estimate gets right (N numbers from N*T observations),
      so the model keeps them untouched and only restructures the correlations.
    - Sigma is positive semi-definite by construction, and its smallest
      eigenvalue is at least min(D). The collapsed noise directions that wreck
      the sample estimate are gone, because the model never estimated them.

    k = N reproduces S exactly (D becomes zero), which is the sanity check in
    the tests. k = 0 gives the diagonal of variances - no correlations at all -
    which is the same target `shrink_covariance` pulls toward at alpha = 1.
    """
    x = np.asarray(returns, dtype=float)
    if x.ndim != 2:
        raise ValueError("returns must be 2-D (T observations x N assets)")
    t_obs, n_assets = x.shape
    if not 0 <= k <= n_assets:
        raise ValueError(f"k must be in [0, {n_assets}], got {k}")
    if t_obs < 2:
        raise ValueError("need at least 2 observations")

    sample = np.cov(x, rowvar=False)
    sample = np.atleast_2d(sample)
    variances = np.diag(sample).copy()

    if k == 0:
        return np.diag(variances)

    eigvals, eigvecs = np.linalg.eigh(sample)      # ascending
    eigvals = eigvals[::-1][:k]
    eigvecs = eigvecs[:, ::-1][:, :k]

    # Negative eigenvalues can appear at the 1e-18 level from floating point on
    # a rank-deficient sample; sqrt of one is a NaN that poisons everything
    # downstream, so clip before taking the root.
    loadings = eigvecs * np.sqrt(np.maximum(eigvals, 0.0))
    common = loadings @ loadings.T

    # Specific variance is what the factors left behind. It can go slightly
    # negative for the same floating-point reason; a zero floor keeps Sigma PSD.
    specific = np.maximum(variances - np.diag(common), 0.0)
    return common + np.diag(specific)


def marchenko_pastur_edge(n_assets: int, n_obs: int) -> float:
    """Largest eigenvalue a *pure noise* correlation matrix can be expected to
    produce: lambda_plus = (1 + sqrt(N/T))^2.

    This is the useful half of the Marchenko-Pastur law. Take T independent
    draws of N uncorrelated series - no structure whatsoever - and compute the
    correlation matrix. Its eigenvalues do not come out at 1.0 each. They spread
    into a band, and the width of that band depends only on the ratio N/T.

    For 50 assets and 252 days the top of the band is 2.03: an eigenvalue of 2.0
    in that setting is not a factor, it is what noise looks like. For 50 assets
    and 4,000 days it is 1.24, so the same eigenvalue would be real. The number
    is a yardstick, not a test statistic - it tells you what a matrix of this
    shape produces when nothing at all is going on.

    Assumes the correlation matrix (unit variances). Pass sample variances
    through it and the units are wrong.
    """
    if n_assets <= 0 or n_obs <= 0:
        raise ValueError("n_assets and n_obs must be positive")
    return float((1.0 + np.sqrt(n_assets / n_obs)) ** 2)


def num_significant_factors(returns: np.ndarray, min_factors: int = 1) -> int:
    """How many eigenvalues of the correlation matrix clear the noise band.

    This is the standard way to pick k without fitting it: count the eigenvalues
    above `marchenko_pastur_edge`, because everything below is indistinguishable
    from a matrix with no structure at all.

    On equity data the first eigenvalue is always enormous - it is the market,
    and it typically carries a third of the total variance on its own. The next
    few are usually sectors. After that you are into the band.

    `min_factors` floors the answer at 1 rather than returning 0, because a
    zero-factor model is just the diagonal and the caller almost certainly
    wanted a covariance.
    """
    x = np.asarray(returns, dtype=float)
    t_obs, n_assets = x.shape
    sd = x.std(axis=0, ddof=1)
    if np.any(sd <= 0):
        raise ValueError("an asset has zero variance; drop it before estimating")
    corr = np.corrcoef(x, rowvar=False)
    eigvals = np.linalg.eigvalsh(corr)
    edge = marchenko_pastur_edge(n_assets, t_obs)
    return max(min_factors, int((eigvals > edge).sum()))


def factor_variance_share(returns: np.ndarray, k: int) -> float:
    """Fraction of total variance the top k principal components explain.

    Reported alongside k because the two answer different questions. k says how
    many directions are distinguishable from noise; this says how much of the
    universe's movement they account for. A universe of 50 stocks usually gives
    a small k and a large share - one market factor does most of the work.
    """
    x = np.asarray(returns, dtype=float)
    eigvals = np.linalg.eigvalsh(np.cov(x, rowvar=False))[::-1]
    total = eigvals.sum()
    if total <= 0:
        return 0.0
    return float(eigvals[:k].sum() / total)
