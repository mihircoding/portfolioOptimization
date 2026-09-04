"""Robust alternatives: shrinkage and risk-based weighting.

Verify with:  pytest tests/test_risk_parity.py

Everything here exists because mean-variance optimization is unstable. It treats
its inputs as known constants when they are noisy estimates, and it responds to
that noise by concentrating. These are the two standard responses: make the
inputs less noisy (shrinkage), or stop using the noisiest input at all
(risk-based weighting, which never touches expected returns).
"""

import numpy as np
from scipy.optimize import minimize


def shrink_covariance(sample_cov: np.ndarray, alpha: float) -> np.ndarray:
    """Shrink the sample covariance toward its own diagonal.

        shrunk = (1 - alpha) * sample_cov + alpha * diag(sample_cov)

    alpha = 0 is the raw sample estimate, alpha = 1 keeps the variances and zeros
    every correlation.

    The rationale is bias-variance. A sample covariance of N assets estimates
    N(N+1)/2 parameters — 1,275 of them for 50 assets — from data that rarely
    justifies it. The extreme eigenvalues are the worst estimated, and those are
    exactly the directions an optimizer loads into, because a spuriously low
    variance looks like free risk reduction. Pulling toward a structured target
    adds bias and removes much more variance.

    Ledoit-Wolf derive the alpha that minimizes expected squared error
    analytically; sklearn.covariance.LedoitWolf implements it.
    """
    target = np.diag(np.diag(sample_cov))
    return (1 - alpha) * sample_cov + alpha * target


def ledoit_wolf_alpha(returns: np.ndarray) -> float:
    """The analytic optimal alpha for shrink_covariance's target (shrink toward
    the diagonal of sample variances), from Ledoit & Wolf (2003), "Honey, I
    Shrunk the Sample Covariance Matrix." `shrink_covariance`'s docstring names
    this as the thing to use instead of a hand-picked alpha; this is that.

    `returns` is T observations x N assets, NOT annualized - the estimator's
    theory is asymptotic in T (number of observations), so it needs the raw
    daily count, not a covariance already scaled by 252.

    The idea: alpha trades off two errors. The sample covariance S is unbiased
    but each entry is noisy (variance shrinks like 1/T). The diagonal target F
    is badly biased (it says every correlation is exactly zero) but has no
    sampling noise at all. The optimal alpha is the one that minimizes expected
    squared error between the shrunk estimate and the unknown true covariance -
    which works out to a ratio of "how noisy is S" to "how far is S from F":

        alpha* = (sum of asymptotic variances of the off-diagonal entries of S)
                 / (T * sum of squared off-diagonal entries of S)

    More data (larger T) drives alpha toward 0 - the sample estimate needs less
    help. Fewer assets relative to observations does too. Clipped to [0, 1]
    because the asymptotic formula can technically overshoot in a finite sample.
    """
    X = np.asarray(returns, dtype=float)
    T, N = X.shape
    X = X - X.mean(axis=0)

    S = (X.T @ X) / T  # population covariance (divisor T, matches the theory)

    # pihat_ij = (1/T) * sum_t (x_it * x_jt - s_ij)^2 : the asymptotic variance
    # of each sample covariance entry. Built via one (T, N, N) array - fine at
    # this scale (a handful of assets), not how you'd do this for N in the
    # hundreds.
    outer = X[:, :, None] * X[:, None, :]           # (T, N, N)
    pihat_matrix = ((outer - S) ** 2).mean(axis=0)   # (N, N)
    pihat = pihat_matrix.sum()
    rhohat = np.trace(pihat_matrix)  # diagonal target's entries ARE the sample
                                      # variances, so their asymptotic covariance
                                      # with the target collapses to Var(s_ii)

    off_diag_sq_sum = (S ** 2).sum() - np.sum(np.diag(S) ** 2)
    if off_diag_sq_sum <= 0:
        return 1.0  # sample is already exactly diagonal - fully "shrunk" already

    kappa = (pihat - rhohat) / off_diag_sq_sum
    return float(np.clip(kappa / T, 0.0, 1.0))


def inverse_vol_weights(cov: np.ndarray) -> np.ndarray:
    """w_i proportional to 1 / vol_i, normalized to sum to 1.

    Naive risk parity: equalizes standalone risk contributions and ignores
    correlations entirely. It is the correct ERC solution when all correlations
    are equal, and a decent starting point for the solver otherwise.
    """
    inv_vol = 1 / np.sqrt(np.diag(cov))
    return inv_vol / inv_vol.sum()


def risk_contributions(w: np.ndarray, cov: np.ndarray) -> np.ndarray:
    """Each asset's share of total portfolio variance. Sums to 1 exactly.

        marginal_i = (Σw)_i                      # d(variance)/d(w_i), up to 2
        rc_i       = w_i * marginal_i / (w' Σ w)

    The decomposition is exact rather than approximate — portfolio variance is
    homogeneous of degree 2 in w, so Euler's theorem makes the parts sum to the
    whole with nothing left over.

    This is THE risk-report number. "Asset X is 4% of the book and 38% of the
    risk" is this quantity, and it is usually the first thing that reveals a
    portfolio is not diversified in the way its weights suggest.
    """
    variance = w @ cov @ w
    if variance <= 0:
        return np.zeros_like(w)
    return w * (cov @ w) / variance


def equal_risk_contribution_weights(cov: np.ndarray) -> np.ndarray:
    """Weights where every asset contributes equally to risk: rc_i = 1/n.

    Diversifies RISK rather than capital. Equal weighting looks balanced and
    isn't — a 20%-vol asset at 10% of the book carries far more risk than a
    5%-vol asset at the same weight.

    No closed form except when all correlations are equal (then it reduces to
    inverse-vol). Solved numerically by minimizing the dispersion of risk
    contributions. Starting from inverse-vol weights matters: the objective is
    non-convex, and a good start keeps SLSQP out of trouble.

    Also note ERC needs no expected returns — the same robustness argument as
    min-variance, one step further.
    """
    n = len(cov)

    def dispersion(w: np.ndarray) -> float:
        rc = risk_contributions(w, cov)
        return float(((rc[:, None] - rc[None, :]) ** 2).sum())

    result = minimize(
        dispersion,
        x0=inverse_vol_weights(cov),
        method="SLSQP",
        bounds=[(1e-8, 1.0)] * n,  # strictly positive: rc_i = 1/n needs w_i > 0
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        options={"maxiter": 2000, "ftol": 1e-16},
    )
    if not result.success:
        raise RuntimeError(f"ERC optimization failed: {result.message}")
    return result.x
