"""Black-Litterman: start from what the market already believes.

Verify with:  pytest tests/test_black_litterman.py

RESULTS.md's headline is that max-Sharpe finishes behind equal weighting out
of sample. The diagnosis is always the same: mu. Sample mean returns are so
noisy that an optimizer handed them will chase whichever asset got lucky in
the estimation window, and no amount of care in the optimizer fixes a bad
input.

Black-Litterman (1990) attacks the input instead of the optimizer. The idea
is to run mean-variance backwards. If you assume the market portfolio - what
everyone actually holds, in the proportions they hold it - is the optimal
portfolio for someone, then there is exactly one vector of expected returns
that would have produced it:

    Pi = delta * Sigma * w_market                       (reverse optimization)

That vector is the *equilibrium* return. It is not a forecast; it is a
restatement of the market's positioning in return space. Its useful property
is that it contains no estimation error from a sample mean, because no sample
mean was used to build it.

You then blend in whatever you actually believe, as views, and get a
posterior:

    mu_BL = [(tau*Sigma)^-1 + P' Omega^-1 P]^-1
            [(tau*Sigma)^-1 Pi + P' Omega^-1 Q]

with P selecting the assets a view is about, Q the view's magnitude, and
Omega how unsure you are of it. Two limits make the formula readable:
no views at all returns Pi unchanged, and a view you are certain about binds
exactly. Everything in between is a precision-weighted average of the two,
which is all a Bayesian posterior of two Gaussians ever is.

The practical payoff is that a view about one asset moves the whole portfolio
coherently. Say gold does well and Sigma knows gold and bonds move together,
and the posterior tilts toward bonds too - without anyone writing a view about
bonds. Plain max-Sharpe has no such mechanism: it only knows what you typed
into mu.
"""

import numpy as np

from .optimizer import max_utility_weights

DEFAULT_TAU = 0.05


def implied_risk_aversion(market_excess_return: float, market_variance: float) -> float:
    """delta = (E[r_market] - rf) / var(r_market).

    Both arguments annualized and in the same units. This is the market's
    price of risk: how much extra return it demands per unit of variance.
    Typical values on equity indices land between 2 and 4; anything wildly
    outside that usually means the two arguments disagree about annualization.
    """
    if market_variance <= 0:
        raise ValueError("market variance must be positive")
    return market_excess_return / market_variance


def equilibrium_returns(cov: np.ndarray, w_market: np.ndarray,
                        risk_aversion: float) -> np.ndarray:
    """Pi = delta * Sigma * w_market — mean-variance run backwards.

    Note what is NOT here: any historical average return. Pi is built from the
    covariance matrix (which converges in months) and the market's own weights
    (which are observable), and never from a sample mean (which needs decades).
    That is the entire reason this is worth doing.
    """
    w_market = np.asarray(w_market, dtype=float)
    if not np.isclose(w_market.sum(), 1.0):
        raise ValueError(f"market weights must sum to 1, got {w_market.sum():.4f}")
    return risk_aversion * (np.asarray(cov, dtype=float) @ w_market)


def default_view_uncertainty(cov: np.ndarray, P: np.ndarray,
                             tau: float = DEFAULT_TAU) -> np.ndarray:
    """Omega = diag(P (tau*Sigma) P'), the He-Litterman convention.

    Omega is the covariance of the views' error terms, and picking it by hand
    for every view is exactly the kind of free parameter that lets a model be
    tuned until it says what you wanted. This convention removes the choice:
    a view's uncertainty is set to the prior variance of the same portfolio
    the view is about, so views about volatile combinations are automatically
    trusted less. It also has a convenient consequence - the answer stops
    depending on tau, because tau then appears in both the prior and the view
    precision and cancels.

    Diagonal, i.e. views are assumed independent of each other. Correlated
    views are expressible (fill in the off-diagonals) and are rarely worth
    the trouble.
    """
    P = np.atleast_2d(np.asarray(P, dtype=float))
    return np.diag(np.diag(P @ (tau * np.asarray(cov, dtype=float)) @ P.T))


def posterior(cov: np.ndarray, pi: np.ndarray, P=None, Q=None,
              tau: float = DEFAULT_TAU, omega: np.ndarray | None = None
              ) -> tuple[np.ndarray, np.ndarray]:
    """Blend equilibrium with views. Returns (mu_BL, cov_BL).

    P is (k x n): one row per view, selecting which assets it is about. A row
    of [0, 0, 1, 0, 0] is an absolute view on asset 3; a row of
    [1, 0, -1, 0, 0] is a relative view, "asset 1 beats asset 3 by Q".
    Q is (k,): the magnitude of each view, annualized like everything else.

    cov_BL = Sigma + M, where M is the posterior covariance of the mean. The
    second term is usually left out of textbook treatments and matters: it is
    the admission that mu_BL is itself an estimate. Adding it makes the
    optimizer slightly less willing to concentrate, which is the right
    direction for a quantity nobody knows precisely.

    With no views this returns (pi, Sigma + tau*Sigma) — the prior, untouched.
    """
    cov = np.asarray(cov, dtype=float)
    pi = np.asarray(pi, dtype=float)
    tau_sigma_inv = np.linalg.inv(tau * cov)

    if P is None or Q is None or len(np.atleast_1d(Q)) == 0:
        M = np.linalg.inv(tau_sigma_inv)
        return pi.copy(), cov + M

    P = np.atleast_2d(np.asarray(P, dtype=float))
    Q = np.atleast_1d(np.asarray(Q, dtype=float))
    if P.shape[0] != Q.shape[0]:
        raise ValueError(f"P has {P.shape[0]} views but Q has {Q.shape[0]}")
    if P.shape[1] != cov.shape[0]:
        raise ValueError(f"P has {P.shape[1]} columns for {cov.shape[0]} assets")

    if omega is None:
        omega = default_view_uncertainty(cov, P, tau)
    omega = np.atleast_2d(np.asarray(omega, dtype=float))
    omega_inv = np.linalg.inv(omega)

    M = np.linalg.inv(tau_sigma_inv + P.T @ omega_inv @ P)
    mu_bl = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
    return mu_bl, cov + M


def black_litterman_weights(cov: np.ndarray, w_market: np.ndarray,
                            risk_aversion: float, P=None, Q=None,
                            tau: float = DEFAULT_TAU,
                            omega: np.ndarray | None = None,
                            long_only: bool = True,
                            use_posterior_cov: bool = True) -> np.ndarray:
    """Equilibrium, plus views, forward-optimized back into weights.

    The round trip is the point: reverse-optimize the market to get Pi, blend
    in views to get mu_BL, then forward-optimize mu_BL at the same risk
    aversion. With no views that has to return the market portfolio, or the
    two halves are solving different problems.

    use_posterior_cov: whether the forward step uses Sigma + M or plain Sigma.
    True is the honest choice - M is the uncertainty in mu_BL itself, and
    ignoring it tells the optimizer a number it should not believe. It also
    breaks the exact round trip: with no views the unconstrained answer comes
    out as w_market / (1 + tau), scaled down because the extra uncertainty
    makes the same risk aversion want less risk. Under a fully-invested
    constraint that missing 1/(1+tau) gets re-invested along the minimum
    variance direction, so the no-view answer is the market portfolio nudged
    slightly toward the quiet assets rather than the market portfolio exactly.
    Set False to get the textbook identity back and see the difference.
    """
    pi = equilibrium_returns(cov, w_market, risk_aversion)
    mu_bl, cov_bl = posterior(cov, pi, P, Q, tau=tau, omega=omega)
    forward_cov = cov_bl if use_posterior_cov else np.asarray(cov, dtype=float)
    return max_utility_weights(mu_bl, forward_cov, risk_aversion, long_only=long_only)


def absolute_view(n_assets: int, asset: int, annual_return: float) -> tuple:
    """'Asset i will return X.' Returns (P, Q) ready for posterior()."""
    P = np.zeros((1, n_assets))
    P[0, asset] = 1.0
    return P, np.array([annual_return])


def relative_view(n_assets: int, outperformer: int, underperformer: int,
                  annual_spread: float) -> tuple:
    """'Asset i beats asset j by X.' Returns (P, Q).

    Relative views are the ones practitioners actually hold. Nobody has a
    confident opinion about the level of equity returns next year; plenty of
    people have one about equities versus bonds.
    """
    P = np.zeros((1, n_assets))
    P[0, outperformer] = 1.0
    P[0, underperformer] = -1.0
    return P, np.array([annual_spread])
