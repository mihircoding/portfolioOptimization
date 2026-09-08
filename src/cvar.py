"""CVaR (Conditional Value-at-Risk) optimization.

Verify with:  pytest tests/test_cvar.py

Everything in optimizer.py and risk_parity.py optimizes against VARIANCE,
which is symmetric: it penalizes a surprise $10 gain exactly as much as a
surprise $10 loss. README's own "what I'd build next" list names this
directly - "nobody actually minds the upside." CVaR fixes that by looking
only at the downside tail, and by working from the actual historical
scenarios rather than a single (mean, covariance) summary of them - so a
fat-tailed, crash-prone asset and a smooth one with the same mean and
variance are no longer indistinguishable to the optimizer.
"""

import numpy as np
from scipy.optimize import linprog


def min_cvar_weights(returns: np.ndarray, alpha: float = 0.95,
                     long_only: bool = True) -> np.ndarray:
    """The portfolio minimizing CVaR_alpha, via Rockafellar & Uryasev (2000).

    `returns` is T scenarios x N assets of RAW (not annualized) periodic
    returns - daily returns from a training window, in this project's usage.

    CVaR_alpha (a.k.a. Expected Shortfall) is the average loss in the worst
    (1 - alpha) fraction of scenarios - "when things are bad, how bad on
    average." Its textbook definition - sort the portfolio's scenario
    returns, average the worst (1-alpha)*T of them - is exactly right but
    useless as an optimization objective: sorting isn't differentiable, and
    the SET of which scenarios are "worst" changes discontinuously as the
    weights move, which breaks every gradient-based method used elsewhere
    in this project.

    Rockafellar and Uryasev's result is that minimizing CVaR is exactly
    equivalent to a LINEAR PROGRAM in an expanded variable space - no
    sorting, no discontinuity:

        minimize_{w, zeta, u}   zeta + 1/((1-alpha)*T) * sum_t(u_t)
        subject to:             u_t >= -(w . r_t) - zeta   for every scenario t
                                 u_t >= 0
                                 sum(w) = 1

    w are the portfolio weights, r_t is scenario t's vector of asset
    returns, and zeta and u are auxiliary variables with no meaning of
    their own until you solve it - at the optimum, zeta lands exactly on
    the portfolio's Value-at-Risk (see var_of_weights), and each u_t is the
    slack soaking up how far scenario t's loss exceeds that VaR (zero for
    every scenario that isn't in the tail). Minimizing that sum is
    minimizing the tail average, without ever sorting anything.

    Long-only by default, matching every other optimizer in this project;
    pass long_only=False to allow shorting (a leverage cap would still be
    needed for that to be well-posed at scale, but isn't needed here -
    sum(w)=1 alone keeps this problem bounded, unlike max-Sharpe's
    scale-invariant objective).
    """
    R = np.asarray(returns, dtype=float)
    T, n = R.shape
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")

    n_vars = n + 1 + T  # [w (n), zeta (1), u (T)]
    c = np.zeros(n_vars)
    c[n] = 1.0
    c[n + 1:] = 1.0 / ((1.0 - alpha) * T)

    # u_t >= -(w.r_t) - zeta  <=>  -r_t.w - zeta - u_t <= 0
    A_ub = np.zeros((T, n_vars))
    A_ub[:, :n] = -R
    A_ub[:, n] = -1.0
    A_ub[np.arange(T), n + 1 + np.arange(T)] = -1.0
    b_ub = np.zeros(T)

    A_eq = np.zeros((1, n_vars))
    A_eq[0, :n] = 1.0
    b_eq = [1.0]

    w_bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    bounds = w_bounds + [(None, None)] + [(0.0, None)] * T

    result = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq,
                     bounds=bounds, method="highs")
    if not result.success:
        raise RuntimeError(f"CVaR optimization failed: {result.message}")
    return result.x[:n]


def var_of_weights(w: np.ndarray, returns: np.ndarray, alpha: float = 0.95) -> float:
    """Value-at-Risk: the loss that is exceeded only (1-alpha) of the time -
    the empirical (1-alpha) quantile of portfolio LOSSES (positive number =
    a loss). This is the threshold; cvar_of_weights is the average beyond
    it. Reported as its own function (not just inside the LP) so "VaR vs
    CVaR" can be shown side by side for ANY weights, not only the ones
    min_cvar_weights happens to produce.
    """
    R = np.asarray(returns, dtype=float)
    losses = -(R @ w)
    return float(np.quantile(losses, alpha))


def cvar_of_weights(w: np.ndarray, returns: np.ndarray, alpha: float = 0.95) -> float:
    """CVaR_alpha of a GIVEN weight vector, computed directly - once w is
    fixed, no LP is needed: sort the portfolio's scenario losses and
    average the worst (1-alpha) fraction. This is what scores portfolios
    min_cvar_weights didn't produce (equal weight, min-variance, ...) on
    the same footing, and what the LP above is provably equivalent to
    minimizing.
    """
    R = np.asarray(returns, dtype=float)
    losses = -(R @ w)
    T = len(losses)
    n_tail = max(1, int(np.ceil((1.0 - alpha) * T)))
    worst = np.sort(losses)[-n_tail:]
    return float(worst.mean())
