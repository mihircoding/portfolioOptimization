"""Mean-variance optimization with real-world constraints.

Verify with:  pytest tests/test_optimizer.py

Every problem here is the same scipy call with a different objective:

    minimize(objective, x0=equal_weight, method="SLSQP",
             bounds=..., constraints=[budget constraint])

SLSQP because these are smooth, nonlinearly constrained problems of low
dimension — exactly what it's for. Always check result.success: an optimizer
that silently fails returns x0, and "why is every portfolio equal weight?" is
the resulting bug.
"""

import numpy as np
from scipy.optimize import minimize


def _setup(n: int, long_only: bool) -> dict:
    """Shared scipy configuration: fully invested, optionally no shorting."""
    return {
        "method": "SLSQP",
        "bounds": [(0.0, 1.0)] * n if long_only else [(None, None)] * n,
        "constraints": [{"type": "eq", "fun": lambda w: w.sum() - 1.0}],
        "options": {"maxiter": 1000, "ftol": 1e-12},
    }


def portfolio_performance(w: np.ndarray, mu: np.ndarray, cov: np.ndarray,
                          rf: float = 0.0) -> tuple[float, float, float]:
    """Return (expected_return, volatility, sharpe) for weights w.

    Note vol = sqrt(w' Σ w), not a weighted average of the individual vols. The
    difference between those two is precisely diversification, and it is the
    entire subject.
    """
    ret = float(w @ mu)
    vol = float(np.sqrt(w @ cov @ w))
    sharpe = (ret - rf) / vol if vol > 0 else 0.0
    return ret, vol, sharpe


def min_variance_weights(cov: np.ndarray, long_only: bool = True) -> np.ndarray:
    """The global minimum-variance portfolio.

    Minimize w' Σ w subject to the weights summing to 1.

    mu is deliberately not an argument. This portfolio needs no return forecasts
    at all — and expected returns are estimated far less reliably than
    covariances, so an optimizer that never sees them cannot be wrecked by them.
    That is why practitioners trust min-variance more than max-Sharpe.
    """
    n = len(cov)
    result = minimize(lambda w: w @ cov @ w, x0=np.full(n, 1 / n), **_setup(n, long_only))
    if not result.success:
        raise RuntimeError(f"min-variance optimization failed: {result.message}")
    return result.x

def min_variance_weights_analytic(cov: np.ndarray) -> np.ndarray:
    """The unconstrained global minimum-variance portfolio, in closed form:

        w = Sigma^-1 1 / (1' Sigma^-1 1)

    Same objective as `min_variance_weights(long_only=False)`, solved by linear
    algebra instead of by SLSQP. Two reasons it exists as its own function.

    Speed: the studies in `factor_study.py` re-solve this a few hundred times
    across several covariance estimators, and a 50-asset SLSQP solve with no
    bounds to guide it is both slow and prone to returning `success=False`.

    Honesty: the closed form makes it obvious where the damage comes from. The
    weights are an inverse covariance applied to a vector of ones, so any
    direction the estimate claims has near-zero variance gets a near-infinite
    weight. That is the whole estimation-error story in one line, and a
    numerical optimizer hides it behind an iteration count.

    Uses `solve` rather than forming the inverse - the inverse of a badly
    conditioned matrix is exactly the object you least want to compute
    explicitly.
    """
    n = len(cov)
    ones = np.ones(n)
    z = np.linalg.solve(cov, ones)
    return z / z.sum()


def max_sharpe_weights(mu: np.ndarray, cov: np.ndarray, rf: float = 0.0,
                       long_only: bool = True) -> np.ndarray:
    """The tangency portfolio: the highest Sharpe ratio available.

    scipy only minimizes, so minimize the negative Sharpe.

    Be suspicious of the output. Max-Sharpe depends on mu, and mu is the least
    reliable input in finance — you need decades of data to estimate a mean
    return to any useful precision, while a covariance converges in months. The
    optimizer treats every input as certain, so it concentrates aggressively into
    whichever asset had the best in-sample luck. See RESULTS.md.
    """
    n = len(mu)
    result = minimize(lambda w: -portfolio_performance(w, mu, cov, rf)[2],
                      x0=np.full(n, 1 / n), **_setup(n, long_only))
    if not result.success:
        raise RuntimeError(f"max-Sharpe optimization failed: {result.message}")
    return result.x


def max_utility_weights(mu: np.ndarray, cov: np.ndarray, risk_aversion: float,
                        long_only: bool = True) -> np.ndarray:
    """Maximize mean-variance utility:  w'mu - (delta/2) w'Sigma w.

    The other objectives here optimize a ratio (Sharpe) or a pure risk number
    (variance, CVaR). This one optimizes the quantity Markowitz actually wrote
    down: return minus a penalty on variance, with `risk_aversion` (delta)
    setting the exchange rate between the two.

    It exists because Black-Litterman needs it. BL's whole construction runs
    through delta - the posterior returns come from reverse-optimizing the
    market portfolio at a given risk aversion, so the forward step has to
    invert the same objective or the round trip won't close. Unconstrained,
    the answer is the closed form w = (delta*Sigma)^-1 mu; this is that same
    problem with a budget constraint and optional no-shorting, which is what
    the rest of this repo assumes.
    """
    n = len(mu)
    result = minimize(lambda w: -(w @ mu) + 0.5 * risk_aversion * (w @ cov @ w),
                      x0=np.full(n, 1 / n), **_setup(n, long_only))
    if not result.success:
        raise RuntimeError(f"max-utility optimization failed: {result.message}")
    return result.x


def max_sharpe_turnover_penalized(mu: np.ndarray, cov: np.ndarray, w_prev: np.ndarray,
                                  penalty: float, rf: float = 0.0,
                                  long_only: bool = True) -> np.ndarray:
    """Max-Sharpe, penalized for straying from last period's weights.

    Plain max-Sharpe re-optimizes every period as if trading were free, so it
    fully re-chases the new mu/cov estimate each time - see RESULTS.md, 16.2%
    turnover a year for not much return. This adds a quadratic penalty on the
    move away from w_prev to the objective:

        minimize   -sharpe(w)  +  penalty * sum((w - w_prev)^2)

    A quadratic penalty rather than the more standard turnover-linear cost
    (|w - w_prev|, i.e. proportional to trade size) keeps the problem smooth
    for SLSQP with no extra slack variables. It still buys less turnover at
    some cost in Sharpe, which is the point; penalty=0 recovers
    max_sharpe_weights() exactly.
    """
    n = len(mu)

    def objective(w):
        _, _, sharpe = portfolio_performance(w, mu, cov, rf)
        return -sharpe + penalty * float(np.sum((w - w_prev) ** 2))

    result = minimize(objective, x0=w_prev, **_setup(n, long_only))
    if not result.success:
        raise RuntimeError(f"turnover-penalized optimization failed: {result.message}")
    return result.x
