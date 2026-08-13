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
