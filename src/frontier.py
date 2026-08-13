"""Trace the efficient frontier.

Verify with:  pytest tests/test_frontier.py
"""

import numpy as np
import pandas as pd
from scipy.optimize import minimize


def efficient_frontier(mu: np.ndarray, cov: np.ndarray, n_points: int = 30,
                       long_only: bool = True) -> pd.DataFrame:
    """For each target return, the minimum-variance portfolio achieving it.

    The frontier is the set of portfolios you cannot improve on: for their level
    of risk there is no higher return, and for their return there is no lower
    risk. Everything below it is dominated. Its upper branch is the only part
    anyone should hold, and the tangency (max-Sharpe) portfolio sits on it.

    Targets sweep [mu.min(), mu.max()]: with long-only weights you cannot achieve
    a return outside the assets' own range, so that is exactly the feasible span.
    Allowing shorts extends it in both directions.

    Returns columns ['target_return', 'volatility', 'weights'], skipping any
    target the optimizer fails to reach.
    """
    n = len(mu)
    bounds = [(0.0, 1.0)] * n if long_only else [(None, None)] * n
    rows = []

    for target in np.linspace(mu.min(), mu.max(), n_points):
        constraints = [
            {"type": "eq", "fun": lambda w: w.sum() - 1.0},
            # bind target as a default arg: a bare closure over the loop variable
            # would make every constraint reference the LAST target
            {"type": "eq", "fun": lambda w, t=target: w @ mu - t},
        ]
        result = minimize(lambda w: w @ cov @ w, x0=np.full(n, 1 / n),
                          method="SLSQP", bounds=bounds, constraints=constraints,
                          options={"maxiter": 1000, "ftol": 1e-12})
        if not result.success:
            continue
        rows.append({
            "target_return": target,
            "volatility": float(np.sqrt(result.x @ cov @ result.x)),
            "weights": result.x,
        })

    return pd.DataFrame(rows)
