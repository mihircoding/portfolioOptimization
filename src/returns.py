"""Return and covariance estimation.

Verify with:  pytest tests/test_returns.py

Conventions everything downstream depends on:
  - simple daily returns:  r_t = P_t / P_{t-1} - 1   (pct_change, drop first NaN)
  - annualized mean return:      daily mean * 252
  - annualized covariance:       daily covariance * 252
  - annualized volatility:       daily std * sqrt(252)   <- sqrt! variance scales
                                                            with time, vol doesn't
"""

import numpy as np
import pandas as pd

TRADING_DAYS = 252


def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """Simple daily returns; the leading NaN row is dropped.

    Simple rather than log returns because portfolio return is a weighted sum of
    asset returns only in simple space — log returns don't aggregate across
    assets. (They do aggregate across time, which is why risk models often use
    them anyway. Know which property you need.)
    """
    return prices.pct_change().dropna()


def annualized_mean(prices: pd.DataFrame) -> pd.Series:
    """Annualized expected return per asset (the mu vector).

    Scales linearly with time: 252 independent daily draws, so 252x the mean.
    """
    return daily_returns(prices).mean() * TRADING_DAYS


def annualized_cov(prices: pd.DataFrame) -> pd.DataFrame:
    """Annualized sample covariance matrix (ddof=1, pandas default).

    Variance scales linearly with time under the i.i.d. assumption, so the whole
    matrix scales by 252. Volatility, being a square root, scales by sqrt(252) —
    mixing these two up is the single most common unit error in the field.
    """
    return daily_returns(prices).cov() * TRADING_DAYS
