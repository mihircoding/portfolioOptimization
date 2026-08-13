import numpy as np
import pytest


@pytest.fixture
def two_asset_cov():
    """sigma1=10%, sigma2=20%, rho=0.3 -> closed-form min-var weights exist."""
    s1, s2, rho = 0.10, 0.20, 0.3
    c12 = rho * s1 * s2
    return np.array([[s1**2, c12], [c12, s2**2]])


@pytest.fixture
def three_asset():
    """Three uncorrelated assets, distinct vols and returns."""
    mu = np.array([0.05, 0.08, 0.12])
    cov = np.diag([0.10**2, 0.15**2, 0.25**2])
    return mu, cov
