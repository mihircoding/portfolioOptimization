import numpy as np
import pandas as pd
import pytest

from src.returns import annualized_cov, annualized_mean, daily_returns


@pytest.fixture
def prices():
    idx = pd.bdate_range("2023-01-02", periods=4)
    return pd.DataFrame({"A": [100.0, 101.0, 100.0, 102.0],
                         "B": [50.0, 50.0, 51.0, 51.0]}, index=idx)


class TestReturns:
    def test_daily_returns_values(self, prices):
        r = daily_returns(prices)
        assert len(r) == 3  # first NaN row dropped
        assert r["A"].iloc[0] == pytest.approx(0.01)
        assert r["B"].iloc[0] == pytest.approx(0.0)

    def test_annualized_mean_scaling(self, prices):
        r = daily_returns(prices)
        mu = annualized_mean(prices)
        assert mu["A"] == pytest.approx(r["A"].mean() * 252)

    def test_annualized_cov_scaling_and_symmetry(self, prices):
        r = daily_returns(prices)
        cov = annualized_cov(prices)
        assert cov.loc["A", "A"] == pytest.approx(r["A"].var(ddof=1) * 252)
        assert cov.loc["A", "B"] == pytest.approx(cov.loc["B", "A"])
