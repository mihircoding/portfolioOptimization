import pandas as pd
import pytest

from src.pairs import engle_granger_pvalue, find_cointegrated_pairs, hedge_ratio


class TestHedgeRatio:
    def test_recovers_planted_beta(self, cointegrated_pair):
        y, x = cointegrated_pair
        beta = hedge_ratio(y, x)
        assert beta == pytest.approx(2.5, abs=0.05)

    def test_returns_plain_float(self, cointegrated_pair):
        y, x = cointegrated_pair
        assert isinstance(hedge_ratio(y, x), float)


class TestEngleGranger:
    def test_flags_cointegrated_pair(self, cointegrated_pair):
        y, x = cointegrated_pair
        assert engle_granger_pvalue(y, x) < 0.05

    def test_rejects_independent_walks(self, independent_walks):
        a, b = independent_walks
        assert engle_granger_pvalue(a, b) > 0.05


class TestPairScan:
    def test_finds_the_planted_pair(self, cointegrated_pair, independent_walks):
        y, x = cointegrated_pair
        a, b = independent_walks
        prices = pd.concat([y, x, a, b], axis=1)
        prices.columns = ["y", "x", "a", "b"]

        result = find_cointegrated_pairs(prices, max_pvalue=0.05)

        assert list(result.columns) == ["a", "b", "pvalue", "beta"]
        top = result.iloc[0]
        assert {top["a"], top["b"]} == {"y", "x"}
        # sorted ascending by pvalue
        assert result["pvalue"].is_monotonic_increasing
