"""The portfolio-of-pairs analysis in portfolio.py.

The numbers it produces about the real scan can't be asserted - the answer
isn't known in advance, which is the point of running it. What can be
asserted is that the machinery does what the write-up says it does: equal
weighting is equal weighting, the diversification formula matches simulated
series with a known correlation, the rank correlation is a rank correlation,
and the p-value buckets partition the survivors without losing or
double-counting one.

The last one matters more than it sounds. A bucket table that quietly drops
the 930th pair, or counts the boundary pair twice, would still produce a
plausible-looking table - and the table is the evidence for the claim that
the screen doesn't rank.
"""

import numpy as np
import pandas as pd
import pytest

from portfolio import (average_correlation, book, diversification_check,
                       does_the_screen_predict, leg_concentration,
                       rank_correlation)
from scan import TRADING_DAYS


def _returns(n_cols=4, n_days=500, sd=0.01, mean=0.0, seed=0, corr=0.0):
    """Daily returns with a known mean, vol and equicorrelation."""
    rng = np.random.default_rng(seed)
    common = rng.normal(0, 1, n_days)
    cols = {}
    for i in range(n_cols):
        idio = rng.normal(0, 1, n_days)
        z = np.sqrt(corr) * common + np.sqrt(1 - corr) * idio
        cols[f"P{i}"] = mean + sd * z
    idx = pd.bdate_range("2021-01-04", periods=n_days)
    return pd.DataFrame(cols, index=idx)


class TestBook:
    def test_one_column_book_is_that_column(self):
        r = _returns(n_cols=1)
        stats = book(r)
        assert stats["n_pairs"] == 1
        assert stats["ann_return"] == pytest.approx(
            float(r["P0"].mean()) * TRADING_DAYS)
        assert stats["total_return"] == pytest.approx(
            float((1 + r["P0"]).prod() - 1))

    def test_equal_weight_is_the_plain_average(self):
        r = _returns(n_cols=5)
        stats = book(r)
        expected_vol = float(r.mean(axis=1).std(ddof=1) * np.sqrt(TRADING_DAYS))
        assert stats["ann_vol"] == pytest.approx(expected_vol)

    def test_drawdown_is_negative_or_zero(self):
        assert book(_returns())["max_drawdown"] <= 0

    def test_equal_risk_ignores_a_rescaled_column(self):
        """The whole point of the equal-risk diagnostic: doubling one pair's
        volatility must not double its say in the book."""
        r = _returns(n_cols=4, seed=3)
        louder = r.copy()
        louder["P0"] = louder["P0"] * 4.0
        assert book(louder, equal_risk=True)["sharpe"] == pytest.approx(
            book(r, equal_risk=True)["sharpe"], abs=1e-9)
        # ...whereas equal DOLLAR weighting very much does notice
        assert book(louder)["sharpe"] != pytest.approx(book(r)["sharpe"], abs=1e-6)


class TestCorrelation:
    def test_independent_series_average_near_zero(self):
        rho = average_correlation(_returns(n_cols=30, n_days=2000, corr=0.0))
        assert abs(rho) < 0.05

    def test_identical_series_average_one(self):
        r = _returns(n_cols=1, n_days=400)
        same = pd.concat([r.rename(columns={"P0": f"C{i}"}) for i in range(5)], axis=1)
        assert average_correlation(same) == pytest.approx(1.0)

    def test_known_equicorrelation_is_recovered(self):
        rho = average_correlation(_returns(n_cols=25, n_days=4000, corr=0.30))
        assert rho == pytest.approx(0.30, abs=0.05)


class TestDiversificationArithmetic:
    def test_prediction_matches_the_equal_risk_book(self):
        """sqrt(N) / sqrt(1 + (N-1)rho) is a statement about equally-risked
        series, so it is the equal-risk book it should match. If this drifts,
        the write-up's claim that the book's Sharpe is diversification
        arithmetic stops being supported."""
        r = _returns(n_cols=20, n_days=3000, mean=0.0004, corr=0.10, seed=5)
        div = diversification_check(r, book(r))
        assert div["predicted_book_sharpe"] == pytest.approx(
            div["equal_risk_book_sharpe"], abs=0.25)

    def test_uncorrelated_series_beat_a_single_one(self):
        r = _returns(n_cols=16, n_days=3000, mean=0.0003, corr=0.0, seed=6)
        div = diversification_check(r, book(r))
        assert div["predicted_book_sharpe"] > 2 * div["mean_pair_sharpe"] > 0

    def test_share_measures_are_fractions_and_can_disagree(self):
        """Positive Sharpe and positive total return are not the same test -
        a series with a positive mean can still compound to a loss. The
        write-up quotes both, so both have to be computed, and the gap
        between them is volatility drag rather than a contradiction."""
        r = _returns(n_cols=10, mean=0.0002, sd=0.02, seed=8)
        div = diversification_check(r, book(r))
        assert 0.0 <= div["share_positive_sharpe"] <= 1.0
        assert 0.0 <= div["share_profitable"] <= 1.0
        assert div["share_profitable"] <= div["share_positive_sharpe"]


class TestRankCorrelation:
    def _frame(self, pvalues, sharpes):
        return pd.DataFrame({"pvalue": pvalues, "sharpe": sharpes})

    def test_perfectly_aligned_ranks(self):
        rc = rank_correlation(self._frame(np.arange(50.0), np.arange(50.0)))
        assert rc["spearman"] == pytest.approx(1.0)

    def test_perfectly_inverted_ranks(self):
        rc = rank_correlation(self._frame(np.arange(50.0), np.arange(50.0)[::-1]))
        assert rc["spearman"] == pytest.approx(-1.0)

    def test_z_scales_with_sample_size(self):
        """Same correlation, more pairs, larger z - which is what makes the
        real scan's z = -1.6 over 930 pairs a meaningful 'no'."""
        small = rank_correlation(self._frame(np.arange(50.0), np.arange(50.0)))
        large = rank_correlation(self._frame(np.arange(500.0), np.arange(500.0)))
        assert abs(large["z"]) > abs(small["z"])


class TestBuckets:
    def _scan(self, n=100, seed=1):
        rng = np.random.default_rng(seed)
        return pd.DataFrame({
            "a": [f"T{i % 10}" for i in range(n)],
            "b": [f"T{(i // 10) % 10 + 10}" for i in range(n)],
            "pvalue": rng.uniform(0, 0.05, n),
            "sharpe": rng.normal(0, 0.4, n),
        })

    def test_buckets_partition_every_pair_exactly_once(self):
        scan = self._scan(n=97)          # deliberately not divisible by 5
        returns = _returns(n_cols=97, n_days=300)
        returns.columns = [f"{r.a}/{r.b}" for r in scan.itertuples()]
        buckets = does_the_screen_predict(scan, returns)
        assert buckets["n_pairs"].sum() == len(scan)
        assert (buckets["n_pairs"] > 0).all()

    def test_buckets_are_ordered_by_pvalue(self):
        scan = self._scan()
        returns = _returns(n_cols=len(scan), n_days=300)
        returns.columns = [f"{r.a}/{r.b}" for r in scan.itertuples()]
        buckets = does_the_screen_predict(scan, returns)
        assert buckets["pvalue_to"].is_monotonic_increasing
        assert (buckets["pvalue_from"] <= buckets["pvalue_to"]).all()


class TestLegConcentration:
    def test_counts_both_sides_of_every_pair(self):
        scan = pd.DataFrame({"a": ["AAA", "AAA", "BBB"], "b": ["BBB", "CCC", "CCC"]})
        conc = leg_concentration(scan).set_index("ticker")
        assert conc.loc["AAA", "n_legs"] == 2
        assert conc.loc["BBB", "n_legs"] == 2
        assert conc.loc["CCC", "n_legs"] == 2
        assert conc["share_of_legs"].sum() == pytest.approx(1.0)

    def test_sorted_most_concentrated_first(self):
        scan = pd.DataFrame({"a": ["AAA"] * 5 + ["BBB"], "b": ["CCC"] * 6})
        conc = leg_concentration(scan)
        assert conc.iloc[0]["ticker"] == "CCC"
        assert conc["n_legs"].is_monotonic_decreasing
