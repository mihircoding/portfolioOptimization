"""Checks on the reject-group comparison in control.py.

The claim the module makes is a negative one - the screen doesn't separate
anything - and negative claims are the easy ones to get by accident. So the
tests here mostly check that the machinery *would* find a difference if one
existed: a permutation test that never rejects proves nothing.
"""
import numpy as np
import pandas as pd
import pytest

import control


@pytest.fixture
def two_pairs():
    """A four-ticker panel and the two pairs traded off it."""
    idx = pd.bdate_range("2021-01-01", periods=400)
    rng = np.random.default_rng(0)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(idx), 4)), axis=0)),
        index=idx, columns=["A", "B", "C", "D"])
    scan = pd.DataFrame({"a": ["A", "C"], "b": ["B", "D"],
                         "beta": [1.0, 1.0], "pvalue": [0.01, 0.50]})
    return scan, prices


def test_every_pair_comes_back_with_a_curve(two_pairs):
    scan, prices = two_pairs
    stats = control.backtest_group(scan, prices)
    returns = stats.attrs["returns"]

    assert len(stats) == 2
    assert list(stats["passed"]) == [True, False]          # split on p = 0.05
    assert list(returns.columns) == ["A/B", "C/D"]
    assert len(returns) == len(prices)
    assert set(stats.columns) >= {"sharpe", "total_return", "n_round_trips"}


def test_a_book_of_one_pair_is_that_pair(two_pairs):
    scan, prices = two_pairs
    stats = control.backtest_group(scan, prices)
    returns = stats.attrs["returns"]

    single = control.book_stats(returns[["A/B"]])
    assert single["n"] == 1
    assert single["sharpe"] == pytest.approx(stats.loc[0, "sharpe"])
    assert single["total_return"] == pytest.approx(stats.loc[0, "total_return"])

    assert control.book_stats(pd.DataFrame())["n"] == 0


def test_the_book_is_not_the_average_of_its_pairs(two_pairs):
    """Averaging returns daily and compounding is not compounding then
    averaging - and the difference is the whole reason a book is worth
    reporting separately from its constituents."""
    scan, prices = two_pairs
    returns = control.backtest_group(scan, prices).attrs["returns"]
    book = control.book_stats(returns)
    each = [control.book_stats(returns[[c]])["total_return"] for c in returns.columns]
    assert book["total_return"] != pytest.approx(float(np.mean(each)))


def test_permutation_test_finds_a_real_difference():
    rng = np.random.default_rng(1)
    good = rng.normal(0.8, 0.3, 300)
    bad = rng.normal(0.0, 0.3, 300)
    res = control.permutation_test(good, bad, n=2000, seed=2)
    assert res["difference"] == pytest.approx(0.8, abs=0.1)
    assert res["p_value"] < 0.01


def test_permutation_test_is_quiet_when_there_is_none():
    rng = np.random.default_rng(3)
    a = rng.normal(0.0, 1.0, 400)
    b = rng.normal(0.0, 1.0, 400)
    res = control.permutation_test(a, b, n=2000, seed=4)
    assert res["p_value"] > 0.05
    # a p-value is never zero out of a finite number of relabellings
    assert res["p_value"] >= 1 / 2001


def test_permutation_test_does_not_consume_its_inputs():
    a = np.array([1.0, 2.0, 3.0, 4.0])
    b = np.array([5.0, 6.0, 7.0, 8.0])
    control.permutation_test(a, b, n=50, seed=5)
    assert list(a) == [1.0, 2.0, 3.0, 4.0]
    assert list(b) == [5.0, 6.0, 7.0, 8.0]


def test_deciles_cover_everything_in_order():
    rng = np.random.default_rng(6)
    n = 1000
    stats = pd.DataFrame({"pvalue": rng.random(n), "sharpe": rng.normal(0, 1, n),
                          "total_return": rng.normal(0, 0.1, n)})
    table = control.decile_table(stats)
    assert len(table) == 10
    assert table["pairs"].sum() == n
    assert table["pvalue_lo"].is_monotonic_increasing
    assert (table["share_profitable"].between(0, 1)).all()
