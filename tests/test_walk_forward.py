"""The walk-forward has to survive one optimizer failing.

Min CVaR and max-Sharpe solve numerically and can fail on an awkward window.
The question these tests pin down is what that failure costs: the one method
for the one year, and nothing else. Getting this wrong is quiet - the table
still prints, it is just measured over a different history than it claims.
"""
import numpy as np
import pandas as pd
import pytest

import run_optimization as ro


@pytest.fixture
def prices():
    """Six calendar years of business-day prices for five assets."""
    idx = pd.bdate_range("2015-01-01", "2020-12-31")
    rng = np.random.default_rng(0)
    steps = rng.normal(0.0003, 0.01, size=(len(idx), 5))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)), index=idx,
                        columns=["A", "B", "C", "D", "E"])


def test_failure_list_keeps_the_other_methods(three_asset):
    mu, cov = three_asset
    failures = []
    portfolios = ro.build_portfolios(mu, cov, failures=failures)
    assert failures == []
    assert "Max Sharpe" in portfolios

    def boom(*args, **kwargs):
        raise RuntimeError("did not converge")

    original, ro.max_sharpe_weights = ro.max_sharpe_weights, boom
    try:
        failures = []
        portfolios = ro.build_portfolios(mu, cov, failures=failures)
        assert [name for name, _ in failures] == ["Max Sharpe"]
        assert "did not converge" in failures[0][1]
        assert "Min variance" in portfolios and "Max Sharpe" not in portfolios

        with pytest.raises(RuntimeError):
            ro.build_portfolios(mu, cov)          # no list -> still raises
    finally:
        ro.max_sharpe_weights = original


def test_one_bad_solve_costs_one_cell_not_the_year(prices):
    calls = {"n": 0}
    original = ro.min_variance_weights

    def fails_on_the_second_window(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("singular covariance")
        return original(*args, **kwargs)

    ro.min_variance_weights = fails_on_the_second_window
    try:
        failures = []
        seen = {year: set(p) for year, _, p in
                ro.rebalances(prices, lookback=2, failures=failures)}
    finally:
        ro.min_variance_weights = original

    assert len(failures) == 1
    bad_year, name, _ = failures[0]
    assert name == "Min variance"
    # the year survives for everyone else
    assert "Min variance" not in seen[bad_year]
    assert "Max Sharpe" in seen[bad_year]
    assert len(seen) == 4                          # 2017-2020, none lost


def test_a_year_nothing_can_solve_is_skipped(prices):
    original = ro.build_portfolios
    calls = {"n": 0}

    def nothing_solves_the_first_window(*args, **kwargs):
        calls["n"] += 1
        return {} if calls["n"] == 1 else original(*args, **kwargs)

    ro.build_portfolios = nothing_solves_the_first_window
    try:
        years = [year for year, _, _ in ro.rebalances(prices, lookback=2)]
    finally:
        ro.build_portfolios = original

    assert years == [2018, 2019, 2020]


def test_summary_rows_cover_the_same_years(prices):
    calls = {"n": 0}
    original = ro.min_variance_weights

    def fails_once(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 2:
            raise RuntimeError("singular covariance")
        return original(*args, **kwargs)

    ro.min_variance_weights = fails_once
    try:
        panel, summary = ro.walk_forward(prices, lookback=2, verbose=False)
    finally:
        ro.min_variance_weights = original

    # the panel is ragged - min variance really is missing a year ...
    assert panel.groupby("portfolio")["year"].nunique().min() == 3
    # ... but every row of the printed table is scored on the same three years
    assert set(summary["years"]) == {3}


def test_no_failures_means_every_year_is_scored(prices):
    panel, summary = ro.walk_forward(prices, lookback=2, verbose=False)
    assert set(summary["years"]) == {4}
    assert panel["year"].nunique() == 4
