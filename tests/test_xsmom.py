"""Tests for the momentum sleeve.

Built on tiny constructed price panels where the right answer can be worked
out by hand, not on the real universe - the real one has 143 months in it
and asserting a Sharpe would turn a data refresh into a test failure. The
one thing checked against the real file is that it loads and that the
sleeve is dollar neutral on every month it traded.
"""

import numpy as np
import pandas as pd
import pytest

from sleeves.xsmom.momentum import (backtest, load_universe, market_beta,
                                    momentum_signal, monthly_prices,
                                    read_results, summary, target_weights,
                                    write_results)


def prices(n_months: int = 30, names=("A", "B", "C", "D", "E"),
           drift=None, seed: int = 0) -> pd.DataFrame:
    """Daily prices with a chosen per-name drift, so the ranking is known."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2020-01-01", periods=n_months * 21)
    drift = np.zeros(len(names)) if drift is None else np.asarray(drift)
    steps = 0.0002 * drift + 0.0001 * rng.standard_normal((len(days), len(names)))
    return pd.DataFrame(100 * np.exp(np.cumsum(steps, axis=0)),
                        index=days, columns=list(names))


# ---------- the signal ----------

def test_signal_is_twelve_to_one_month_return():
    m = pd.DataFrame({"A": np.arange(1.0, 25.0)},
                     index=pd.date_range("2020-01-31", periods=24, freq="ME"))
    sig = momentum_signal(m)
    # row 13: price[12] / price[1] - 1 = 13/2 - 1
    assert sig["A"].iloc[13] == pytest.approx(13 / 2 - 1)
    assert sig["A"].iloc[:12].isna().all()


def test_signal_skips_the_most_recent_month():
    """A move in the last month must not show up in the signal."""
    idx = pd.date_range("2020-01-31", periods=14, freq="ME")
    flat = pd.DataFrame({"A": np.full(14, 100.0)}, index=idx)
    jumped = flat.copy()
    jumped.loc[idx[-1], "A"] = 200.0
    assert momentum_signal(flat)["A"].iloc[-1] == pytest.approx(0.0)
    assert momentum_signal(jumped)["A"].iloc[-1] == pytest.approx(0.0)


def test_signal_does_not_see_the_future():
    idx = pd.date_range("2020-01-31", periods=20, freq="ME")
    base = pd.DataFrame({"A": np.linspace(100, 120, 20)}, index=idx)
    changed = base.copy()
    changed.iloc[15:, 0] = 1000.0
    pd.testing.assert_series_equal(momentum_signal(base)["A"].iloc[:15],
                                   momentum_signal(changed)["A"].iloc[:15])


# ---------- the weights ----------

def test_weights_are_dollar_neutral_and_equal_within_legs():
    sig = pd.Series({f"s{i}": float(i) for i in range(10)})
    w = target_weights(sig, quantile=0.2, min_names=5)
    assert w.sum() == pytest.approx(0.0)
    assert w[w > 0].sum() == pytest.approx(1.0)
    assert w[w < 0].sum() == pytest.approx(-1.0)
    assert set(w[w > 0].index) == {"s8", "s9"}
    assert set(w[w < 0].index) == {"s0", "s1"}


def test_unranked_names_get_no_weight():
    sig = pd.Series({"a": 1.0, "b": 2.0, "c": np.nan, "d": 3.0, "e": 4.0,
                     "f": 5.0})
    w = target_weights(sig, quantile=0.2, min_names=3)
    assert w["c"] == 0.0
    assert (w != 0).sum() == 2


def test_too_few_names_means_no_trade():
    sig = pd.Series({"a": 1.0, "b": 2.0, "c": 3.0})
    w = target_weights(sig, quantile=0.2, min_names=20)
    assert (w == 0).all()


def test_sector_neutral_ranks_inside_each_sector():
    """Every name in sector X beats every name in sector Y; an unconditional
    sort is long all of X and short all of Y, a sector-neutral one is not."""
    sig = pd.Series({"x1": 10.0, "x2": 11.0, "x3": 12.0, "x4": 13.0, "x5": 14.0,
                     "y1": 0.0, "y2": 1.0, "y3": 2.0, "y4": 3.0, "y5": 4.0})
    sectors = {n: n[0] for n in sig.index}
    plain = target_weights(sig, quantile=0.2, min_names=5)
    neutral = target_weights(sig, quantile=0.2, min_names=5, sectors=sectors)
    assert (plain[plain > 0].index.str.startswith("x")).all()
    assert (plain[plain < 0].index.str.startswith("y")).all()
    for s in "xy":
        in_sector = neutral[neutral.index.str.startswith(s)]
        assert in_sector[in_sector > 0].sum() == pytest.approx(0.5)
        assert in_sector[in_sector < 0].sum() == pytest.approx(-0.5)


# ---------- the backtest ----------

def test_backtest_is_long_the_drifting_winners():
    """Five names, one drifts up and one down, the sleeve should earn it."""
    p = prices(n_months=30, drift=[+1, 0, 0, 0, -1])
    frame = backtest(p, cost_bps=0.0, min_names=3)
    assert len(frame) > 0
    assert frame["gross"].mean() > 0
    assert (frame["n_long"] == 1).all()
    assert (frame["n_short"] == 1).all()


def test_untraded_signal_months_are_dropped():
    p = prices(n_months=30)
    frame = backtest(p, cost_bps=0.0, min_names=3)
    # 30 months of prices, 12 needed for a signal, trade starts the month after
    assert frame.index[0] > monthly_prices(p).index[12]
    assert (frame["n_long"] > 0).all()


def test_costs_only_ever_reduce_the_return():
    p = prices(n_months=30, drift=[+1, 0.5, 0, -0.5, -1])
    free = backtest(p, cost_bps=0.0, min_names=3)
    paid = backtest(p, cost_bps=25.0, min_names=3)
    pd.testing.assert_series_equal(free["gross"], paid["gross"])
    assert (paid["net"] <= free["net"]).all()
    assert (paid["net"] < free["net"]).any()
    assert (paid["turnover"] >= 0).all()


def test_turnover_is_measured_from_the_drifted_book():
    """Same target two months running still costs something, because the
    positions moved with their prices in between."""
    p = prices(n_months=30, drift=[+3, 0, 0, 0, -3])
    frame = backtest(p, cost_bps=10.0, min_names=3)
    steady = frame[(frame["n_long"] == 1)]
    assert (steady["turnover"] > 0).all()
    # but a lot less than a full rebuild of the book, which is 1.0 one-way
    assert steady["turnover"].median() < 0.5


def test_summary_annualizes_monthly():
    idx = pd.date_range("2020-01-31", periods=24, freq="ME")
    frame = pd.DataFrame({"net": np.full(24, 0.01), "gross": 0.01,
                          "turnover": 0.3, "n_long": 5, "n_short": 5},
                         index=idx)
    s = summary(frame)
    assert s["ann_return"] == pytest.approx(0.12)
    assert s["ann_vol"] == pytest.approx(0.0)
    assert s["monthly_turnover"] == pytest.approx(0.3)


def test_market_beta_of_the_market_is_one():
    p = prices(n_months=30, drift=[1, 1, 1, 1, 1], seed=3)
    monthly = monthly_prices(p).pct_change(fill_method=None)
    frame = pd.DataFrame({"net": monthly.mean(axis=1)}).dropna()
    assert market_beta(frame, p) == pytest.approx(1.0)


def test_results_round_trip(tmp_path):
    p = prices(n_months=30, drift=[+1, 0, 0, 0, -1])
    frame = backtest(p, min_names=3)
    path = tmp_path / "r.json"
    write_results(frame, str(path))
    back = read_results(str(path))
    pd.testing.assert_frame_equal(back, frame, check_freq=False,
                                  check_dtype=False)


# ---------- the real file ----------

def test_real_universe_loads_and_the_sleeve_is_neutral():
    try:
        p = load_universe()
    except FileNotFoundError:
        pytest.skip("universe files not on disk")
    assert p.shape[1] > 90
    assert p.index.is_monotonic_increasing
    assert not p.index.duplicated().any()
    frame = backtest(p.iloc[:, :30].loc[:"2016"])
    assert (frame["n_long"] == frame["n_short"]).all()
    assert (frame["turnover"] <= 1.0 + 1e-9).all()
