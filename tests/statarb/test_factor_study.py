"""Checks on factor_study.py, which hedges every spread against the first
principal component of the cross-section.

The finding is that the hedge does not change the thing it was built to
change, so what is checked is the machinery: that the component is fitted
on nothing it is applied to, that a pair whose P&L is all factor hedges to
nothing, that the quiet flag is known the day before, and that a book's
name weights reproduce its gross return.
"""
import numpy as np
import pandas as pd
import pytest

from sleeves.statarb import factor_study as fs
from sleeves.statarb.scan import backtest, generate_positions, rolling_zscore


@pytest.fixture
def one_factor_tape():
    """Twenty names that are one factor plus noise, with known loadings."""
    rng = np.random.default_rng(7)
    idx = pd.bdate_range("2019-01-01", periods=700)
    loadings = np.linspace(0.5, 1.5, 20)
    f = rng.normal(0, 0.02, len(idx))
    noise = rng.normal(0, 0.004, (len(idx), 20))
    returns = pd.DataFrame(np.outer(f, loadings) + noise, index=idx,
                           columns=[f"N{i:02d}" for i in range(20)])
    return returns, loadings, pd.Series(f, index=idx)


def test_first_component_recovers_the_loadings_signed_positive(one_factor_tape):
    returns, loadings, _ = one_factor_tape
    w = fs.first_component(returns.to_numpy())
    assert abs(w.sum() - 1.0) < 1e-12 and (w > 0).all()
    # up to scale, the component is the loading vector
    assert np.corrcoef(w, loadings)[0, 1] > 0.99


def test_rolling_factor_is_fitted_on_nothing_it_is_applied_to(one_factor_tape):
    returns, loadings, f = one_factor_tape
    start = returns.index[300]
    fit = fs.rolling_factor(returns, window=250, refit_every=21, start=start)
    assert fit["factor"].loc[:start].iloc[:-1].isna().all()
    assert fit["factor"].loc[start:].notna().all()
    # the betas on the first fitted day are the true loadings, up to scale
    b = fit["betas"].loc[start].to_numpy()
    assert np.corrcoef(b, loadings)[0, 1] > 0.99
    # shock the day the factor is first applied to: loadings must not move
    shocked = returns.copy()
    shocked.loc[start] = 0.5
    refit = fs.rolling_factor(shocked, window=250, refit_every=21, start=start)
    pd.testing.assert_series_equal(refit["betas"].loc[start], fit["betas"].loc[start])
    assert refit["factor"].loc[start] != fit["factor"].loc[start]
    # the factor is the universe's return on the lagged loadings
    assert fit["explained"].loc[start] > 0.9


def test_hedged_backtest_with_no_loadings_is_the_shipped_backtest():
    rng = np.random.default_rng(1)
    idx = pd.bdate_range("2021-01-01", periods=300)
    y = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    x = pd.Series(100 * np.exp(np.cumsum(rng.normal(0, 0.01, len(idx)))), index=idx)
    pos = generate_positions(rolling_zscore(y - x))
    zero = pd.Series(0.0, index=idx)
    factor = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    plain = backtest(y, x, pos)
    hedged = fs.hedged_backtest(y, x, pos, factor, zero, zero)
    pd.testing.assert_frame_equal(plain, hedged)


def test_a_spread_that_is_all_factor_hedges_to_nothing_gross_and_pays_for_the_hedge():
    idx = pd.bdate_range("2021-01-01", periods=120)
    f = pd.Series(np.r_[np.zeros(1), np.tile([0.01, -0.01], 60)[:119]], index=idx)
    # y moves twice as much as x on the factor and nothing else
    y = 100 * (1 + 2 * f).cumprod()
    x = 100 * (1 + f).cumprod()
    pos = pd.Series(1, index=idx)
    beta_y, beta_x = pd.Series(2.0, index=idx), pd.Series(1.0, index=idx)
    out = fs.hedged_backtest(y, x, pos, f, beta_y, beta_x, cost_bps=5.0)
    assert out["ret"].iloc[2:].abs().max() < 1e-12
    # one trade on day one for the legs, and the hedge notional is put on once
    # and never changes, so the costs are the legs' plus one hedge ticket
    costs = (out["ret"] - out["ret_net"])
    assert costs.iloc[0] == pytest.approx(5.0 / 10_000)          # the legs
    assert costs.iloc[1] == pytest.approx(0.5 * 5.0 / 10_000)    # the hedge, notional 1/2
    assert costs.iloc[2:].abs().max() < 1e-15


def test_quiet_regime_is_known_the_day_before():
    idx = pd.bdate_range("2021-01-01", periods=400)
    rng = np.random.default_rng(2)
    f = pd.Series(rng.normal(0, 0.01, len(idx)), index=idx)
    f.iloc[300:] *= 5                                    # the factor gets loud
    quiet = fs.quiet_regime(f, window=20, median_window=250)
    assert quiet.dtype == bool and not quiet.iloc[0]
    vol = f.rolling(20).std()
    ref = vol.rolling(250, min_periods=20).median()
    expected = [False] + (vol < ref).tolist()[:-1]
    assert quiet.tolist() == expected
    # loud from day 300: once the 20-day vol has caught up, the flag is off
    assert not quiet.iloc[325:].any()


def test_name_weights_reproduce_the_book_and_mirror_keeps_the_costs():
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2021-01-01", periods=300)
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rng.normal(0, 0.01, (len(idx), 3)), axis=0)),
                          index=idx, columns=["A", "B", "C"])
    scan = pd.DataFrame({"a": ["A", "A", "B"], "b": ["B", "C", "C"], "beta": [1.0, 1.0, 1.0],
                         "pvalue": [0.01, 0.5, 0.5], "same_sector": [True, False, False]})
    mask = np.array([True, True, True])
    weights = fs.name_weights(scan, prices, mask)
    from_weights = (weights * prices.pct_change().fillna(0.0)).sum(axis=1)
    gross = pd.concat([backtest(prices[r.a], prices[r.b],
                                generate_positions(rolling_zscore(prices[r.a] - prices[r.b])))["ret"]
                       for r in scan.itertuples()], axis=1).mean(axis=1)
    assert (from_weights - gross).abs().max() < 1e-12
    net = gross - 0.001
    mg, mn = fs.mirror_panels(gross.to_frame(), net.to_frame())
    pd.testing.assert_frame_equal(mg, -gross.to_frame())
    assert ((mg - mn) - (gross - net).to_frame()).abs().max().max() < 1e-15
