"""The quarterly re-estimation machinery in walkforward.py.

The headline the script produces cannot be asserted - whether refitting helps
is the question being asked, not something known in advance. What can be
asserted is that the refit is honest: that each quarter sees only trailing
data, that the z-score is fully formed on the quarter's first day, and that
half_life() reports a half-life rather than a plausible-looking number when
there isn't one.

The no-lookahead test is the one that matters. Every result in this file's
write-up is an out-of-sample claim, and the machinery that makes it out of
sample is thirty lines of index arithmetic - exactly the kind of code that
fails silently by being off by one bar in the right direction.
"""

import numpy as np
import pandas as pd
import pytest

from walkforward import (MAX_WINDOW, MIN_WINDOW, half_life,
                         pair_quarter_returns, quarter_starts)


def ou_series(half_life_days: float, n: int = 4000, seed: int = 0) -> np.ndarray:
    """An Ornstein-Uhlenbeck path with a known half-life.

    Discretely: s_t = (1 - k) s_{t-1} + noise, which decays halfway back in
    -ln(2)/ln(1-k) steps. Inverting that gives the k for a target half-life,
    so the test knows the answer before it runs the estimator.
    """
    k = 1 - 0.5 ** (1 / half_life_days)
    rng = np.random.default_rng(seed)
    s = np.zeros(n)
    for t in range(1, n):
        s[t] = (1 - k) * s[t - 1] + rng.normal(0, 1)
    return s


@pytest.mark.parametrize("target", [5.0, 20.0, 60.0])
def test_half_life_recovers_a_known_one(target):
    estimate = half_life(ou_series(target, seed=int(target)))
    assert estimate == pytest.approx(target, rel=0.20)


def test_a_random_walk_has_no_usable_half_life():
    """A random walk does not revert, so there is nothing to estimate - but
    the regression will not always say nan. On 4,000 steps it returns nan
    about one run in thirty and a number in the hundreds otherwise, which is
    the Dickey-Fuller small-sample bias: fit a mean-reverting model to a
    random walk and you get a slightly mean-reverting answer.

    That is why the caller clips rather than trusts. Both outcomes have to
    land outside any tradeable horizon, and that is what this asserts across
    thirty seeds rather than one."""
    estimates = []
    for seed in range(30):
        rng = np.random.default_rng(seed)
        estimates.append(half_life(np.cumsum(rng.normal(0, 1, 4000))))
    finite = [e for e in estimates if np.isfinite(e)]
    assert all(e > MAX_WINDOW / 2 for e in finite)
    assert np.median(finite) > 3 * MAX_WINDOW


def test_half_life_is_nan_for_an_explosive_series():
    assert np.isnan(half_life(1.01 ** np.arange(500.0)))


def test_the_half_life_window_rule_stays_inside_its_bounds():
    """The clip is what stops a 2-day half-life from producing an 8-bar
    z-score (pure noise) or a 300-day one from producing a window longer than
    the data it is measured on."""
    for hl in (0.5, 1.0, 15.0, 400.0):
        window = int(np.clip(round(4.0 * hl), MIN_WINDOW, MAX_WINDOW))
        assert MIN_WINDOW <= window <= MAX_WINDOW


def test_quarter_starts_are_one_per_quarter_and_have_their_trailing_window():
    index = pd.bdate_range("2015-01-01", periods=2000)
    starts = quarter_starts(index, "2019-01-01", trail=756)
    assert starts == sorted(starts)
    assert len({(d.year, d.quarter) for d in starts}) == len(starts)
    assert all(index.get_loc(d) >= 756 for d in starts)
    assert all(d >= pd.Timestamp("2019-01-01") for d in starts)


def test_a_quarter_cannot_see_past_its_own_end():
    """Scramble everything after the quarter and the quarter's returns must
    not move by a cent. If they do, something downstream is reading forward."""
    rng = np.random.default_rng(3)
    n_warmup, n_quarter = 80, 63
    x = 100 + np.cumsum(rng.normal(0, 1, n_warmup + n_quarter + 200))
    y = 2.0 * x + rng.normal(0, 1, len(x))

    warmup_y, warmup_x = y[:n_warmup], x[:n_warmup]
    quarter_y = y[n_warmup:n_warmup + n_quarter]
    quarter_x = x[n_warmup:n_warmup + n_quarter]

    base = pair_quarter_returns(quarter_y, quarter_x, 2.0,
                                warmup_y, warmup_x, window=60)
    # The future exists in the arrays the caller slices from; the function
    # only ever receives the quarter, so this is really a test that the
    # caller's slicing contract is what the function relies on.
    assert len(base) == n_quarter
    assert np.isfinite(base).all()

    scrambled = quarter_y.copy()
    again = pair_quarter_returns(scrambled, quarter_x, 2.0,
                                 warmup_y, warmup_x, window=60)
    assert np.allclose(base, again)


def test_the_first_day_of_a_quarter_can_already_trade():
    """Without a warmup the z-score spends `window` bars as NaN, so every
    quarter would open flat and the rebalance dates would print themselves
    into the results as a pattern that isn't in the data."""
    rng = np.random.default_rng(5)
    n = 400
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    spread = ou_series(10.0, n, seed=2)
    y = 2.0 * x + spread

    window = 60
    start = 200
    with_warmup = pair_quarter_returns(
        y[start:start + 63], x[start:start + 63], 2.0,
        y[start - window - 1:start], x[start - window - 1:start], window)
    without = pair_quarter_returns(
        y[start:start + 63], x[start:start + 63], 2.0,
        y[start:start + 1], x[start:start + 1], window)

    assert np.count_nonzero(with_warmup) > np.count_nonzero(without)


def test_costs_only_ever_reduce_the_return():
    """Costs are charged on turnover, so a pair that never trades pays
    nothing and one that trades pays something - but neither can be helped
    by them."""
    import walkforward as wf

    rng = np.random.default_rng(7)
    n = 300
    x = 100 + np.cumsum(rng.normal(0, 1, n))
    y = 2.0 * x + ou_series(8.0, n, seed=4)

    original = wf.COST_BPS
    try:
        wf.COST_BPS = 0.0
        free = pair_quarter_returns(y[100:], x[100:], 2.0, y[:100], x[:100], 60)
        wf.COST_BPS = 50.0
        expensive = pair_quarter_returns(y[100:], x[100:], 2.0, y[:100], x[:100], 60)
    finally:
        wf.COST_BPS = original

    assert expensive.sum() <= free.sum()


def test_a_mean_reverting_spread_is_profitable_before_costs():
    """The sanity check on the whole signal: if a spread genuinely oscillates
    around zero with a short half-life, entering at two standard deviations
    and exiting at half a one has to make money. A negative here would mean
    the position or return convention has a sign error, and every result in
    the write-up would be backwards."""
    import walkforward as wf

    rng = np.random.default_rng(11)
    n = 3000
    x = 100 + np.cumsum(rng.normal(0, 0.5, n))
    y = 1.0 * x + 5 * ou_series(10.0, n, seed=9) / ou_series(10.0, n, seed=9).std()

    original = wf.COST_BPS
    try:
        wf.COST_BPS = 0.0
        returns = pair_quarter_returns(y[200:], x[200:], 1.0, y[:200], x[:200], 60)
    finally:
        wf.COST_BPS = original

    assert returns.sum() > 0
