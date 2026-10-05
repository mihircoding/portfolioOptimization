"""Kalman-filter hedge ratio: a time-varying beta_t instead of one OLS beta
held fixed for the whole sample. See src/pairs.ipynb (kalman_hedge_ratio,
compute_dynamic_spread) for the implementation - this project's version of
the README's Kalman-filter stretch goal, built from the update equations
rather than pulled from pykalman.
"""

import numpy as np
import pandas as pd
import pytest

from src.pairs import compute_dynamic_spread, compute_spread, hedge_ratio, kalman_hedge_ratio


class TestKalmanHedgeRatio:
    def test_converges_to_a_constant_true_beta(self, rng):
        n = 1000
        idx = pd.bdate_range("2018-01-01", periods=n)
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="x")
        y = pd.Series(1.7 * x.to_numpy() + rng.normal(0, 0.5, n), index=idx, name="y")

        beta_t = kalman_hedge_ratio(y, x)

        # Ignore the first 200 bars while P is still large and beta is
        # swinging around; after that it should have settled near the truth.
        tail_beta = beta_t.iloc[200:]
        assert tail_beta.mean() == pytest.approx(1.7, abs=0.05)
        assert tail_beta.std() < 0.05  # settled, not still wandering

    def test_tracks_a_drifting_beta_better_than_a_single_ols_fit(self, rng):
        n = 1000
        idx = pd.bdate_range("2018-01-01", periods=n)
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="x")
        true_beta = 1.0 + 1.5 * np.arange(n) / n  # ramps 1.0 -> 2.5 over the sample
        y = pd.Series(true_beta * x.to_numpy() + rng.normal(0, 0.3, n), index=idx, name="y")

        beta_t = kalman_hedge_ratio(y, x, delta=1e-3)
        kalman_sq_error = (beta_t.to_numpy() - true_beta) ** 2

        static_beta = hedge_ratio(y, x)
        static_sq_error = (static_beta - true_beta) ** 2

        # A single OLS beta is, by construction, one number for a target
        # that moved the whole time - the Kalman filter should track it
        # closely enough to have much lower squared error on average.
        assert kalman_sq_error.mean() < static_sq_error.mean() / 3

    def test_returns_one_value_per_input_date(self, cointegrated_pair):
        y, x = cointegrated_pair
        beta_t = kalman_hedge_ratio(y, x)
        assert list(beta_t.index) == list(y.index)
        assert beta_t.name == "beta"

    def test_smaller_delta_moves_less(self, rng):
        """delta controls how fast beta is allowed to drift - a much smaller
        delta should produce a visibly steadier (lower-variance) beta path
        on the same data."""
        n = 500
        idx = pd.bdate_range("2018-01-01", periods=n)
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="x")
        y = pd.Series(2.0 * x.to_numpy() + rng.normal(0, 1.0, n), index=idx, name="y")

        stiff = kalman_hedge_ratio(y, x, delta=1e-7)
        loose = kalman_hedge_ratio(y, x, delta=1e-2)

        assert stiff.iloc[100:].diff().abs().mean() < loose.iloc[100:].diff().abs().mean()


class TestDynamicSpread:
    def test_tracks_close_to_the_static_spread_once_settled(self, rng):
        """Fed a genuinely constant-beta, no-intercept pair (matching
        compute_spread's own y - beta*x convention), the filter's spread
        should end up close to the static OLS spread once it has converged
        - it isn't a different model, just a slower-to-commit version of the
        same one. (cointegrated_pair isn't used here on purpose: it bakes in
        a real intercept of 10, which a through-origin filter has to absorb
        into beta rather than a separate term - a real and correct
        difference from the static fit, not a bug, but it would make this
        specific "do the two spreads agree" check compare two honestly
        different quantities.)
        """
        n = 1500
        idx = pd.bdate_range("2018-01-01", periods=n)
        x = pd.Series(100 + np.cumsum(rng.normal(0, 1, n)), index=idx, name="x")
        noise = np.zeros(n)
        for t in range(1, n):
            noise[t] = 0.85 * noise[t - 1] + rng.normal(0, 1)
        y = pd.Series(2.5 * x.to_numpy() + noise, index=idx, name="y")

        beta = hedge_ratio(y, x)
        static = compute_spread(y, x, beta)

        beta_t = kalman_hedge_ratio(y, x)
        dynamic = compute_dynamic_spread(y, x, beta_t)

        diff = (static.iloc[-200:] - dynamic.iloc[-200:]).abs()
        # x is on a ~150-260 scale here and delta=1e-4 keeps the filter
        # slightly adaptive forever by design (that's what lets it track a
        # REAL drift elsewhere in this file) - so "close" means a couple of
        # percent of the price level, not an exact match to a static fit.
        assert diff.mean() < 5.0

    def test_aligns_on_the_intersection_of_available_dates(self, rng):
        idx = pd.bdate_range("2018-01-01", periods=50)
        x = pd.Series(np.arange(1, 51, dtype=float), index=idx, name="x")
        y = pd.Series(np.arange(1, 51, dtype=float) * 2, index=idx, name="y")
        beta_t = kalman_hedge_ratio(y, x).iloc[5:]  # pretend the filter starts later

        spread = compute_dynamic_spread(y, x, beta_t)
        assert len(spread) == len(beta_t)
