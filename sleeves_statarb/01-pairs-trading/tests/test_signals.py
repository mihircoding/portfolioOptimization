import numpy as np
import pandas as pd
import pytest

from src.signals import compute_spread, generate_positions, half_life, rolling_zscore


class TestSpread:
    def test_definition(self):
        y = pd.Series([10.0, 12.0, 14.0])
        x = pd.Series([2.0, 3.0, 4.0])
        expected = y - 1.5 * x
        pd.testing.assert_series_equal(compute_spread(y, x, 1.5), expected)


class TestRollingZscore:
    def test_no_lookahead(self, cointegrated_pair):
        # truncating the series must not change earlier z-scores
        y, x = cointegrated_pair
        spread = y - 2.5 * x
        z_full = rolling_zscore(spread, window=60)
        z_trunc = rolling_zscore(spread.iloc[:800], window=60)
        pd.testing.assert_series_equal(z_full.iloc[:800], z_trunc)

    def test_warmup_is_nan(self, cointegrated_pair):
        y, x = cointegrated_pair
        z = rolling_zscore(y - 2.5 * x, window=60)
        assert z.iloc[:59].isna().all()
        assert not np.isnan(z.iloc[59])

    def test_standardization(self, rng):
        # on stationary noise the z-score should be roughly N(0,1)
        s = pd.Series(rng.normal(5.0, 2.0, 3000))
        z = rolling_zscore(s, window=100).dropna()
        assert abs(z.mean()) < 0.1
        assert 0.8 < z.std() < 1.2


class TestPositions:
    def make_z(self, values):
        return pd.Series(values, dtype=float)

    def test_entry_and_exit_thresholds(self):
        #        flat   entry short      hold        exit    flat
        z = self.make_z([0.0, 2.5, 1.5, 1.0, 0.4, 0.0])
        pos = generate_positions(z, entry=2.0, exit=0.5)
        assert pos.tolist() == [0, -1, -1, -1, 0, 0]

    def test_long_side(self):
        z = self.make_z([0.0, -2.5, -1.0, -0.3, 0.0])
        pos = generate_positions(z, entry=2.0, exit=0.5)
        assert pos.tolist() == [0, 1, 1, 0, 0]

    def test_hysteresis_no_reentry_between_thresholds(self):
        # z between exit and entry while flat -> stay flat
        z = self.make_z([0.0, 1.5, 1.9, 1.0])
        pos = generate_positions(z, entry=2.0, exit=0.5)
        assert pos.tolist() == [0, 0, 0, 0]

    def test_nan_stays_flat(self):
        z = self.make_z([np.nan, np.nan, 2.5, 0.1])
        pos = generate_positions(z, entry=2.0, exit=0.5)
        assert pos.tolist() == [0, 0, -1, 0]


class TestHalfLife:
    def test_recovers_known_half_life_from_the_ar1_noise(self, cointegrated_pair):
        # spread = y - beta*x = 10 + AR(1) noise with phi=0.85, so the
        # theoretical half-life is ln(2) / ln(1/0.85) ~= 4.3 bars. Give the
        # OLS estimate on 1500 points reasonable room around that.
        y, x = cointegrated_pair
        spread = compute_spread(y, x, 2.5)
        hl = half_life(spread)
        assert 2 < hl < 8

    def test_non_mean_reverting_series_gives_no_useful_half_life(self, rng):
        # a random walk has no pull back to any level. The OLS lambda on a
        # single finite sample can still land a hair below zero by chance
        # (it did here), so this doesn't assert exactly inf - it asserts
        # the number is too large to trade on, which is the point: nothing
        # should report a short, tradeable-looking half-life on pure noise.
        idx = pd.bdate_range("2018-01-01", periods=1500)
        walk = pd.Series(100 + np.cumsum(rng.normal(0, 1, 1500)), index=idx)
        assert half_life(walk) > 100  # vs. ~4 bars for the true mean-reverter above
