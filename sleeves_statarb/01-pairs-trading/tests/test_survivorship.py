"""The takeover injection, and the three ways its measurement can lie.

Nothing here can check whether the deal rate is right - it is an external
estimate and the study sweeps it for exactly that reason. What can be checked
is that the machinery measures the thing it claims to, and that is worth more
attention than usual because this measurement has three separate ways of
producing a flattering number:

  the window        a takeover truncates its leg's history, and this book loses
                    money, so an affected pair trading fewer days looks better
                    for a reason that has nothing to do with the merger. Every
                    delta has to be scored over the SAME dates on both
                    histories.
  the dropouts      a pair left unusable earns zero, and zero beats this book's
                    average pair. Folding those into the mean makes mergers
                    look profitable.
  the untouched     a deal must not change a pair that does not hold the
                    acquired name, or the comparison is measuring the random
                    number generator.

The bucketing tests are the other half: the formation-performance split is what
the "does the screen pass co-winners more often" table rests on, and a bucket
boundary off by one would move the finding.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from survivorship import (formation_performance, inject_deals,
                          pass_rate_by_performance, pair_return, pair_returns)


@pytest.fixture
def prices():
    """Five names, four years of business days, all with a full history."""
    rng = np.random.default_rng(3)
    idx = pd.bdate_range("2021-01-04", periods=1_000)
    out = {}
    for i, name in enumerate(["AAA", "BBB", "CCC", "DDD", "EEE"]):
        steps = rng.normal(0.0003, 0.013, len(idx))
        out[name] = 100.0 * np.exp(np.cumsum(steps)) * (1 + 0.1 * i)
    return pd.DataFrame(out, index=idx)


@pytest.fixture
def pairs():
    return pd.DataFrame({"a": ["AAA", "AAA", "CCC"],
                         "b": ["BBB", "CCC", "DDD"],
                         "beta": [1.0, 1.0, 1.0]})


class TestInjectDeals:
    def test_no_deals_leaves_the_frame_identical(self, prices):
        rng = np.random.default_rng(0)
        out, events = inject_deals(prices, rng, rate=0.0, premium=0.30)
        assert events == []
        pd.testing.assert_frame_equal(out, prices)

    def test_an_acquired_name_gaps_up_by_the_premium(self, prices):
        rng = np.random.default_rng(1)
        out, events = inject_deals(prices, rng, rate=1.0, premium=0.30)
        assert events
        for e in events:
            i, name = e["day"], e["ticker"]
            before = float(prices[name].iloc[i - 1])
            after = float(out[name].iloc[i])
            assert after / before == pytest.approx(1.30, rel=0.02)

    def test_the_target_stops_moving_after_the_gap(self, prices):
        rng = np.random.default_rng(1)
        out, events = inject_deals(prices, rng, rate=1.0, premium=0.25)
        for e in events:
            i, name = e["day"], e["ticker"]
            live = out[name].iloc[i:].dropna()
            if len(live) < 20:
                continue
            deal_vol = float(live.pct_change().std())
            normal_vol = float(prices[name].pct_change().std())
            assert deal_vol < normal_vol / 3

    def test_it_delists_rather_than_trading_forever(self, prices):
        rng = np.random.default_rng(1)
        out, events = inject_deals(prices, rng, rate=1.0, premium=0.25)
        closed = [e for e in events if e["closed"]]
        assert closed, "no deal closed inside the window with this seed"
        for e in closed:
            tail = out[e["ticker"]].iloc[e["day"] + 110:]
            assert tail.isna().all()          # delisted, not trading forever
            assert out[e["ticker"]].iloc[:e["day"]].notna().all()

    def test_names_that_were_not_acquired_are_untouched(self, prices):
        # 5 names over 4 years at 2%/yr is well under one expected deal, so
        # there is something left over to compare against.
        rng = np.random.default_rng(2)
        out, events = inject_deals(prices, rng, rate=0.02, premium=0.30)
        touched = {e["ticker"] for e in events}
        untouched = [c for c in prices.columns if c not in touched]
        assert untouched
        pd.testing.assert_frame_equal(out[untouched], prices[untouched])

    def test_a_higher_rate_draws_more_deals_on_average(self, prices):
        counts = {}
        for rate in (0.01, 0.20):
            rng = np.random.default_rng(5)
            counts[rate] = np.mean([len(inject_deals(prices, rng, rate, 0.3)[1])
                                    for _ in range(40)])
        assert counts[0.01] < counts[0.20]

    def test_the_draw_is_poisson_around_rate_times_names_times_years(self, prices):
        rng = np.random.default_rng(9)
        years = len(prices) / 252
        expected = 0.08 * prices.shape[1] * years
        drawn = [len(inject_deals(prices, rng, 0.08, 0.3)[1]) for _ in range(400)]
        assert np.mean(drawn) == pytest.approx(expected, rel=0.25)


class TestMatchedWindow:
    """The one that stops the measurement lying."""

    def test_restricting_the_index_changes_the_answer(self, prices):
        full = pair_return(prices, "AAA", "BBB", 1.0)
        half = pair_return(prices, "AAA", "BBB", 1.0,
                           index=prices.index[:500])
        assert full is not None and half is not None
        assert full != half

    def test_a_pair_with_too_little_overlap_returns_none(self, prices):
        assert pair_return(prices, "AAA", "BBB", 1.0,
                           index=prices.index[:50]) is None

    def test_the_same_dates_on_the_same_prices_give_the_same_number(self, prices):
        a = pair_return(prices, "AAA", "BBB", 1.0, index=prices.index)
        b = pair_return(prices, "AAA", "BBB", 1.0)
        assert a == pytest.approx(b)

    def test_an_untouched_pair_is_unaffected_by_someone_elses_deal(self, prices,
                                                                  pairs):
        # Acquire one named stock directly rather than drawing, so the test is
        # about the effect on other pairs and not about which seed came up.
        rng = np.random.default_rng(4)
        modified = prices.copy()
        i = 500
        offer = float(modified["AAA"].iloc[i - 1]) * 1.30
        modified.iloc[i:, modified.columns.get_loc("AAA")] = offer
        touched = {"AAA"}
        clean = [r for r in pairs.itertuples()
                 if r.a not in touched and r.b not in touched]
        assert clean
        for row in clean:
            assert pair_return(modified, row.a, row.b, row.beta) == \
                   pytest.approx(pair_return(prices, row.a, row.b, row.beta))

    def test_the_gap_alone_changes_an_affected_pair(self, prices):
        """Same dates, same rules, only the acquired leg's prices differ."""
        modified = prices.copy()
        i = 600
        offer = float(modified["BBB"].iloc[i - 1]) * 1.30
        modified.iloc[i:, modified.columns.get_loc("BBB")] = offer

        idx = modified.index
        after = pair_return(modified, "AAA", "BBB", 1.0, index=idx)
        before = pair_return(prices, "AAA", "BBB", 1.0, index=idx)
        assert after is not None and before is not None
        assert after != pytest.approx(before)


class TestPairReturns:
    def test_it_keys_on_the_pair_and_skips_what_cannot_trade(self, prices,
                                                            pairs):
        got = pair_returns(pairs, prices)
        assert set(got) == {("AAA", "BBB"), ("AAA", "CCC"), ("CCC", "DDD")}

        short = prices.copy()
        short.iloc[100:, short.columns.get_loc("BBB")] = np.nan
        got = pair_returns(pairs, short)
        assert ("AAA", "BBB") not in got
        assert ("CCC", "DDD") in got


class TestPerformanceBuckets:
    def test_formation_return_is_first_to_last(self, prices):
        perf = formation_performance(prices)
        for name in prices.columns:
            expected = prices[name].iloc[-1] / prices[name].iloc[0] - 1.0
            assert perf[name] == pytest.approx(expected)

    def test_it_comes_back_sorted_worst_first(self, prices):
        perf = formation_performance(prices)
        assert list(perf.values) == sorted(perf.values)

    def test_every_pair_lands_in_exactly_one_bucket(self, prices):
        perf = formation_performance(prices)
        scan = pd.DataFrame({"a": ["AAA", "AAA", "BBB", "CCC", "DDD"],
                             "b": ["BBB", "CCC", "CCC", "DDD", "EEE"],
                             "pvalue": [0.01, 0.9, 0.04, 0.5, 0.02]})
        table = pass_rate_by_performance(scan, perf)
        assert table["pairs"].sum() == len(scan)

    def test_a_pair_is_bucketed_by_its_WEAKER_leg(self, prices):
        """'Both legs at least top quartile' is the claim, so the minimum."""
        perf = formation_performance(prices)
        best, worst = perf.index[-1], perf.index[0]
        scan = pd.DataFrame({"a": [best], "b": [worst], "pvalue": [0.01]})
        table = pass_rate_by_performance(scan, perf)
        assert len(table) == 1
        assert table["bucket"].iloc[0] == 1        # the bottom bucket, not the top
