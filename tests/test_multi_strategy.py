"""Tests for the cross-sleeve allocation study.

The study's conclusion is that mean-variance beats equal weighting here because
one sleeve is genuinely worse and the risk-based methods cannot see it. Every
step of that argument is a function, and each is tested on constructed sleeves
where the right answer is known in advance - a real two-sleeve panel has 46
months in it and would make these assertions a recording of one sample rather
than a check on the logic.

The numbers in the write-up are not asserted here. They depend on a download
and on the stat-arb sleeve's committed results, and pinning them in a test
would mean a data refresh shows up as a test failure instead of as a changed
write-up.
"""

import numpy as np
import pandas as pd
import pytest

from multi_strategy import (allocators, diversification_ratio, estimate,
                            lookback_sensitivity, performance,
                            return_gap_tstat, walk_forward)

MONTHS = 60


def panel(ret_a=0.000, ret_b=0.004, vol_a=0.004, vol_b=0.005, rho=0.0,
          n=MONTHS, seed=0) -> pd.DataFrame:
    """Two sleeves with chosen means, volatilities and correlation.

    Defaults mirror the real shape: sleeve A earns nothing at slightly lower
    volatility, sleeve B earns something, and they are uncorrelated.
    """
    rng = np.random.default_rng(seed)
    cov = [[vol_a ** 2, rho * vol_a * vol_b], [rho * vol_a * vol_b, vol_b ** 2]]
    draws = rng.multivariate_normal([ret_a, ret_b], cov, size=n)
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.DataFrame(draws, index=idx, columns=["statarb", "volcarry"])


# ---------- the summary statistics ----------

def test_performance_annualizes_monthly_input():
    monthly = pd.Series([0.01] * 12)
    p = performance(monthly)
    assert p["ann_return"] == pytest.approx(0.12)
    assert p["ann_vol"] == pytest.approx(0.0)
    assert p["hit_rate"] == 1.0


def test_sharpe_is_mean_over_vol_not_a_ratio_of_annualized_pieces():
    """Annualizing the mean by 12 and the vol by sqrt(12) and then dividing is
    the same as annualizing the monthly Sharpe by sqrt(12). If those ever
    disagree, one of them has the wrong power of 12 in it."""
    monthly = pd.Series([0.01, -0.005, 0.02, 0.0, 0.015, -0.01])
    p = performance(monthly)
    direct = monthly.mean() / monthly.std(ddof=1) * np.sqrt(12)
    assert p["sharpe"] == pytest.approx(direct)


def test_the_diversification_ratio_is_one_when_the_sleeves_are_identical():
    """Perfect correlation and equal volatility means combining buys nothing,
    and the ratio has to say exactly 1.00 rather than almost."""
    rng = np.random.default_rng(3)
    x = pd.Series(rng.normal(0, 0.01, MONTHS))
    frame = pd.DataFrame({"statarb": x, "volcarry": x})
    assert diversification_ratio(frame) == pytest.approx(1.0)


def test_the_diversification_ratio_approaches_root_two_at_zero_correlation():
    frame = panel(vol_a=0.005, vol_b=0.005, rho=0.0, n=1200, seed=5)
    assert diversification_ratio(frame) == pytest.approx(np.sqrt(2), abs=0.05)


def test_root_two_is_the_ceiling_only_for_non_negative_correlation():
    """Worth pinning because the bound is often quoted without the condition.
    Two anti-correlated sleeves can post a ratio far above it, which is real
    diversification and not a bug in the formula."""
    for rho in (0.0, 0.3, 0.6, 0.9):
        assert diversification_ratio(panel(rho=rho, n=600, seed=11)) <= np.sqrt(2) + 1e-6
    assert diversification_ratio(panel(rho=-0.9, n=600, seed=11)) > np.sqrt(2)


# ---------- the regime test ----------

def test_the_tstat_is_large_when_the_gap_is_real():
    frame = panel(ret_a=0.0, ret_b=0.006, vol_a=0.004, vol_b=0.004,
                  n=400, seed=1)
    assert return_gap_tstat(frame)["tstat"] > 4


def test_the_tstat_is_small_when_both_sleeves_have_the_same_mean():
    """The case that matters for the argument: equal means, so an optimizer
    allocating on the difference is allocating on noise. This is the regime
    notes/allocator.md is in, and the function has to be able to say so."""
    frame = panel(ret_a=0.003, ret_b=0.003, n=400, seed=2)
    assert abs(return_gap_tstat(frame)["tstat"]) < 2


def test_the_tstat_shrinks_as_the_sample_shrinks():
    """Same underlying gap, fewer months. Confidence has to fall, or the
    statistic is not measuring confidence."""
    long = return_gap_tstat(panel(n=400, seed=4))["tstat"]
    short = return_gap_tstat(panel(n=24, seed=4))["tstat"]
    assert abs(short) < abs(long)


def test_the_reported_gap_is_signed_toward_the_better_sleeve():
    assert return_gap_tstat(panel(ret_a=0.0, ret_b=0.005))["monthly_gap"] > 0
    assert return_gap_tstat(panel(ret_a=0.005, ret_b=0.0))["monthly_gap"] < 0


# ---------- the allocators ----------

def test_every_scheme_returns_weights_that_sum_to_one():
    frame = panel()
    mu, cov = estimate(frame)
    for name, fn in allocators().items():
        w = np.asarray(fn(mu, cov), dtype=float)
        assert w.sum() == pytest.approx(1.0, abs=1e-6), name


def test_the_risk_based_schemes_ignore_expected_returns_entirely():
    """The central claim of section 3, as an assertion. Change the means by a
    factor of ten and the four risk-based schemes must not move at all."""
    frame = panel()
    mu, cov = estimate(frame)
    risk_based = ("inverse vol", "risk parity", "HRP", "min variance")
    schemes = allocators()
    for name in risk_based:
        base = np.asarray(schemes[name](mu, cov), dtype=float)
        shocked = np.asarray(schemes[name](mu * 10, cov), dtype=float)
        assert base == pytest.approx(shocked), name


def test_max_sharpe_does_move_when_expected_returns_do():
    frame = panel()
    mu, cov = estimate(frame)
    base = np.asarray(allocators()["max Sharpe"](mu, cov), dtype=float)
    flipped = np.asarray(allocators()["max Sharpe"](mu[::-1], cov), dtype=float)
    assert not np.allclose(base, flipped)


def test_inverse_vol_tilts_toward_the_quieter_sleeve_whatever_it_earns():
    """Why the blind spot exists. The quieter sleeve gets more weight even when
    it is the one losing money, because no return ever enters the formula."""
    frame = panel(ret_a=-0.002, ret_b=0.006, vol_a=0.003, vol_b=0.009,
                  n=400, seed=7)
    mu, cov = estimate(frame)
    w = np.asarray(allocators()["inverse vol"](mu, cov), dtype=float)
    assert w[0] > w[1]


# ---------- the walk-forward ----------

def test_the_walk_forward_trades_every_month_after_the_first_window():
    frame = panel(n=40)
    result = walk_forward(frame, lookback=24)
    assert result["months_traded"] == 16
    assert set(result["schemes"]) == set(allocators())


def test_no_scheme_sees_the_month_it_is_allocating_into():
    """The only property that makes the comparison meaningful. Replace the
    final month with an enormous return: if any scheme's reported performance
    for the *earlier* months changes, something is reading ahead."""
    frame = panel(n=40)
    before = walk_forward(frame, lookback=24)

    tampered = frame.copy()
    tampered.iloc[-1] = [5.0, -5.0]
    after = walk_forward(tampered, lookback=24)

    for name in before["schemes"]:
        w_before = before["schemes"][name]["weight_on_statarb"]
        w_after = after["schemes"][name]["weight_on_statarb"]
        # The last month's allocation is made from months that did not change,
        # so every weight in the path is identical. Only the realized return
        # of that final month differs.
        # 1e-6 rather than exact: min_variance_weights goes through a convex
        # solver whose last few digits are not reproducible run to run.
        assert w_before == pytest.approx(w_after, abs=1e-6), name


def test_a_better_sleeve_earns_more_weight_from_max_sharpe_out_of_sample():
    frame = panel(ret_a=0.0, ret_b=0.008, vol_a=0.004, vol_b=0.004,
                  n=60, seed=8)
    result = walk_forward(frame, lookback=24)
    assert result["schemes"]["max Sharpe"]["weight_on_statarb"] < 0.2
    assert result["schemes"]["equal weight"]["weight_on_statarb"] == pytest.approx(0.5)


def test_equal_weight_always_holds_half_of_each_by_definition():
    result = walk_forward(panel(n=50), lookback=24)
    assert result["schemes"]["equal weight"]["weight_on_statarb"] == pytest.approx(0.5)


def test_the_lookback_sweep_skips_windows_too_long_for_the_sample():
    """A 30-month window on 32 months leaves two observations, which is not a
    result. The sweep has to drop those rather than report them."""
    rows = lookback_sensitivity(panel(n=32), lookbacks=(12, 24, 30))
    assert [r["lookback"] for r in rows] == [12, 24]


def test_shrinkage_pulls_the_covariance_toward_a_diagonal():
    frame = panel(rho=0.8, n=40, seed=9)
    _, plain = estimate(frame, shrink=False)
    _, shrunk = estimate(frame, shrink=True)
    assert abs(shrunk[0, 1]) <= abs(plain[0, 1]) + 1e-12


# ---------- the cost the sleeves were not matched on ----------
#
# volcarry sold its option at the mid while statarb paid bid-ask on every
# share, so the return gap between them was measured on unequal terms. The
# fix is one parameter and the study sweeps it; what is tested here is the
# mechanism and the one structural claim the sweep rests on - that a monthly
# half-spread is close enough to a constant drag that it moves the means and
# leaves the covariance alone. That is why it cannot reorder the risk-based
# allocators, and it is worth an assertion rather than a sentence.


def drag(p: pd.DataFrame, bps_per_month: float) -> pd.DataFrame:
    """Charge a constant monthly cost to the volcarry column only."""
    out = p.copy()
    out["volcarry"] = out["volcarry"] - bps_per_month
    return out


def test_a_constant_monthly_cost_does_not_move_the_correlation():
    base = panel()
    charged = drag(base, 0.0004)

    assert (return_gap_tstat(charged)["correlation"]
            == pytest.approx(return_gap_tstat(base)["correlation"]))


def test_a_constant_monthly_cost_does_not_move_the_diversification_ratio():
    base = panel()
    charged = drag(base, 0.0004)

    assert (diversification_ratio(charged)
            == pytest.approx(diversification_ratio(base)))


def test_a_constant_monthly_cost_shrinks_the_return_gap():
    base = panel()
    charged = drag(base, 0.0004)

    assert (return_gap_tstat(charged)["annualized_gap"]
            < return_gap_tstat(base)["annualized_gap"])
    assert return_gap_tstat(charged)["tstat"] < return_gap_tstat(base)["tstat"]


def test_the_return_blind_allocators_do_not_notice_the_cost_at_all():
    """The section 4 finding, restated as the thing that makes section 7 work.

    Inverse volatility, risk parity, HRP and minimum variance never read an
    expected return, so charging one sleeve a cost cannot move their weights
    by any amount. Mean-variance is the only one that reacts. If this ever
    fails, one of those four has started reading means and the blind-spot
    argument needs rewriting rather than patching.
    """
    base = walk_forward(panel(), lookback=24)["schemes"]
    charged = walk_forward(drag(panel(), 0.0010), lookback=24)["schemes"]

    for name in ("inverse vol", "risk parity", "HRP", "min variance"):
        assert (charged[name]["weight_on_statarb"]
                == pytest.approx(base[name]["weight_on_statarb"])), name

    assert (charged["max Sharpe"]["weight_on_statarb"]
            >= base["max Sharpe"]["weight_on_statarb"])


def test_a_big_enough_cost_eventually_makes_mean_variance_fund_the_other_sleeve():
    """The break-even the sweep is looking for, on a constructed panel.

    Charge enough that the good sleeve is the worse one and mean-variance has
    to switch. A study that reported a break-even without this would not have
    shown that one exists.
    """
    base = walk_forward(panel(), lookback=24)["schemes"]
    heavy = walk_forward(drag(panel(), 0.0060), lookback=24)["schemes"]

    # Not a hard floor on the first number: 60 months of draws put a little
    # weight on the flat sleeve in some windows even with no cost charged, and
    # a threshold tight enough to exclude that would be fitted to the seed.
    assert base["max Sharpe"]["weight_on_statarb"] < 0.3
    assert heavy["max Sharpe"]["weight_on_statarb"] > 0.8
    assert (heavy["max Sharpe"]["weight_on_statarb"]
            > base["max Sharpe"]["weight_on_statarb"] + 0.5)
