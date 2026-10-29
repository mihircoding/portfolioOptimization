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


# ---------- section 9, the third sleeve ----------

from multi_strategy import (three_sleeve_panel, weighted_diversification_ratio,
                            xsmom_monthly)


def test_weighted_ratio_agrees_with_equal_weight_version():
    p = panel()
    w = np.ones(2) / 2
    assert weighted_diversification_ratio(p, w) == pytest.approx(
        diversification_ratio(p))


def test_negative_correlation_can_beat_the_sqrt_n_ceiling():
    p = panel(rho=-0.6, n=600)
    assert weighted_diversification_ratio(p, np.ones(2) / 2) > np.sqrt(2)


def test_mismatched_scale_makes_equal_weight_the_big_sleeve():
    """Equal weight on a 1% sleeve and a 10% sleeve is a 10% sleeve."""
    p = panel(vol_a=0.004, vol_b=0.04, n=600)
    w = np.ones(2) / 2
    book = (p * w).sum(axis=1)
    big = p["volcarry"] * 0.5
    assert book.corr(big) > 0.98
    assert weighted_diversification_ratio(p, w) < 1.15


def test_xsmom_monthly_reads_the_net_column(tmp_path):
    idx = pd.date_range("2020-01-15", periods=3, freq="MS")
    frame = pd.DataFrame({"gross": [0.02, 0.01, 0.0], "net": [0.01, 0.0, -0.01],
                          "turnover": 0.5}, index=idx)
    from sleeves.xsmom.momentum import write_results
    path = tmp_path / "x.json"
    write_results(frame, str(path))
    s = xsmom_monthly(str(path))
    assert s.name == "xsmom"
    assert list(s.values) == [0.01, 0.0, -0.01]
    assert all(d.is_month_end for d in s.index)


def test_three_sleeve_panel_joins_on_the_month(monkeypatch):
    two = panel(n=12)
    extra = pd.Series(0.001, index=two.index[2:], name="xsmom")
    monkeypatch.setattr("multi_strategy.sleeve_panel", lambda **k: two)
    monkeypatch.setattr("multi_strategy.xsmom_monthly", lambda: extra)
    three = three_sleeve_panel()
    assert list(three.columns) == ["statarb", "volcarry", "xsmom"]
    assert len(three) == 10


# ---------- 10. the -0.46 in return units ----------

from multi_strategy import factor_exposure_study, regress


def factor(n=MONTHS, mean=0.0, vol=0.05, seed=1, name="x") -> pd.Series:
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    return pd.Series(rng.normal(mean, vol, size=n), index=idx, name=name)


def test_regress_recovers_a_known_alpha_and_beta():
    x = factor(n=600)
    noise = factor(n=600, vol=0.001, seed=2)
    y = 0.002 + 0.3 * x + noise
    r = regress(y.rename("y"), x.to_frame())
    assert r["beta"]["x"] == pytest.approx(0.3, abs=0.01)
    assert r["alpha_monthly"] == pytest.approx(0.002, abs=0.0002)
    assert r["alpha_ann"] == pytest.approx(12 * r["alpha_monthly"])
    assert r["n_months"] == 600


def test_one_factor_r2_is_the_correlation_squared():
    """The thing section 9 got wrong: -0.46 of correlation is 0.21 of variance."""
    x = factor(n=400)
    y = -0.4 * x + factor(n=400, vol=0.03, seed=3)
    r = regress(y.rename("y"), x.to_frame())
    assert r["r2"] == pytest.approx(y.corr(x) ** 2, abs=1e-9)
    assert r["r2"] < abs(y.corr(x))


def test_the_tstat_grows_with_the_sample_at_fixed_correlation():
    def fit(n):
        x = factor(n=n)
        y = (-0.4 * x + factor(n=n, vol=0.03, seed=3)).rename("y")
        return regress(y, x.to_frame())

    assert abs(fit(1000)["beta_t"]["x"]) > abs(fit(40)["beta_t"]["x"])


def test_hedged_equals_raw_when_the_sleeve_has_no_exposure():
    x = factor(n=200)
    y = factor(n=200, mean=0.003, vol=0.004, seed=5)
    # make y exactly orthogonal to x so the fitted beta is zero, not just small
    y = y - (y.cov(x) / x.var()) * x
    r = regress(y.rename("y"), x.to_frame())
    assert r["beta"]["x"] == pytest.approx(0.0, abs=1e-12)
    assert r["hedged"]["sharpe"] == pytest.approx(r["raw"]["sharpe"])


def test_hedging_a_zero_mean_tilt_raises_the_sharpe():
    """Comovement with a factor that earns nothing is pure variance."""
    x = factor(n=400, mean=0.0, vol=0.05)
    y = 0.003 + factor(n=400, vol=0.004, seed=5) - 0.05 * x
    r = regress(y.rename("y"), x.to_frame())
    assert r["hedged"]["ann_vol"] < r["raw"]["ann_vol"]
    assert r["hedged"]["sharpe"] > r["raw"]["sharpe"]


def test_hedging_a_paid_tilt_takes_the_return_with_it():
    """Short a factor that lost money over the window, and the tilt was income.

    This is the real-panel case: momentum lost over 2021-24, the stat-arb
    sleeve was short it, and hedging the tilt makes the sleeve worse.
    """
    x = factor(n=400, mean=-0.01, vol=0.05)
    y = 0.0005 + factor(n=400, vol=0.004, seed=5) - 0.05 * x
    r = regress(y.rename("y"), x.to_frame())
    assert r["hedged"]["ann_return"] < r["raw"]["ann_return"]
    assert r["hedged"]["ann_return"] == pytest.approx(
        r["raw"]["ann_return"] - 12 * r["beta"]["x"] * float(x.mean()), rel=1e-6)


def test_two_factors_split_a_shared_exposure():
    """A sleeve short momentum, where momentum is itself short the market,
    shows a market beta on its own that disappears once momentum is in the
    regression. That is the pattern the real stat-arb sleeve has."""
    mkt = factor(n=2000, vol=0.04, seed=7, name="market")
    mom = (-0.5 * mkt + factor(n=2000, vol=0.04, seed=8)).rename("xsmom")
    y = (0.001 - 0.04 * mom + factor(n=2000, vol=0.004, seed=9)).rename("y")
    alone = regress(y, mkt.to_frame())
    both = regress(y, pd.concat([mkt, mom], axis=1))
    assert alone["beta"]["market"] > 0.01
    assert abs(both["beta"]["market"]) < 0.005
    assert both["beta"]["xsmom"] == pytest.approx(-0.04, abs=0.005)


def test_factor_exposure_study_reports_the_variance_share_as_r_squared():
    two = panel(n=48, rho=0.0)
    mom = factor(n=48, vol=0.05, seed=11, name="xsmom")
    three = pd.concat([two, mom], axis=1)
    mkt = factor(n=48, vol=0.04, seed=12, name="market")
    study = factor_exposure_study(panel=three, market=mkt)
    rho = three["statarb"].corr(three["xsmom"])
    cv = study["correlation_vs_variance_share"]
    assert cv["correlation"] == pytest.approx(rho)
    assert cv["variance_share"] == pytest.approx(rho ** 2)
    assert set(study["on_market"]) == {"statarb", "volcarry", "xsmom"}
    assert study["statarb_on_xsmom"]["r2"] == pytest.approx(rho ** 2)
    assert set(study["on_market_and_xsmom"]["statarb"]["beta"]) == {"market", "xsmom"}


# ---------- 11. beta-hedge the momentum sleeve before it goes in ----------

from multi_strategy import beta_hedged, hedged_momentum_study, trailing_beta


def test_trailing_beta_is_nan_until_the_window_is_full():
    x = factor(n=60)
    y = (0.5 * x + factor(n=60, vol=0.01, seed=2)).rename("y")
    beta = trailing_beta(y, x, window=36)
    # 35 months of history give no estimate; the 36th gives the first one,
    # which is used on the 37th month, so 36 NaNs in all
    assert beta.isna().sum() == 36
    assert beta.iloc[36:].notna().all()


def test_trailing_beta_recovers_a_constant_beta():
    x = factor(n=400)
    y = (-0.4 * x + factor(n=400, vol=0.001, seed=2)).rename("y")
    beta = trailing_beta(y, x, window=36).dropna()
    assert beta.mean() == pytest.approx(-0.4, abs=0.01)
    assert beta.std() < 0.02


def test_trailing_beta_never_sees_the_month_it_is_used_on():
    """Change the last month of the data and no earlier beta may move.

    This is the whole point of the trailing estimate. The full-window beta in
    section 10 was fitted on the months it would then have been used to hedge.
    """
    x = factor(n=100)
    y = (0.3 * x + factor(n=100, vol=0.01, seed=2)).rename("y")
    before = trailing_beta(y, x, window=24)
    y2 = y.copy()
    y2.iloc[-1] += 0.5
    x2 = x.copy()
    x2.iloc[-1] -= 0.5
    after = trailing_beta(y2, x2, window=24)
    pd.testing.assert_series_equal(before, after)


def test_trailing_beta_lags_a_change_in_the_true_beta():
    """A beta that flips at month 200 is still reported as the old one at
    month 201, and only reaches the new one a window later. That lag is why
    the hedged sleeve keeps a residual market beta after the fact."""
    x = factor(n=400)
    noise = factor(n=400, vol=0.001, seed=2)
    true = pd.Series(np.where(np.arange(400) < 200, 0.5, -0.5), index=x.index)
    y = (true * x + noise).rename("y")
    beta = trailing_beta(y, x, window=36)
    assert beta.iloc[200] == pytest.approx(0.5, abs=0.02)
    assert beta.iloc[236] == pytest.approx(-0.5, abs=0.02)
    assert 0.5 > beta.iloc[218] > -0.5


def test_beta_hedged_removes_a_constant_exposure():
    x = factor(n=400, mean=0.01)
    y = (0.002 - 0.4 * x + factor(n=400, vol=0.002, seed=2)).rename("y")
    out = beta_hedged(y, x, window=36)
    assert list(out.columns) == ["raw", "factor", "beta", "hedged"]
    assert len(out) == 400 - 36
    assert abs(out["hedged"].corr(out["factor"])) < 0.1
    assert abs(y.corr(x)) > 0.9


def test_beta_hedged_arithmetic_is_raw_minus_beta_times_factor():
    x = factor(n=80)
    y = factor(n=80, vol=0.03, seed=4).rename("y")
    out = beta_hedged(y, x, window=12)
    recomputed = out["raw"] - out["beta"] * out["factor"]
    pd.testing.assert_series_equal(out["hedged"], recomputed, check_names=False)


def test_hedging_a_losing_factor_raises_the_sleeves_return():
    """The real case: momentum was short a market that went up. Take the
    short out and the sleeve keeps its own return without the drag."""
    x = factor(n=400, mean=0.01, vol=0.04)
    y = (0.001 - 0.5 * x + factor(n=400, vol=0.03, seed=2)).rename("y")
    out = beta_hedged(y, x, window=36)
    assert out["hedged"].mean() > out["raw"].mean()
    assert performance(out["hedged"])["sharpe"] > performance(out["raw"])["sharpe"]


def test_hedged_momentum_study_runs_both_panels_on_the_same_months(monkeypatch):
    n = 120
    idx = pd.date_range("2014-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(21)
    market = pd.Series(rng.normal(0.008, 0.03, n), index=idx, name="market")
    xsmom = (0.004 - 0.5 * market + rng.normal(0.0, 0.02, n)).rename("xsmom")
    two = panel(n=60).set_index(idx[-60:])
    three = pd.concat([two, xsmom], axis=1).dropna()
    monkeypatch.setattr("multi_strategy.xsmom_monthly", lambda: xsmom)
    monkeypatch.setattr("multi_strategy.market_monthly", lambda: market)
    monkeypatch.setattr("multi_strategy.three_sleeve_panel", lambda **k: three)
    monkeypatch.setattr("multi_strategy.momentum_crashes",
                        lambda s, n=3: [{"month": f"{d:%Y-%m}"}
                                        for d in s.nsmallest(n).index])
    study = hedged_momentum_study(lookback=24, window=36)
    raw = study["walk_forward_raw"]
    hedged = study["walk_forward_hedged"]
    assert raw["months_traded"] == hedged["months_traded"]
    assert raw["first"] == hedged["first"]
    assert study["overlap"]["n_months"] == 60
    assert study["full_window_beta"] == pytest.approx(-0.5, abs=0.2)
    assert abs(study["overlap"]["market_beta_after_hedge"]) < 0.15
    assert study["trailing_beta"]["share_of_months_negative"] == 1.0
    # a worst month before the first full window has no hedge ratio to report
    crashes = study["trailing_beta"]["at_crash_months"]
    assert 1 <= len(crashes) <= 3
    assert all(b < 0 for b in crashes.values())


# ---------- 12. lever every sleeve to the same risk ----------

from multi_strategy import (statarb_trailing_vol, trailing_vol, vol_target_study,
                            vol_targeted)


def test_trailing_vol_is_nan_until_the_window_is_full():
    x = factor(n=60)
    vol = trailing_vol(x, window=36)
    assert vol.isna().sum() == 36
    assert vol.iloc[36:].notna().all()


def test_trailing_vol_recovers_a_constant_vol():
    x = factor(n=2000, vol=0.03)
    vol = trailing_vol(x, window=36).dropna()
    assert vol.mean() == pytest.approx(0.03 * np.sqrt(12), rel=0.05)


def test_trailing_vol_never_sees_the_month_it_is_used_on():
    x = factor(n=100)
    before = trailing_vol(x, window=24)
    x2 = x.copy()
    x2.iloc[-1] += 1.0
    after = trailing_vol(x2, window=24)
    pd.testing.assert_series_equal(before, after)


def test_vol_targeted_lands_on_the_target_when_vol_is_constant():
    x = factor(n=400, vol=0.03)
    out = vol_targeted(x, trailing_vol(x, window=36), target=0.10)
    assert list(out.columns) == ["raw", "vol", "leverage", "scaled"]
    assert len(out) == 400 - 36
    realized = out["scaled"].std(ddof=1) * np.sqrt(12)
    assert realized == pytest.approx(0.10, rel=0.05)


def test_vol_targeted_arithmetic_is_raw_times_target_over_vol():
    x = factor(n=80, vol=0.02)
    out = vol_targeted(x, trailing_vol(x, window=12), target=0.10)
    pd.testing.assert_series_equal(out["leverage"], 0.10 / out["vol"], check_names=False)
    pd.testing.assert_series_equal(out["scaled"], out["raw"] * out["leverage"],
                                   check_names=False)


def test_vol_targeted_leverage_cap_binds_on_a_quiet_sleeve():
    x = factor(n=200, vol=0.004)     # 1.4% a year: asks for about 7x
    free = vol_targeted(x, trailing_vol(x, window=36), target=0.10)
    capped = vol_targeted(x, trailing_vol(x, window=36), target=0.10, max_leverage=3.0)
    assert free["leverage"].min() > 3.0
    assert (capped["leverage"] == 3.0).all()
    assert capped["scaled"].std() < free["scaled"].std()


def test_scaling_a_constant_vol_sleeve_leaves_its_sharpe_alone():
    x = factor(n=400, mean=0.005, vol=0.03)
    out = vol_targeted(x, trailing_vol(x, window=36), target=0.10)
    assert (performance(out["scaled"])["sharpe"]
            == pytest.approx(performance(out["raw"])["sharpe"], abs=0.05))


def test_vol_targeting_is_still_levered_up_when_the_vol_regime_breaks():
    """The rule sizes month t on the vol of the months before it, so the first
    month of a vol spike is taken at the old leverage. That is why the
    leverage going into the worst month in the write-up is the quiet-regime
    number, and why the scaled worst month is not target / raw vol times the
    raw worst month."""
    rng = np.random.default_rng(3)
    idx = pd.date_range("2000-01-31", periods=400, freq="ME")
    vol = np.where(np.arange(400) < 200, 0.01, 0.05)
    x = pd.Series(rng.normal(0.0, vol), index=idx, name="x")
    out = vol_targeted(x, trailing_vol(x, window=36), target=0.10)
    quiet = 0.10 / (0.01 * np.sqrt(12))
    assert out["leverage"].iloc[200 - 36] == pytest.approx(quiet, rel=0.15)
    # three years later the window is all loud and the leverage is a fifth of it
    assert out["leverage"].iloc[236 - 36] == pytest.approx(quiet / 5, rel=0.15)


def test_statarb_trailing_vol_stamps_the_prior_months_estimate(tmp_path):
    days = pd.bdate_range("2021-01-04", periods=260)
    rng = np.random.default_rng(5)
    rets = rng.normal(0.0, 0.001, size=len(days))
    equity = 100.0 * np.cumprod(1 + rets)
    path = tmp_path / "portfolio.json"
    path.write_text(pd.Series(equity, index=days).to_frame("e").reset_index()
                    .apply(lambda r: [r["index"].strftime("%Y-%m-%d"), r["e"]], axis=1)
                    .pipe(lambda s: '{"equity": ' + s.to_json(orient="values") + "}"))
    vol = statarb_trailing_vol(str(path), window_days=63)
    assert vol.index.is_month_end.all()
    # the 63rd daily return lands on 1 April, so the first full window closes
    # in April and the first month it can size is May
    assert vol.dropna().index[0] == pd.Timestamp("2021-05-31")
    daily = pd.Series(equity, index=days).pct_change().dropna()
    april = daily.loc[:"2021-04-30"].iloc[-63:].std(ddof=1) * np.sqrt(252)
    assert vol.loc["2021-05-31"] == pytest.approx(april)
    # and nothing from May is in May's number
    assert vol.loc["2021-05-31"] != pytest.approx(
        daily.loc[:"2021-05-31"].iloc[-63:].std(ddof=1) * np.sqrt(252))


def test_vol_target_study_runs_both_panels_on_the_same_months(monkeypatch):
    n = 120
    idx = pd.date_range("2014-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(21)
    market = pd.Series(rng.normal(0.008, 0.03, n), index=idx, name="market")
    xsmom = (0.004 - 0.5 * market + rng.normal(0.0, 0.02, n)).rename("xsmom")
    two = panel(n=60).set_index(idx[-60:])
    days = pd.bdate_range(idx[-60] - pd.offsets.MonthBegin(1), idx[-1])
    daily_vol = pd.Series(0.004 * np.sqrt(12 / 252), index=days)   # ~1.4% a year, flat
    monkeypatch.setattr("multi_strategy.xsmom_monthly", lambda: xsmom)
    monkeypatch.setattr("multi_strategy.market_monthly", lambda: market)
    monkeypatch.setattr("multi_strategy.statarb_monthly", lambda: two["statarb"])
    monkeypatch.setattr("multi_strategy.volcarry_monthly", lambda **k: two["volcarry"])
    monkeypatch.setattr("multi_strategy.statarb_trailing_vol",
                        lambda **k: (daily_vol * np.sqrt(252)).resample("ME").last().shift(1))
    study = vol_target_study(lookback=24, target=0.10, window=24, beta_window=36)
    raw, scaled = study["walk_forward_raw"], study["walk_forward_scaled"]
    assert raw["months_traded"] == scaled["months_traded"]
    assert raw["first"] == scaled["first"]
    for name, s in study["sleeves"].items():
        assert 0.6 < s["realized_over_target"] < 1.6, name
        assert s["leverage"]["min"] > 0
    # equal weight was a momentum book with two small positions attached; at a
    # common scale it is a third of each by risk, not only by capital
    ew_raw = raw["schemes"]["equal weight"]["ann_vol"]
    ew_scaled = scaled["schemes"]["equal weight"]["ann_vol"]
    assert ew_scaled > 2 * ew_raw
    assert 0.04 < ew_scaled < 0.10      # three ~10% sleeves, not all correlated


# ---------- 13. size the vol sleeve on the vol it sells ----------

from multi_strategy import (implied_sizing_study, implied_vol_forecast,
                            leverage_alignment, volcarry_implied)


def _implied_sleeve(n=120, seed=3):
    """A sleeve whose monthly return scales with the vol it sold: r = k * iv * z."""
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2007-01-31", periods=n, freq="ME")
    iv = pd.Series(rng.uniform(0.10, 0.40, n), index=idx, name="implied")
    z = pd.Series(rng.normal(0.0, 1.0, n), index=idx)
    return (0.02 * iv * z).rename("volcarry"), iv


def test_implied_forecast_is_nan_until_there_is_history_to_fit_on():
    r, iv = _implied_sleeve()
    vol = implied_vol_forecast(r, iv, min_months=36)
    assert vol.iloc[:36].isna().all()
    assert vol.iloc[36:].notna().all()


def test_implied_forecast_recovers_the_ratio_when_it_is_exact():
    # r^2 = k^2 iv^2 every month, so the expanding ratio is k^2 from the first
    # month and the forecast is sqrt(12) * k * iv_t exactly
    idx = pd.date_range("2007-01-31", periods=48, freq="ME")
    iv = pd.Series(np.linspace(0.10, 0.40, 48), index=idx)
    r = pd.Series(0.05 * iv.to_numpy() * np.where(np.arange(48) % 2 == 0, 1, -1), index=idx)
    vol = implied_vol_forecast(r, iv, min_months=12).dropna()
    expected = np.sqrt(12) * 0.05 * iv.loc[vol.index]
    assert np.allclose(vol, expected)


def test_implied_forecast_never_sees_the_month_it_sizes():
    r, iv = _implied_sleeve()
    before = implied_vol_forecast(r, iv)
    bumped = r.copy()
    bumped.iloc[-1] = 0.5      # a catastrophe in the last month
    after = implied_vol_forecast(bumped, iv)
    # nothing earlier moves, and neither does the last month's own forecast,
    # because the ratio it uses was closed before the month began
    pd.testing.assert_series_equal(before, after)


def test_implied_forecast_moves_with_this_months_implied_and_the_leverage_against_it():
    r, iv = _implied_sleeve()
    doubled = iv.copy()
    doubled.iloc[-1] *= 2
    base = implied_vol_forecast(r, iv)
    high = implied_vol_forecast(r, doubled)
    assert high.iloc[-1] == pytest.approx(2 * base.iloc[-1])
    pd.testing.assert_series_equal(base.iloc[:-1], high.iloc[:-1])
    lev_base = vol_targeted(r, base, 0.10)["leverage"].iloc[-1]
    lev_high = vol_targeted(r, high, 0.10)["leverage"].iloc[-1]
    assert lev_high == pytest.approx(lev_base / 2)


def test_implied_sizing_lands_nearer_the_target_than_trailing_when_vol_is_known():
    # the sleeve's vol genuinely is proportional to implied here, so the rule
    # that reads implied should hit the target and the one that reads the last
    # three years should miss it by the dispersion of implied
    r, iv = _implied_sleeve(n=240)
    on_implied = vol_targeted(r, implied_vol_forecast(r, iv), 0.10)["scaled"]
    on_trailing = vol_targeted(r, trailing_vol(r, 36), 0.10)["scaled"]
    common = on_implied.index.intersection(on_trailing.index)
    miss_implied = abs(performance(on_implied.loc[common])["ann_vol"] - 0.10)
    miss_trailing = abs(performance(on_trailing.loc[common])["ann_vol"] - 0.10)
    assert miss_implied < miss_trailing


def test_leverage_alignment_is_negative_when_the_rule_sees_the_big_months_coming():
    r, iv = _implied_sleeve(n=240)
    seen = vol_targeted(r, implied_vol_forecast(r, iv), 0.10)
    # a constant forecast has no view at all, so nothing to correlate with
    flat = vol_targeted(r, pd.Series(0.05, index=r.index), 0.10)
    assert leverage_alignment(seen) < -0.2
    assert abs(leverage_alignment(flat)) < 1e-12 or np.isnan(leverage_alignment(flat))


def test_volcarry_implied_stamps_each_sale_on_its_month_end(tmp_path):
    detail = pd.DataFrame({
        "implied": [0.12, 0.18, 0.33],
        "half_spread": 0.07, "spread_cost": 0.01, "premium": 1.0, "pnl": 0.4,
    }, index=pd.to_datetime(["2020-01-02", "2020-02-03", "2020-03-02"]))
    detail.index.name = "date"
    path = tmp_path / "detail.csv"
    detail.to_csv(path)
    implied = volcarry_implied(cache=str(path))
    assert list(implied.index) == list(pd.to_datetime(["2020-01-31", "2020-02-29", "2020-03-31"]))
    assert implied.loc["2020-03-31"] == pytest.approx(0.33)


def test_vol_target_study_sizes_the_vol_sleeve_on_the_forecast_it_is_handed(monkeypatch):
    n = 120
    idx = pd.date_range("2014-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(21)
    market = pd.Series(rng.normal(0.008, 0.03, n), index=idx, name="market")
    xsmom = (0.004 - 0.5 * market + rng.normal(0.0, 0.02, n)).rename("xsmom")
    two = panel(n=60).set_index(idx[-60:])
    days = pd.bdate_range(idx[-60] - pd.offsets.MonthBegin(1), idx[-1])
    daily_vol = pd.Series(0.004 * np.sqrt(12 / 252), index=days)
    monkeypatch.setattr("multi_strategy.xsmom_monthly", lambda: xsmom)
    monkeypatch.setattr("multi_strategy.market_monthly", lambda: market)
    monkeypatch.setattr("multi_strategy.statarb_monthly", lambda: two["statarb"])
    monkeypatch.setattr("multi_strategy.volcarry_monthly", lambda **k: two["volcarry"])
    monkeypatch.setattr("multi_strategy.statarb_trailing_vol",
                        lambda **k: (daily_vol * np.sqrt(252)).resample("ME").last().shift(1))
    # 2% a year, flat, on the months the trailing rule would also have sized
    handed = pd.Series(0.02, index=idx[-60:][24:], name="vol")
    study = vol_target_study(lookback=24, target=0.10, window=24, beta_window=36,
                             volcarry_vol=handed)
    lev = study["sleeves"]["volcarry"]["leverage"]
    assert lev["min"] == pytest.approx(5.0) and lev["max"] == pytest.approx(5.0)
    # and the other two sleeves were sized exactly as before
    default = vol_target_study(lookback=24, target=0.10, window=24, beta_window=36)
    for name in ("statarb", "xsmom"):
        assert study["sleeves"][name]["leverage"] == default["sleeves"][name]["leverage"]


def test_implied_sizing_study_runs_both_allocator_panels_on_the_same_months(monkeypatch):
    n = 120
    idx = pd.date_range("2014-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(8)
    market = pd.Series(rng.normal(0.008, 0.03, n), index=idx, name="market")
    xsmom = (0.004 - 0.5 * market + rng.normal(0.0, 0.02, n)).rename("xsmom")
    volcarry, iv = _implied_sleeve(n=n, seed=4)
    volcarry.index = idx
    iv.index = idx
    statarb = pd.Series(rng.normal(0.0, 0.004, 60), index=idx[-60:], name="statarb")
    days = pd.bdate_range(idx[-60] - pd.offsets.MonthBegin(1), idx[-1])
    daily_vol = pd.Series(0.004 * np.sqrt(12 / 252), index=days)
    monkeypatch.setattr("multi_strategy.xsmom_monthly", lambda: xsmom)
    monkeypatch.setattr("multi_strategy.market_monthly", lambda: market)
    monkeypatch.setattr("multi_strategy.statarb_monthly", lambda: statarb)
    monkeypatch.setattr("multi_strategy.volcarry_monthly", lambda **k: volcarry)
    monkeypatch.setattr("multi_strategy.volcarry_implied", lambda **k: iv)
    monkeypatch.setattr("multi_strategy.statarb_trailing_vol",
                        lambda **k: (daily_vol * np.sqrt(252)).resample("ME").last().shift(1))
    levered = vol_target_study(lookback=24, target=0.10, window=36, beta_window=36)
    study = implied_sizing_study(levered, lookback=24, target=0.10, window=36)
    tr, im = study["walk_forward_trailing"], study["walk_forward_implied"]
    assert tr["months_traded"] == im["months_traded"]
    assert tr["first"] == im["first"]
    assert study["sleeve"]["trailing"]["own_history"]["sharpe"] != study["sleeve"]["implied"]["own_history"]["sharpe"]
    # the trailing column is section 11's number, not a re-estimate of it
    assert tr == levered["walk_forward_scaled"]
    # and both rules are reported on one set of months
    assert set(study["sleeve"]["trailing"]["leverage_into_worst_months"]) == \
        set(study["sleeve"]["implied"]["leverage_into_worst_months"])


# ---------- 13. charge the leverage ----------

from multi_strategy import (atm_call_delta, fed_funds_monthly, financed_frames,
                            financing_study, long_short_financing,
                            margin_loan_financing, sleeve_frames, statarb_occupancy)


def test_fed_funds_is_a_decimal_on_month_ends(tmp_path):
    raw = tmp_path / "fredgraph.csv"
    raw.write_text("observation_date,FEDFUNDS\n2023-01-01,4.33\n2023-02-01,4.57\n")
    cache = tmp_path / "results" / "fed_funds.csv"
    ff = fed_funds_monthly(cache=str(cache), url=str(raw))
    assert list(ff.index) == list(pd.to_datetime(["2023-01-31", "2023-02-28"]))
    assert ff.loc["2023-02-28"] == pytest.approx(0.0457)
    # and the second call reads the cache rather than the "download"
    raw.write_text("observation_date,FEDFUNDS\n2023-01-01,9.99\n")
    assert fed_funds_monthly(cache=str(cache), url=str(raw)).loc["2023-01-31"] == pytest.approx(0.0433)


def test_occupancy_is_the_share_of_pairs_with_a_position_on(tmp_path):
    # two pairs: one whose spread never moves (never trades) and one that is
    # pushed three sigma wide and held there, so it enters and stays in
    days = pd.bdate_range("2021-01-04", periods=120)
    rng = np.random.default_rng(0)
    x = pd.Series(100.0 + rng.normal(0, 0.5, len(days)).cumsum(), index=days)
    y = x + 10.0                                   # a constant spread
    z = pd.Series(100.0 + rng.normal(0, 0.5, len(days)).cumsum(), index=days)
    w = z + 10.0 + np.r_[np.zeros(60), np.full(60, 8.0)]   # a spread that blows out at day 60
    trading = pd.DataFrame({"A": y, "B": x, "C": w, "D": z})
    results = pd.DataFrame({"a": ["A", "C"], "b": ["B", "D"], "beta": [1.0, 1.0]})
    trading.to_parquet(tmp_path / "trading.parquet")
    results.to_parquet(tmp_path / "results.parquet")
    occ = statarb_occupancy(str(tmp_path / "results.parquet"), str(tmp_path / "trading.parquet"))
    assert occ.index.is_month_end.all()
    assert occ.iloc[0] == pytest.approx(0.0)       # nothing in a position before the window fills
    assert 0.0 < occ.iloc[-1] <= 0.5               # one of the two pairs is in, at most
    assert (occ >= 0).all() and (occ <= 1).all()


def test_atm_call_delta_is_a_little_over_a_half_and_rises_with_vol():
    iv = pd.Series([0.10, 0.20, 0.40, 0.80])
    delta = atm_call_delta(iv, t_years=21 / 252)
    assert (delta > 0.5).all()
    assert delta.is_monotonic_increasing
    assert delta.iloc[0] == pytest.approx(0.5, abs=0.02)


def test_long_short_financing_has_no_rate_in_it_and_scales_with_leverage():
    idx = pd.date_range("2023-01-31", periods=3, freq="ME")
    lev = pd.Series([1.0, 2.0, 4.0], index=idx)
    one = pd.Series(1.0, index=idx)
    charge = long_short_financing(lev, one, one, spread_bps=50, borrow_fee_bps=50)
    # a dollar long and a dollar short at 1x: 50bp on each leg plus 50bp borrow
    assert charge.iloc[0] * 12 == pytest.approx(0.0150)
    assert charge.iloc[2] == pytest.approx(4 * charge.iloc[0])
    assert long_short_financing(lev, one, one, 0, 0).eq(0).all()


def test_margin_loan_is_free_until_the_hedge_outgrows_the_capital():
    idx = pd.date_range("2023-01-31", periods=4, freq="ME")
    lev = pd.Series([1.0, 2.0, 3.0, 3.0], index=idx)
    delta = pd.Series(0.5, index=idx)
    ff = pd.Series([0.05, 0.05, 0.05, 0.01], index=idx)
    charge = margin_loan_financing(lev, delta, ff, spread_bps=50)
    assert charge.iloc[0] == 0.0 and charge.iloc[1] == 0.0    # half a dollar of stock per dollar of capital, twice: still covered
    assert charge.iloc[2] * 12 == pytest.approx(0.5 * 0.055)   # half a dollar borrowed at 5.5%
    assert charge.iloc[3] < charge.iloc[2]                     # and cheaper when the rate falls


def _frames_for(n=48, seed=5):
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    frames = {}
    for name, lev in (("statarb", 6.0), ("volcarry", 4.0), ("xsmom", 0.6)):
        raw = pd.Series(rng.normal(0.002, 0.01, n), index=idx)
        frames[name] = pd.DataFrame({"raw": raw, "vol": 0.10 / lev,
                                     "leverage": lev, "scaled": raw * lev})
    ff = pd.Series(0.05, index=idx)
    occ = pd.Series(0.5, index=idx)
    delta = pd.Series(0.5, index=idx)
    return frames, ff, occ, delta


def test_financed_net_is_scaled_minus_the_charge_and_the_cap_binds_on_the_charge_too():
    frames, ff, occ, delta = _frames_for()
    fin = financed_frames(frames, ff, occ, delta, spread_bps=50, borrow_fee_bps=50, cap=5.0)
    for name, f in fin.items():
        assert np.allclose(f["net"], f["scaled"] - f["charge"])
        assert (f["charge"] >= 0).all()
    # pairs: a quarter long, a quarter short at 1x, times 6x
    pairs = fin["statarb"]
    assert pairs["gross"].iloc[0] == pytest.approx(6.0 * 0.5)
    assert pairs["charge"].iloc[0] * 12 == pytest.approx(6.0 * (0.5 * 0.005 + 0.25 * 0.005))
    assert pairs["charge_capped"].iloc[0] * 12 == pytest.approx(5.0 * (0.5 * 0.005 + 0.25 * 0.005))
    # vol sleeve: 0.5 x 4 = 2 dollars of stock on 1 of capital, 1 borrowed at 5.5%
    vol = fin["volcarry"]
    assert vol["charge"].iloc[0] * 12 == pytest.approx(1.0 * 0.055)
    assert vol["short"].eq(0).all()
    # momentum at 0.6x never borrows anything and pays spreads and borrow only
    assert fin["xsmom"]["charge"].iloc[0] * 12 == pytest.approx(0.6 * (2 * 0.005 + 0.005))


def test_the_sleeve_with_a_cash_leg_is_the_one_the_rate_touches():
    frames, ff, occ, delta = _frames_for()
    low = financed_frames(frames, ff * 0.0, occ, delta)
    high = financed_frames(frames, ff * 2.0, occ, delta)
    for name in ("statarb", "xsmom"):
        assert np.allclose(low[name]["charge"], high[name]["charge"])
    assert (high["volcarry"]["charge"] > low["volcarry"]["charge"]).all()


def _patch_sleeves(monkeypatch, n=120, seed=8):
    idx = pd.date_range("2014-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    market = pd.Series(rng.normal(0.008, 0.03, n), index=idx, name="market")
    xsmom = (0.004 - 0.5 * market + rng.normal(0.0, 0.02, n)).rename("xsmom")
    volcarry, iv = _implied_sleeve(n=n, seed=4)
    volcarry.index = idx
    iv.index = idx
    statarb = pd.Series(rng.normal(0.0, 0.004, 60), index=idx[-60:], name="statarb")
    days = pd.bdate_range(idx[-60] - pd.offsets.MonthBegin(1), idx[-1])
    daily_vol = pd.Series(0.004 * np.sqrt(12 / 252), index=days)
    monkeypatch.setattr("multi_strategy.xsmom_monthly", lambda: xsmom)
    monkeypatch.setattr("multi_strategy.market_monthly", lambda: market)
    monkeypatch.setattr("multi_strategy.statarb_monthly", lambda: statarb)
    monkeypatch.setattr("multi_strategy.volcarry_monthly", lambda **k: volcarry)
    monkeypatch.setattr("multi_strategy.volcarry_implied", lambda **k: iv)
    monkeypatch.setattr("multi_strategy.statarb_trailing_vol",
                        lambda **k: (daily_vol * np.sqrt(252)).resample("ME").last().shift(1))
    monkeypatch.setattr("multi_strategy.fed_funds_monthly",
                        lambda **k: pd.Series(0.04, index=idx, name="fed_funds"))
    monkeypatch.setattr("multi_strategy.statarb_occupancy",
                        lambda **k: pd.Series(0.45, index=idx[-60:], name="occupancy"))


def test_vol_target_study_still_sizes_exactly_as_it_did_before_the_refactor(monkeypatch):
    _patch_sleeves(monkeypatch)
    frames = sleeve_frames(target=0.10, window=36, beta_window=36)
    study = vol_target_study(lookback=24, target=0.10, window=36, beta_window=36)
    overlap = pd.DataFrame({n: f["scaled"] for n, f in frames.items()}).dropna().index
    for name in ("statarb", "volcarry", "xsmom"):
        s = study["sleeves"][name]
        assert s["own_history_months"] == len(frames[name])
        assert s["own_history_scaled"]["sharpe"] == pytest.approx(
            performance(frames[name]["scaled"])["sharpe"])
        assert s["leverage"]["mean"] == pytest.approx(
            float(frames[name]["leverage"].loc[overlap].mean()))


def test_free_leverage_reproduces_section_12_and_paid_leverage_costs_sharpe(monkeypatch):
    _patch_sleeves(monkeypatch)
    levered = vol_target_study(lookback=24, target=0.10, window=36, beta_window=36)
    implied = implied_sizing_study(levered, lookback=24, target=0.10, window=36)
    study = financing_study(lookback=24, target=0.10, window=36, beta_window=36,
                            settings=((0.0, 0.0), (1.0, 1.0)))
    free, paid = study["sensitivity"]
    for scheme, s in implied["walk_forward_implied_capped"]["schemes"].items():
        assert free["schemes"][scheme] == pytest.approx(s["sharpe"])
        assert paid["schemes"][scheme] < free["schemes"][scheme]
    rule = study["rules"]["trailing"]
    assert rule["walk_forward_gross"] == levered["walk_forward_scaled"]
    for name, s in rule["sleeves"].items():
        assert s["own_history_net"]["sharpe"] < s["own_history_gross"]["sharpe"]
        assert s["drag_own_history"] > 0
    # a 4% rate on a vol sleeve that borrows is a charge; on the two
    # long/short books it is not
    assert rule["sleeves"]["volcarry"]["drag_own_history"] > rule["sleeves"]["xsmom"]["drag_own_history"]


# ---------- section 14: the window re-charged at today's rate ----------

from multi_strategy import (_financed_panels, funding_aware_walk_forward,
                            funding_forecast_study, repriced_window)


def _financed_for(rate=0.05, **kw):
    frames, ff, occ, delta = _frames_for(**kw)
    ff = pd.Series(rate, index=ff.index)
    return financed_frames(frames, ff, occ, delta, spread_bps=50, borrow_fee_bps=50, cap=5.0), ff


def test_repricing_only_touches_the_sleeve_with_a_loan():
    fin, ff = _financed_for()
    window = fin["volcarry"].index[:24]
    cheap = repriced_window(fin, window, 0.0, spread_bps=50, cap=5.0)
    dear = repriced_window(fin, window, 0.10, spread_bps=50, cap=5.0)
    for name in ("statarb", "xsmom"):
        assert np.allclose(cheap[name], dear[name])
        assert np.allclose(cheap[name], fin[name].loc[window, "net_capped"])
    assert (dear["volcarry"] < cheap["volcarry"]).all()


def test_repricing_at_the_rate_actually_paid_is_the_window_as_financed():
    fin, ff = _financed_for(rate=0.05)
    window = fin["volcarry"].index[:24]
    again = repriced_window(fin, window, 0.05, spread_bps=50, cap=5.0)
    assert np.allclose(again["volcarry"], fin["volcarry"].loc[window, "net_capped"])


def test_as_financed_reproduces_section_13_to_the_decimal():
    fin, ff = _financed_for()
    net = _financed_panels(fin)["net_capped"]
    reference = walk_forward(net, lookback=24)
    run = funding_aware_walk_forward(fin, ff, lookback=24, lag=None)
    assert run["months_traded"] == reference["months_traded"]
    for scheme, s in reference["schemes"].items():
        assert run["schemes"][scheme]["sharpe"] == pytest.approx(s["sharpe"])
        assert run["schemes"][scheme]["mean_weight"] == s["mean_weight"]
    # and the path it keeps averages to the mean weight walk_forward reports
    path = run["schemes"]["max Sharpe"]["weight_path"]["volcarry"]
    assert np.mean(path) == pytest.approx(reference["schemes"]["max Sharpe"]["mean_weight"]["volcarry"], abs=1e-3)


def test_the_lag_only_matters_in_the_months_the_fed_moved():
    fin, ff = _financed_for(rate=0.04)
    runs = [funding_aware_walk_forward(fin, ff, lookback=24, lag=lag) for lag in (None, 1, 0)]
    for scheme in runs[0]["schemes"]:
        assert runs[0]["schemes"][scheme]["sharpe"] == pytest.approx(runs[1]["schemes"][scheme]["sharpe"])
        assert runs[1]["schemes"][scheme]["sharpe"] == pytest.approx(runs[2]["schemes"][scheme]["sharpe"])
    assert runs[1]["rate_priced"] == runs[2]["rate_priced"]


def test_a_dearer_loan_in_the_window_moves_max_sharpe_off_the_sleeve_and_leaves_1n_alone():
    fin, ff = _financed_for()
    cheap = funding_aware_walk_forward(fin, ff, lookback=24, rate_override=0.0)
    dear = funding_aware_walk_forward(fin, ff, lookback=24, rate_override=0.15)
    assert (dear["schemes"]["max Sharpe"]["mean_weight"]["volcarry"]
            < cheap["schemes"]["max Sharpe"]["mean_weight"]["volcarry"])
    # equal weight never reads the window, so it cannot care what it was charged at
    assert dear["schemes"]["equal weight"]["sharpe"] == pytest.approx(cheap["schemes"]["equal weight"]["sharpe"])
    # and the month held is net of what it actually cost, whatever the window was told
    assert all(r == pytest.approx(0.15) for r in dear["rate_priced"])
    assert dear["rate_window_mean"] == cheap["rate_window_mean"]


def test_funding_forecast_study_starts_from_section_13(monkeypatch):
    _patch_sleeves(monkeypatch)
    financed = financing_study(lookback=24, target=0.10, window=36, beta_window=36,
                               settings=((1.0, 1.0),))
    study = funding_forecast_study(lookback=24, target=0.10, window=36, beta_window=36,
                                   rate_sweep=(0.0, 0.15))
    for rule in ("trailing", "implied"):
        base = study["rules"][rule]["runs"]["as_financed"]
        ref = financed["rules"][rule]["walk_forward_net_capped"]
        for scheme, s in ref["schemes"].items():
            assert base["schemes"][scheme]["sharpe"] == pytest.approx(s["sharpe"])
        seen = study["rules"][rule]["first_window"]
        # a flat 4% rate: the window as financed and re-charged at it are the same window
        for name in seen["sleeves_as_financed"]:
            assert seen["sleeves_repriced"][name] == pytest.approx(seen["sleeves_as_financed"][name])
        free, dear = study["rules"][rule]["rate_sweep"]
        assert dear["volcarry_in_first_window"] < free["volcarry_in_first_window"]
        assert dear["max_sharpe_weight_on_volcarry"] <= free["max_sharpe_weight_on_volcarry"]


# ---------- 15. shrink the mean on purpose ----------

from multi_strategy import (bayes_stein_strength, equal_weight_prior,
                            min_variance_return, shrinkage_study, shrunk_mean,
                            shrunk_walk_forward)
from src.optimizer import max_sharpe_weights, min_variance_weights_analytic


def _three_sleeves(n=48, seed=11, means=(-0.006, 0.012, -0.003)):
    rng = np.random.default_rng(seed)
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    draws = rng.normal(means, 0.03, size=(n, 3))
    return pd.DataFrame(draws, index=idx, columns=["statarb", "volcarry", "xsmom"])


def test_min_variance_return_is_what_the_analytic_min_variance_portfolio_earns():
    mu, cov = estimate(_three_sleeves())
    w = min_variance_weights_analytic(cov)
    assert min_variance_return(mu, cov) == pytest.approx(float(w @ mu))


def test_shrunk_mean_is_the_window_at_zero_and_the_prior_at_one():
    mu, cov = estimate(_three_sleeves())
    assert np.allclose(shrunk_mean(mu, cov, 0.0), mu)
    assert np.allclose(shrunk_mean(mu, cov, 0.0, "1/N"), mu)
    flat = shrunk_mean(mu, cov, 1.0, "min variance")
    assert np.allclose(flat, flat[0])
    assert flat[0] == pytest.approx(min_variance_return(mu, cov))
    assert np.allclose(shrunk_mean(mu, cov, 1.0, "1/N"), equal_weight_prior(mu, cov))
    with pytest.raises(ValueError):
        shrunk_mean(mu, cov, 0.5, "market")


def test_max_sharpe_on_the_equal_weight_prior_is_equal_weight_even_when_1n_lost():
    # a window where every sleeve lost: 1/N's own return is negative, and the
    # prior still has to be the one that makes 1/N the answer
    mu, cov = estimate(_three_sleeves(means=(-0.01, -0.004, -0.008)))
    assert float(mu.mean()) < 0
    w = max_sharpe_weights(equal_weight_prior(mu, cov), cov)
    assert np.allclose(w, 1 / 3, atol=1e-3)
    # and the prior's level is the size of the 1/N return, not its sign
    assert float(np.ones(3) / 3 @ equal_weight_prior(mu, cov)) == pytest.approx(abs(float(mu.mean())))


def test_bayes_stein_shrinks_more_when_the_means_agree_and_less_with_more_months():
    mu, cov = estimate(_three_sleeves())
    phi = bayes_stein_strength(mu, cov, 24)
    assert 0.0 < phi < 1.0
    # twice the months is the same evidence twice over: less shrinkage
    assert bayes_stein_strength(mu, cov, 48) < phi
    # means that already sit at the target get pulled all the way
    flat = np.full(3, min_variance_return(mu, cov))
    assert bayes_stein_strength(flat, cov, 24) == pytest.approx(1.0)
    # and means that disagree by a mile are left alone
    assert bayes_stein_strength(mu * 50, cov, 24) < 0.01


def test_shrunk_walk_forward_ends_where_the_textbook_says():
    panel = _three_sleeves(n=60)
    reference = walk_forward(panel, lookback=24)["schemes"]
    none = shrunk_walk_forward(panel, 24, 0.0, "1/N")
    assert none["sharpe"] == pytest.approx(reference["max Sharpe"]["sharpe"])
    assert none["mean_weight"] == reference["max Sharpe"]["mean_weight"]
    to_1n = shrunk_walk_forward(panel, 24, 1.0, "1/N")
    assert to_1n["sharpe"] == pytest.approx(reference["equal weight"]["sharpe"], abs=1e-3)
    to_mv = shrunk_walk_forward(panel, 24, 1.0, "min variance")
    assert to_mv["sharpe"] == pytest.approx(reference["min variance"]["sharpe"], abs=1e-3)
    # the data-driven strength is one number per traded month, inside (0, 1]
    bs = shrunk_walk_forward(panel, 24, "bayes-stein", "min variance")
    assert len(bs["strength_path"]) == bs["months_traded"] == len(panel) - 24
    assert all(0.0 < s <= 1.0 for s in bs["strength_path"])
    assert bs["mean_strength"] == pytest.approx(np.mean(bs["strength_path"]), abs=1e-3)


def test_shrinkage_study_brackets_max_sharpe_and_equal_weight(monkeypatch):
    _patch_sleeves(monkeypatch)
    study = shrinkage_study(lookback=24, target=0.10, window=36, beta_window=36,
                            strengths=(0.0, 0.5, 1.0))
    for rule in ("trailing", "implied"):
        r = study["rules"][rule]
        first, _, last = r["sweeps"]["1/N"]
        assert first["sharpe"] == pytest.approx(r["max_sharpe"])
        assert last["sharpe"] == pytest.approx(r["equal_weight"], abs=1e-3)
        assert r["sweeps"]["min variance"][-1]["sharpe"] == pytest.approx(r["min_variance"], abs=1e-3)
        assert r["best_toward_1n"]["sharpe"] == max(s["sharpe"] for s in r["sweeps"]["1/N"])
        assert r["best_toward_1n"]["over_equal_weight"] == pytest.approx(
            r["best_toward_1n"]["sharpe"] - r["equal_weight"])
        assert set(r["view"]["mean_weight"]) == {"statarb", "volcarry", "xsmom"}
