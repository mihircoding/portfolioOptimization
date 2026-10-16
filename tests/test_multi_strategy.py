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
