"""holdout.py: the thirteen months the main study never saw."""

import json
import numpy as np
import pandas as pd
import pytest

import holdout
from holdout import (agreement, capped_panel, frames, holdout_study, split_scores,
                     walk_forward_paths)
from multi_strategy import sleeve_frames, walk_forward


def _patch(monkeypatch, n=120, seed=11, extra=13):
    """Three sleeves on a monthly index, with the vol sleeve running `extra`
    months past the other two's common end - the shape of the real data."""
    idx = pd.date_range("2014-01-31", periods=n + extra, freq="ME")
    rng = np.random.default_rng(seed)
    market = pd.Series(rng.normal(0.008, 0.03, n + extra), index=idx, name="market")
    xsmom = (0.004 - 0.5 * market + rng.normal(0.0, 0.02, n + extra)).rename("xsmom")
    volcarry = pd.Series(rng.normal(0.004, 0.012, n + extra), index=idx, name="ret")
    statarb = pd.Series(rng.normal(0.001, 0.004, n + extra), index=idx, name="statarb")
    days = pd.bdate_range(idx[0] - pd.offsets.MonthBegin(1), idx[-1])
    daily_vol = pd.Series(0.004 * np.sqrt(12 / 252), index=days)
    for module in ("multi_strategy", "holdout"):
        monkeypatch.setattr(f"{module}.xsmom_monthly", lambda: xsmom)
        monkeypatch.setattr(f"{module}.market_monthly", lambda: market)
        monkeypatch.setattr(f"{module}.statarb_monthly", lambda: statarb)
        monkeypatch.setattr(f"{module}.statarb_trailing_vol",
                            lambda **k: (daily_vol * np.sqrt(252)).resample("ME").last().shift(1))
    shipped = volcarry.iloc[:n]
    monkeypatch.setattr("multi_strategy.volcarry_monthly", lambda **k: shipped)
    monkeypatch.setattr("holdout.volcarry_monthly", lambda **k: shipped)
    monkeypatch.setattr("holdout.volcarry_extended", lambda **k: volcarry)
    return volcarry, shipped, idx[n]


def test_the_extended_panel_is_section_11s_panel_on_the_months_both_have(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    ours = capped_panel(frames(volcarry, target=0.10, window=36, beta_window=36))
    theirs = sleeve_frames(target=0.10, window=36, beta_window=36)
    scaled = pd.DataFrame({n: f["scaled"] for n, f in theirs.items()}).dropna()
    raw = pd.DataFrame({n: f["raw"] for n, f in theirs.items()}).loc[scaled.index]
    lev = pd.DataFrame({n: f["leverage"] for n, f in theirs.items()}).loc[scaled.index]
    reference = raw * lev.clip(upper=5.0)
    common = ours.index.intersection(reference.index)
    assert len(common) == len(reference)
    pd.testing.assert_frame_equal(ours.loc[common], reference, check_freq=False)
    # and it runs on past the main study's end
    assert ours.index[-1] > reference.index[-1]
    assert (ours.index > reference.index[-1]).sum() == 13


def test_the_paths_summarise_to_multi_strategys_walk_forward(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    panel = capped_panel(frames(volcarry, target=0.10, window=36, beta_window=36))
    paths, held = walk_forward_paths(panel, 24)
    reference = walk_forward(panel, lookback=24)
    assert len(paths) == reference["months_traded"]
    for name, summary in reference["schemes"].items():
        assert paths[name].mean() * 12 / (paths[name].std(ddof=1) * np.sqrt(12)) == pytest.approx(summary["sharpe"])
    # the weights kept are the ones the return was earned on
    for date, w in held.items():
        month = panel.loc[date].to_numpy()
        for name, weights in w.items():
            assert sum(weights[c] * month[i] for i, c in enumerate(panel.columns)) == pytest.approx(paths.loc[date, name], abs=1e-3)
        assert all(abs(sum(ws.values()) - 1.0) < 1e-3 for ws in w.values())


def test_the_split_is_seen_then_holdout_and_the_two_make_all(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    panel = capped_panel(frames(volcarry, target=0.10, window=36, beta_window=36))
    paths, _ = walk_forward_paths(panel, 24)
    scores = split_scores(paths, holdout_from=f"{first_new:%Y-%m-01}")
    assert scores["seen"]["months"] + scores["holdout"]["months"] == scores["all"]["months"]
    assert scores["holdout"]["months"] == 13
    assert scores["holdout"]["first"] == f"{first_new:%Y-%m}"
    # equal weight against itself is a gap of zero with no interval
    ew = scores["holdout"]["schemes"]["equal weight"]["vs_equal_weight"]
    assert ew["gap"] == 0.0 and ew["low"] == 0.0 and ew["high"] == 0.0
    # and the bootstrap is centred where the Sharpes say
    for name, sc in scores["all"]["schemes"].items():
        assert sc["vs_equal_weight"]["gap"] == pytest.approx(
            sc["sharpe"] - scores["all"]["schemes"]["equal weight"]["sharpe"])


def test_agreement_reads_zero_on_the_same_series_and_counts_a_flipped_sign():
    idx = pd.date_range("2020-01-31", periods=6, freq="ME")
    a = pd.Series([0.01, -0.02, 0.005, 0.0, 0.03, -0.01], index=idx)
    same = agreement(a, a)
    assert same["max_abs_diff"] == 0.0 and same["sign_flips"] == 0 and same["months_compared"] == 6
    b = a.copy()
    b.iloc[1] = 0.001
    moved = agreement(a.iloc[:5], b)   # the longer series has a month the shorter lacks
    assert moved["months_compared"] == 5
    assert moved["sign_flips"] == 1
    assert moved["max_abs_diff"] == pytest.approx(0.021)
    assert moved["max_abs_diff_month"] == "2020-02"


def test_holdout_study_reports_the_sleeve_the_panel_and_both_windows(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    study = holdout_study(lookbacks=(24, 12), holdout_from=f"{first_new:%Y-%m-01}")
    assert study["sleeve"]["new_months"] == 13
    assert study["sleeve"]["shipped_last"] == f"{shipped.index[-1]:%Y-%m}"
    assert study["sleeve"]["extended_last"] == f"{volcarry.index[-1]:%Y-%m}"
    assert study["sleeve_agreement"]["max_abs_diff"] == 0.0
    assert set(study["lookbacks"]) == {"24", "12"}
    for lb in study["lookbacks"].values():
        assert lb["holdout"]["months"] == 13
        assert set(lb["weights_into_worst"]) == {"equal weight", "inverse vol", "risk parity",
                                                 "HRP", "min variance", "max Sharpe"}
        assert lb["weights_into_worst"]["equal weight"]["volcarry"] == pytest.approx(1 / 3, abs=1e-3)
        assert len(lb["volcarry_weight_path"]["max Sharpe"]) == 13


# ---------- the window the cut was made on ----------

from holdout import RISK_BASED, WINDOWS, window_sweep, window_verdicts


def _shock_panel(n=60, shock_at=48, shock=-0.12, seed=3):
    """Three quiet sleeves and one month in which the first loses `shock`:
    the shape of April 2025 in the capped panel, with nothing else in it."""
    idx = pd.date_range("2020-01-31", periods=n, freq="ME")
    rng = np.random.default_rng(seed)
    panel = pd.DataFrame({"statarb": rng.normal(0.003, 0.01, n),
                          "volcarry": rng.normal(0.004, 0.012, n),
                          "xsmom": rng.normal(0.003, 0.011, n)}, index=idx)
    panel.iloc[shock_at, 1] = shock
    return panel, idx[shock_at]


def test_the_sweep_reads_each_window_going_into_the_worst_month_and_out_of_it():
    panel, when = _shock_panel()
    holdout_from = f"{panel.index[44]:%Y-%m-01}"
    sweep = window_sweep(panel, windows=(6, 12, 24), holdout_from=holdout_from)
    assert sweep["worst_month"] == f"{when:%Y-%m}" and sweep["worst_month_return"] == pytest.approx(-0.12)
    # one -12% month against a 1.2% sleeve is on the order of a hundred ordinary months
    assert sweep["worst_month_in_ordinary_months"] > 50
    assert sweep["months_after_worst"] == 60 - 48 - 1
    assert set(sweep["windows"]) == {"6", "12", "24"}
    for w, part in sweep["windows"].items():
        assert part["holdout_months"] == 16 and part["months_traded"] == 60 - int(w)
        assert part["sleeve_vol_multiplier"] > 1.0
        s = part["schemes"]
        assert set(s) == {"equal weight", "inverse vol", "risk parity", "HRP", "min variance", "max Sharpe"}
        # equal weight holds a third throughout, cuts nothing, and is never behind itself
        ew = s["equal weight"]
        assert ew["weight_into_worst"] == pytest.approx(1 / 3, abs=1e-3)
        assert ew["cut"] == pytest.approx(0.0, abs=1e-3) and ew["months_behind_after_worst"] == 0
        assert ew["sharpe_gap"] == 0.0 and ew["shortfall_after_worst"] == pytest.approx(0.0)
        # every risk-based rule cuts the shocked sleeve the month after
        for name in RISK_BASED:
            assert s[name]["cut"] > 0 and s[name]["weight_after_worst"] < s[name]["weight_into_worst"]
            assert s[name]["cut_share"] == pytest.approx(s[name]["cut"] / s[name]["weight_into_worst"])
            assert s[name]["lowest_weight_after"] <= s[name]["weight_after_worst"]
        assert s["max Sharpe"]["worst_month_return"] == pytest.approx(
            float(panel.loc[when] @ np.array([1.0, 0.0, 0.0])), abs=1.0)  # some number: it ran


def test_a_shorter_window_cuts_harder_and_the_vol_multiplier_says_by_how_much():
    panel, when = _shock_panel()
    holdout_from = f"{panel.index[44]:%Y-%m-01}"
    sweep = window_sweep(panel, windows=(6, 12, 24, 36), holdout_from=holdout_from)
    mult = [sweep["windows"][str(w)]["sleeve_vol_multiplier"] for w in (6, 12, 24, 36)]
    # the one month is a bigger share of a shorter window's variance
    assert all(b < a for a, b in zip(mult, mult[1:]))
    # and the share of the position inverse vol keeps after the month is the
    # inverse of that multiplier, before the weights are re-normalized
    for w in (6, 12, 24, 36):
        s = sweep["windows"][str(w)]["schemes"]["inverse vol"]
        kept = s["weight_after_worst"] / s["weight_into_worst"]
        assert kept > 1 / sweep["windows"][str(w)]["sleeve_vol_multiplier"]
    cut_share = [sweep["windows"][str(w)]["schemes"]["inverse vol"]["cut_share"] for w in (6, 12, 24, 36)]
    assert all(b < a for a, b in zip(cut_share, cut_share[1:]))


def test_the_verdicts_score_the_three_predictions_on_the_sweep():
    panel, when = _shock_panel()
    holdout_from = f"{panel.index[44]:%Y-%m-01}"
    sweep = window_sweep(panel, windows=(6, 12, 24), holdout_from=holdout_from)
    v = window_verdicts(sweep)
    assert v["windows"] == [6, 12, 24]
    for name in RISK_BASED:
        assert isinstance(v["cut_shrinks_with_window"][name], bool)
        assert v["cut_longest_over_shortest"][name] == pytest.approx(
            sweep["windows"]["24"]["schemes"][name]["cut"] / sweep["windows"]["6"]["schemes"][name]["cut"])
        # 1/window from 6 to 24 is a quarter, whatever the cuts did
        assert v["one_over_window_would_give"][name] == pytest.approx(0.25)
        assert v["best_window_for"][name] in (6, 12, 24)
        assert len(v["months_behind"][name]) == 3
    assert v["max_sharpe_heaviest_at"] in (6, 12, 24)
    assert v["gap_negative_at_every_window"]["equal weight"] is False  # its gap is zero, not negative
    assert WINDOWS == (6, 12, 18, 24, 36, 42)


def test_holdout_main_writes_the_sweep_beside_the_study(monkeypatch, tmp_path, capsys):
    volcarry, shipped, first_new = _patch(monkeypatch)
    # the sleeve's worst new month has to have months after it to be cut in
    volcarry.iloc[-9] = -0.06
    monkeypatch.setattr("holdout.HOLDOUT_FROM", f"{first_new:%Y-%m-01}")
    monkeypatch.setattr("holdout.LOOKBACKS", (12,))
    monkeypatch.setattr("holdout.WINDOWS", (12, 24))
    monkeypatch.setattr("holdout.SLEEVE_WINDOWS", (12, 24))
    monkeypatch.setattr("holdout.OUT", str(tmp_path / "holdout.json"))
    monkeypatch.chdir(tmp_path)
    holdout.main()
    out = capsys.readouterr().out
    assert "THE WINDOW THE CUT WAS MADE ON" in out
    assert "THE SLEEVE'S OWN WINDOW" in out
    written = json.load(open(tmp_path / "holdout.json"))
    assert set(written["window_sweep"]["windows"]) == {"12", "24"}
    assert written["window_verdicts"]["windows"] == [12, 24]
    assert set(written["sleeve_window_sweep"]["windows"]) == {"12", "24"}
    assert written["sleeve_window_verdicts"]["windows"] == [12, 24]


# ---------- the sleeve's own window ----------

from holdout import SLEEVE_WINDOWS, sleeve_window_sweep, sleeve_window_verdicts
from multi_strategy import trailing_vol, vol_targeted


def test_frames_sizes_the_vol_sleeve_on_its_own_window_and_leaves_momentum_on_the_shipped_one(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    shipped_frames = frames(volcarry)
    own = frames(volcarry, volcarry_window=12)
    monthly = volcarry.copy()
    monthly.index = monthly.index.to_period("M").to_timestamp("M")
    pd.testing.assert_frame_equal(own["volcarry"], vol_targeted(monthly, trailing_vol(monthly, 12), 0.10))
    pd.testing.assert_frame_equal(own["xsmom"], shipped_frames["xsmom"])
    pd.testing.assert_frame_equal(own["statarb"], shipped_frames["statarb"])
    # None is the shipped setting
    pd.testing.assert_frame_equal(frames(volcarry, volcarry_window=None)["volcarry"], shipped_frames["volcarry"])


def test_the_sleeve_sweep_reads_the_sleeves_own_cut_and_every_books_exposure(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    # one bad month inside the holdout, with months after it, on a sleeve
    # the cap never binds on (4% a year against a 10% target is 2.4x)
    volcarry.iloc[-9] = -0.06
    sweep = sleeve_window_sweep(volcarry, windows=(6, 12, 24), lookback=12,
                                holdout_from=f"{first_new:%Y-%m-01}")
    assert set(sweep["windows"]) == {"6", "12", "24"} and sweep["lookback"] == 12
    mult, cuts = [], []
    for w in (6, 12, 24):
        r = sweep["windows"][str(w)]
        assert r["worst_month"] == f"{volcarry.index[-9]:%Y-%m}"
        assert not r["cap_binds_into_worst"] and not r["cap_binds_after_worst"]
        assert r["leverage_into_worst"] == pytest.approx(r["uncapped_leverage_into_worst"])
        # the sleeve's leverage is target over the vol it saw, so its own
        # cut is one minus the inverse of the vol multiplier
        assert r["vol_multiplier"] > 1.0
        assert r["own_cut"] == pytest.approx(1 - 1 / r["vol_multiplier"])
        assert r["leverage_after_worst"] == pytest.approx(0.10 / r["vol_after_worst"])
        mult.append(r["vol_multiplier"])
        cuts.append(r["own_cut"])
        s = r["schemes"]
        assert set(s) == {"equal weight", "inverse vol", "risk parity", "HRP", "min variance", "max Sharpe"}
        for name, sc in s.items():
            assert sc["exposure_into_worst"] == pytest.approx(sc["weight_into_worst"] * r["leverage_into_worst"])
            assert sc["exposure_after_worst"] == pytest.approx(sc["weight_after_worst"] * r["leverage_after_worst"])
        # equal weight cuts no weight, and its exposure still falls by the sleeve's own cut
        ew = s["equal weight"]
        assert ew["weight_cut_share"] == pytest.approx(0.0, abs=1e-3)
        assert ew["exposure_cut_share"] == pytest.approx(r["own_cut"], abs=1e-3)
        # the risk-based books cut weight on top of that
        for name in RISK_BASED:
            assert s[name]["weight_cut_share"] > 0
            assert s[name]["exposure_cut_share"] > r["own_cut"]
    # one month is a bigger share of a shorter window
    assert all(b < a for a, b in zip(mult, mult[1:]))
    assert all(b < a for a, b in zip(cuts, cuts[1:]))


def test_when_the_target_asks_for_more_than_the_cap_the_window_sets_nothing(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    # a sleeve at 1% a year asks for 10x and gets 5: the cap is the leverage
    # going into the bad month and, on a long window, out of it
    quiet = (volcarry * 0.25).rename("ret")
    quiet.iloc[-9] = -0.02
    sweep = sleeve_window_sweep(quiet, windows=(6, 36), lookback=12,
                                holdout_from=f"{first_new:%Y-%m-01}")
    long = sweep["windows"]["36"]
    assert long["cap_binds_into_worst"] and long["cap_binds_after_worst"]
    assert long["leverage_into_worst"] == 5.0 and long["leverage_after_worst"] == 5.0
    assert long["uncapped_leverage_into_worst"] > 5.0
    assert long["own_cut"] == 0.0 and long["predicted_own_cut"] > 0.0
    assert long["cap_binds_share"] > 0.9 and long["cap_binds_share_holdout"] == 1.0
    assert long["schemes"]["equal weight"]["exposure_cut_share"] == pytest.approx(0.0, abs=1e-3)
    # the short window's estimate climbs over the target after the month, and the cap lets go
    short = sweep["windows"]["6"]
    assert short["cap_binds_into_worst"] and not short["cap_binds_after_worst"]
    assert short["own_cut"] > 0.0


def test_the_sleeve_verdicts_score_the_cap_first_and_then_the_four_predictions(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    volcarry.iloc[-9] = -0.06
    sweep = sleeve_window_sweep(volcarry, windows=(6, 12, 24), lookback=12,
                                holdout_from=f"{first_new:%Y-%m-01}")
    v = sleeve_window_verdicts(sweep)
    assert v["windows"] == [6, 12, 24]
    assert v["cap_binds_into_worst"] == [False, False, False]
    assert len(v["own_cut"]) == len(v["predicted_own_cut"]) == 3
    assert v["own_cut_shrinks_with_window"] is True
    assert all(v["equal_weight_exposure_cut_is_own_cut"])
    assert all(v["risk_based_exposure_cut_above_own"].values())
    assert v["sleeve_april_worst_at"] in (6, 12, 24) and v["sleeve_best_window"] in (6, 12, 24)
    assert set(v["weight_cut_deepest_at"]) == set(RISK_BASED)
    assert v["exposure_cut_at_shipped"] is None   # 36 is not in this sweep
    assert SLEEVE_WINDOWS == (6, 12, 24, 36, 60)


# ---------- section 22: unpin the dial ----------

from holdout import UNPINNED, print_unpinned, unpinned_sweep, unpinned_verdicts


def test_the_cap_and_the_target_reach_the_panel_and_halving_both_is_the_same_sleeve(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    # a sleeve at 1.3% a year asks for about 8x: capped at 5 it is the cap,
    # at 10 it is the ask, and a 5% target at 5x is the 10% target at 10x
    quiet = (volcarry * 0.33).rename("ret")
    quiet.iloc[-9] = -0.02
    since = f"{first_new:%Y-%m-01}"
    capped = sleeve_window_sweep(quiet, windows=(36,), lookback=12, holdout_from=since)["windows"]["36"]
    loose = sleeve_window_sweep(quiet, windows=(36,), lookback=12, holdout_from=since,
                                cap=10.0)["windows"]["36"]
    half = sleeve_window_sweep(quiet, windows=(36,), lookback=12, holdout_from=since,
                               cap=5.0, target=0.05)["windows"]["36"]
    assert capped["leverage_into_worst"] == 5.0 and capped["own_cut"] == 0.0
    assert 5.0 < loose["uncapped_leverage_into_worst"] < 10.0
    assert loose["leverage_into_worst"] == pytest.approx(loose["uncapped_leverage_into_worst"])
    assert loose["own_cut"] == pytest.approx(1 - 1 / loose["vol_multiplier"])
    # the scaled April is the raw month times the leverage the cap allowed,
    # so the panel saw the cap and not only the report
    assert capped["sleeve"]["worst_month_return"] == pytest.approx(5.0 * capped["sleeve"]["raw_worst_month_return"])
    assert loose["sleeve"]["worst_month_return"] == pytest.approx(
        loose["leverage_into_worst"] * loose["sleeve"]["raw_worst_month_return"])
    assert half["leverage_into_worst"] == pytest.approx(loose["leverage_into_worst"] / 2)
    assert half["own_cut"] == pytest.approx(loose["own_cut"])
    assert half["sleeve"]["sharpe"] == pytest.approx(loose["sleeve"]["sharpe"])
    assert half["cap_binds_share"] == pytest.approx(loose["cap_binds_share"])


def test_the_unpinned_sweep_runs_every_setting_and_the_verdicts_read_them_against_the_first(monkeypatch, capsys):
    volcarry, shipped, first_new = _patch(monkeypatch)
    volcarry.iloc[-9] = -0.06
    settings = (("10% / 5x", 0.10, 5.0), ("10% / 10x", 0.10, 10.0), ("5% / 5x", 0.05, 5.0))
    sweep = unpinned_sweep(volcarry, settings, windows=(6, 12), lookback=12,
                           holdout_from=f"{first_new:%Y-%m-01}")
    assert list(sweep["settings"]) == [s[0] for s in settings]
    for label, target, cap in settings:
        s = sweep["settings"][label]
        assert s["cap"] == cap and s["target"] == target
        assert set(s["windows"]) == {"6", "12"}
    v = unpinned_verdicts(sweep)
    assert v["settings"] == [s[0] for s in settings] and v["windows"] == [6, 12]
    # this sleeve never hits the cap, so every window is cap-free and the
    # own cut is the ratio's to score, at every setting
    for label in v["settings"]:
        assert v["cap_free_windows"][label] == [True, True]
        assert len(v["ratio_within_five_points_where_cap_free"][label]) == 2
        assert v["own_cut"][label] == pytest.approx(v["own_cut"][v["settings"][0]])
    # the first setting has nothing to be compared against
    assert set(v["allocator_cuts_deeper_than_shipped"]) == {"10% / 10x", "5% / 5x"}
    for d in v["allocator_cuts_deeper_than_shipped"].values():
        assert set(d) == set(RISK_BASED)
    print_unpinned(sweep, v)
    out = capsys.readouterr().out
    assert "UNPIN THE DIAL" in out and "the predictions:" in out
    assert len(UNPINNED) == 5 and UNPINNED[0][1:] == (0.10, 5.0)


# ---------- section 23: keep the mean ----------

from holdout import (ESTIMATOR_ORDER, KEPT_MEAN, estimator_sweep, estimator_verdicts,
                     print_estimator)
from multi_strategy import ewma_vol, trailing_rms_vol


def test_the_rms_keeps_the_mean_the_sample_std_throws_away():
    idx = pd.date_range("2015-01-31", periods=48, freq="ME")
    # a sleeve that makes the same 0.6% every month has no sample std at
    # all, and a root mean square of exactly that 0.6%
    steady = pd.Series(0.006, index=idx)
    assert trailing_vol(steady, 12).dropna().max() == pytest.approx(0.0)
    assert trailing_rms_vol(steady, 12).dropna().iloc[0] == pytest.approx(0.006 * np.sqrt(12))
    rng = np.random.default_rng(3)
    noisy = pd.Series(rng.normal(0.004, 0.012, 48), index=idx)
    std, rms = trailing_vol(noisy, 12), trailing_rms_vol(noisy, 12)
    # same window, same shift: NaN for the same months, nothing from the month sized
    assert std.isna().sum() == rms.isna().sum() == 12
    bumped = noisy.copy()
    bumped.iloc[-1] += 1.0
    pd.testing.assert_series_equal(trailing_rms_vol(bumped, 12), rms)
    # and the gap between them is the mean, to the digit: the mean square is
    # the sample variance (less its n/(n-1)) plus the mean squared, so the
    # rms is not always the larger of the two - it is when the mean is
    window = noisy.iloc[:12]
    assert rms.iloc[12] ** 2 == pytest.approx(12 * (11 / 12 * window.var(ddof=1) + window.mean() ** 2))
    assert std.iloc[12] ** 2 == pytest.approx(12 * window.var(ddof=1))


def test_the_ewma_weights_the_newest_month_one_over_the_window_and_starts_where_the_window_does():
    idx = pd.date_range("2010-01-31", periods=200, freq="ME")
    rng = np.random.default_rng(5)
    x = pd.Series(rng.normal(0.004, 0.010, 200), index=idx)
    for window in (6, 36):
        ew = ewma_vol(x, window)
        assert ew.isna().sum() == window
        # one shock k ordinary months wide moves the estimate by
        # sqrt((w - 1 + k) / w), which is the ratio with lam = 1 - 1/w
        shocked = x.copy()
        shocked.iloc[150] = -0.05
        before = ewma_vol(shocked, window).iloc[150]
        after = ewma_vol(shocked, window).iloc[151]
        k = 0.05 ** 2 / (before ** 2 / 12)
        assert after / before == pytest.approx(np.sqrt((window - 1 + k) / window))
        # seeded with the first window's mean square, so it starts as the rms
        assert ew.iloc[window] == pytest.approx(trailing_rms_vol(x, window).iloc[window])
    # the window drops the shock after w months, the ewma only decays it
    shocked = x.copy()
    shocked.iloc[150] = -0.05
    gone = trailing_rms_vol(shocked, 6).iloc[157]
    kept = ewma_vol(shocked, 6).iloc[157]
    assert gone == pytest.approx(trailing_rms_vol(x, 6).iloc[157])
    assert kept > ewma_vol(x, 6).iloc[157]


def test_frames_sizes_the_vol_sleeve_on_the_estimator_asked_for_and_nothing_else_moves(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    monthly = volcarry.copy()
    monthly.index = monthly.index.to_period("M").to_timestamp("M")
    shipped_frames = frames(volcarry)
    for name, fn in (("rms", trailing_rms_vol), ("ewma", ewma_vol)):
        fr = frames(volcarry, volcarry_window=12, estimator=name)
        pd.testing.assert_frame_equal(fr["volcarry"], vol_targeted(monthly, fn(monthly, 12), 0.10))
        pd.testing.assert_frame_equal(fr["xsmom"], shipped_frames["xsmom"])
        pd.testing.assert_frame_equal(fr["statarb"], shipped_frames["statarb"])
    pd.testing.assert_frame_equal(frames(volcarry, estimator="std")["volcarry"], shipped_frames["volcarry"])
    with pytest.raises(KeyError):
        frames(volcarry, estimator="garch")


def test_the_sweep_names_the_month_the_window_drops_and_the_exact_multiplier_is_the_measured_one(monkeypatch):
    volcarry, shipped, first_new = _patch(monkeypatch)
    volcarry.iloc[-9] = -0.06
    since = f"{first_new:%Y-%m-01}"
    worst = volcarry.index[-9]
    for name in ("rms", "ewma"):
        sweep = sleeve_window_sweep(volcarry, windows=(6, 24), lookback=12, holdout_from=since,
                                    cap=10.0, estimator=name)
        assert sweep["estimator"] == name
        for w in (6, 24):
            r = sweep["windows"][str(w)]
            assert r["month_dropped"] == f"{volcarry.index[-9 - w]:%Y-%m}"
            assert r["month_dropped_return"] == pytest.approx(volcarry.iloc[-9 - w])
            assert not r["cap_binds_into_worst"] and not r["cap_binds_after_worst"]
            # the formula with the month going out (rms) or on the estimator's
            # own k (ewma) is the multiplier, not an approximation to it
            assert r["predicted_multiplier_exact"] == pytest.approx(r["vol_multiplier"], rel=1e-6)
            assert r["own_cut"] == pytest.approx(1 - 1 / r["predicted_multiplier_exact"], rel=1e-6)
            assert r["worst_in_ordinary_months_own_estimate"] == pytest.approx(
                volcarry.loc[worst] ** 2 / (r["vol_into_worst"] ** 2 / 12))
    # the shipped estimator has no exact form; it carries section 22's correction instead
    std = sleeve_window_sweep(volcarry, windows=(6,), lookback=12, holdout_from=since,
                              cap=10.0)["windows"]["6"]
    assert std["predicted_multiplier_exact"] is None
    assert std["predicted_multiplier_demeaned"] == pytest.approx(
        np.sqrt((6 - 1 + std["worst_in_ordinary_months_demeaned"]) / 6))


def test_the_estimator_sweep_runs_every_estimator_at_every_setting_and_scores_the_five(monkeypatch, capsys):
    volcarry, shipped, first_new = _patch(monkeypatch)
    volcarry.iloc[-9] = -0.06
    settings = (("10% / 10x", 0.10, 10.0), ("5% / 10x", 0.05, 10.0))
    sweep = estimator_sweep(volcarry, settings, windows=(6, 12), lookback=12,
                            holdout_from=f"{first_new:%Y-%m-01}")
    assert list(sweep["settings"]) == ["10% / 10x", "5% / 10x"]
    for label in sweep["settings"]:
        assert list(sweep["settings"][label]) == list(ESTIMATOR_ORDER)
        for est, s in sweep["settings"][label].items():
            assert s["estimator"] == est and set(s["windows"]) == {"6", "12"}
    v = estimator_verdicts(sweep)
    assert v["settings"] == ["10% / 10x", "5% / 10x"]
    for label, vs in v["by_setting"].items():
        assert vs["windows"] == [6, 12]
        assert set(vs["own_cut"]) == set(ESTIMATOR_ORDER)
        assert len(vs["rms_within_three_points_of_ratio"]) == 2
        assert set(vs["hrp_within_five_points_of_std"]) == {"rms", "ewma"}
        assert set(vs["predicted_multiplier_exact"]) == {"rms", "ewma"}
        assert isinstance(vs["rms_leverage_lower_everywhere"], bool)
    # halving the target at the same cap is the same sleeve where the cap is off
    a = v["by_setting"]["10% / 10x"]
    b = v["by_setting"]["5% / 10x"]
    for est in ESTIMATOR_ORDER:
        assert a["own_cut"][est] == pytest.approx(b["own_cut"][est])
    print_estimator(sweep, v)
    out = capsys.readouterr().out
    assert "KEEP THE MEAN" in out and "the predictions:" in out
    assert [s[0] for s in KEPT_MEAN] == ["10% / 10x", "5% / 10x"]
