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
    monkeypatch.setattr("holdout.OUT", str(tmp_path / "holdout.json"))
    monkeypatch.chdir(tmp_path)
    holdout.main()
    out = capsys.readouterr().out
    assert "THE WINDOW THE CUT WAS MADE ON" in out
    written = json.load(open(tmp_path / "holdout.json"))
    assert set(written["window_sweep"]["windows"]) == {"12", "24"}
    assert written["window_verdicts"]["windows"] == [12, 24]
