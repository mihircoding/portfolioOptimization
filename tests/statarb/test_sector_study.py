"""Checks on sector_study.py, which reads the sector on all 4,950 pairs.

The finding is a negative one again - the sector does not rescue the
strategy, and same-sector pairs are the worse ones - so what is checked is
the machinery: that the groups partition the pairs, that the after-entry
measure has the sign it claims, and that the verdicts score what they say.
"""
import numpy as np
import pandas as pd
import pytest

from sleeves.statarb import sector_study
from sleeves.statarb.control import backtest_group


@pytest.fixture
def panel():
    idx = pd.bdate_range("2021-01-01", periods=400)
    rng = np.random.default_rng(1)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(idx), 4)), axis=0)),
        index=idx, columns=["A", "B", "C", "D"])
    scan = pd.DataFrame({"a": ["A", "A", "C"], "b": ["B", "C", "D"],
                         "beta": [1.0, 1.0, 1.0], "pvalue": [0.01, 0.50, 0.03],
                         "same_sector": [True, False, False],
                         "sector_a": ["Tech", "Tech", "Energy"],
                         "sector_b": ["Tech", "Energy", "Utilities"]})
    return scan, prices


def test_the_groups_partition_the_pairs_and_the_book_is_the_groups_book(panel):
    scan, prices = panel
    stats = backtest_group(scan, prices)
    stats["same_sector"] = scan["same_sector"].to_numpy()
    returns = stats.attrs["returns"]
    same = sector_study.group_row(stats, returns, stats["same_sector"])
    cross = sector_study.group_row(stats, returns, ~stats["same_sector"])
    assert same["pairs"] + cross["pairs"] == 3
    assert same["pass_rate"] == 1.0 and cross["pass_rate"] == 0.5
    assert same["book"]["n"] == 1 and cross["book"]["n"] == 2
    # a book of one pair is that pair
    assert same["book"]["sharpe"] == pytest.approx(stats.loc[0, "sharpe"])


def test_by_sector_reads_only_the_same_sector_pairs(panel):
    scan, prices = panel
    stats = backtest_group(scan, prices)
    stats["same_sector"] = scan["same_sector"].to_numpy()
    stats["sector_a"] = scan["sector_a"].to_numpy()
    rows = sector_study.by_sector(stats)
    assert [r["sector"] for r in rows] == ["Tech"]
    assert rows[0]["pairs"] == 1 and rows[0]["pass_rate"] == 1.0


def test_after_entry_is_signed_so_that_continuing_is_positive():
    # a spread that marches up: every entry is on the high side and the
    # spread keeps going, so the signed move is positive at every horizon
    idx = pd.bdate_range("2021-01-01", periods=300)
    up = pd.Series(np.linspace(0, 30, len(idx)) + np.sin(np.arange(len(idx)) / 3.0), index=idx)
    prices = pd.DataFrame({"A": 100 + up, "B": 100.0}, index=idx)
    scan = pd.DataFrame({"a": ["A"], "b": ["B"], "beta": [1.0], "pvalue": [0.01],
                         "same_sector": [True]})
    out = sector_study.after_entry(scan, prices, horizons=(5, 20))
    assert set(out) == {"same sector", "cross sector"}
    same = out["same sector"]
    assert same["5"]["entries"] > 0 and same["20"]["entries"] > 0
    assert same["5"]["mean_move"] > 0 and same["20"]["mean_move"] > same["5"]["mean_move"]
    assert same["20"]["share_continued"] == 1.0
    # nothing was cross-sector, so that side has no entries
    assert out["cross sector"]["5"]["entries"] == 0


def test_the_verdicts_score_the_four_claims():
    book = lambda s: {"sharpe": s, "total_return": 0.0, "n": 1}
    groups = {"same sector": {"mean_sharpe": -0.04, "round_trips": 20.6, "book": book(-0.37)},
              "cross sector": {"mean_sharpe": 0.01, "round_trips": 20.6, "book": book(-0.09)},
              "screened": {"book": book(-0.19)}, "everything": {"book": book(-0.12)}}
    tests = {"same_vs_cross": {"difference": -0.05, "p_value": 0.002}}
    scored = sector_study.verdicts(groups, tests, within=0.04)
    assert [v["held"] for v in scored] == [False, False, False, True]
    tests["same_vs_cross"]["p_value"] = 0.4
    groups["same sector"]["round_trips"] = 25.0
    groups["same sector"]["book"] = book(-0.15)
    scored = sector_study.verdicts(groups, tests, within=0.2)
    assert [v["held"] for v in scored] == [True, True, True, False]
