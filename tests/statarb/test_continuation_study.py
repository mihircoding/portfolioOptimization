"""Checks on continuation_study.py, which trades the sector as a sign.

The result is a near-zero again, so what is checked is the machinery: that
the mirrored trade really is the reverting trade with the sign flipped and
the same costs, that the fixed-horizon trade goes with the move and holds
for as long as it says, that the lumpiness measure orders a jump above a
drift, and that the verdicts score what they claim.
"""
import numpy as np
import pandas as pd
import pytest

from sleeves.statarb import continuation_study as cs
from sleeves.statarb.scan import backtest, generate_positions, rolling_zscore


@pytest.fixture
def panel():
    idx = pd.bdate_range("2021-01-01", periods=400)
    rng = np.random.default_rng(3)
    prices = pd.DataFrame(
        100 * np.exp(np.cumsum(rng.normal(0, 0.01, size=(len(idx), 3)), axis=0)),
        index=idx, columns=["A", "B", "C"])
    scan = pd.DataFrame({"a": ["A", "A"], "b": ["B", "C"], "beta": [1.0, 1.0],
                         "pvalue": [0.01, 0.50], "same_sector": [True, False],
                         "sector_a": ["Tech", "Tech"], "sector_b": ["Tech", "Energy"]})
    return scan, prices


def test_mirror_is_the_trade_with_the_sign_flipped_and_the_same_costs(panel):
    scan, prices = panel
    y, x = prices["A"], prices["B"]
    z = rolling_zscore(y - x)
    pos = generate_positions(z)
    reverting = backtest(y, x, pos)
    mirrored = cs.mirror(reverting)
    flipped = backtest(y, x, -pos)
    pd.testing.assert_series_equal(mirrored["ret"], flipped["ret"], check_names=False)
    pd.testing.assert_series_equal(mirrored["ret_net"], flipped["ret_net"], check_names=False)
    # costs are paid on turnover, and turnover does not know the sign
    costs = reverting["ret"] - reverting["ret_net"]
    assert (costs >= 0).all() and costs.sum() > 0
    assert ((mirrored["ret"] - mirrored["ret_net"]) - costs).abs().max() < 1e-15


def test_hold_positions_goes_with_the_move_and_holds_for_the_horizon():
    idx = pd.bdate_range("2021-01-01", periods=30)
    z = pd.Series(0.0, index=idx)
    z.iloc[5] = 2.5          # fresh crossing on the high side: long the spread
    z.iloc[7] = -3.0         # a crossing inside the trade is ignored
    z.iloc[20] = -2.1        # a fresh crossing after the trade: short the spread
    pos = cs.hold_positions(z, horizon=5)
    assert pos.iloc[:5].tolist() == [0] * 5
    assert pos.iloc[5:10].tolist() == [1] * 5
    assert pos.iloc[10:20].tolist() == [0] * 10
    assert pos.iloc[20:25].tolist() == [-1] * 5
    assert pos.iloc[25:].tolist() == [0] * 5


def test_hold_positions_needs_a_crossing_not_a_level():
    idx = pd.bdate_range("2021-01-01", periods=12)
    z = pd.Series([np.nan, 3.0, 3.0, 3.0, 1.0, 2.5, 2.5, 0.0, 0.0, 0.0, 0.0, 0.0], index=idx)
    pos = cs.hold_positions(z, horizon=2)
    # a NaN before the first 3.0 is not a crossing, so the first trade is at 2.5
    assert pos.tolist() == [0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 0, 0]


def test_run_all_returns_every_trade_and_the_mirror_is_exact(panel):
    scan, prices = panel
    run = cs.run_all(scan, prices)
    assert set(run["gross"]) == {"revert", "mirrored", "hold5", "hold10", "hold20"}
    assert list(run["gross"]["revert"].columns) == ["A/B", "A/C"]
    pd.testing.assert_frame_equal(run["gross"]["mirrored"], -run["gross"]["revert"])
    stats = run["stats"]
    assert stats["same_sector"].tolist() == [True, False]
    for h in cs.HORIZONS:
        assert f"hold{h}_sharpe" in stats and f"hold{h}_return" in stats
    book = cs.book_row(run["gross"]["revert"], run["net"]["revert"], np.array([True, False]))
    assert book["pairs"] == 1 and book["gross_return"] >= book["total_return"]


def test_lumpiness_puts_a_jump_above_a_drift():
    idx = pd.bdate_range("2021-01-01", periods=200)
    drift = pd.Series(0.001, index=idx)
    jump = pd.Series(0.0, index=idx)
    jump.iloc[100] = 0.2
    assert cs.lumpiness(jump)["top5pct_share_of_gains"] == 1.0
    assert cs.lumpiness(drift)["top5pct_share_of_gains"] == pytest.approx(0.05)
    assert cs.lumpiness(jump)["excess_kurtosis"] > cs.lumpiness(drift)["excess_kurtosis"]


def test_the_verdicts_score_the_five_claims():
    row = lambda g, n: {"gross_sharpe": g, "sharpe": n, "gross_return": g / 10, "total_return": n / 10}
    books = {"same sector, mirrored": row(0.14, -0.08), "cross sector, mirrored": row(-0.09, -0.26),
             "cross sector, reverting": row(0.09, -0.09), "screened, reverting (shipped)": row(0.01, -0.19),
             "everything, reverting": row(0.07, -0.12), "sign by sector": row(0.11, -0.11),
             "same sector, hold 20": row(0.17, -0.18), "cross sector, hold 20": row(-0.14, -0.41),
             "cross sector, hold 5": row(-0.05, -0.73), "cross sector, hold 10": row(-0.06, -0.48)}
    lumps = {"same sector, mirrored": {"excess_kurtosis": 4.6, "top5pct_share_of_gains": 0.29},
             "cross sector, reverting": {"excess_kurtosis": 9.1, "top5pct_share_of_gains": 0.37}}
    scored = cs.verdicts(books, lumps)
    assert [v["held"] for v in scored] == [True, False, False, True, False]
