"""Pick the universe by economic relationship instead of by the screen.

Usage:  python -m sleeves.statarb.sector_study        (reads results/, no network)

Every write-up of this sleeve ends the same way: the screen does not rank,
the control group did as well as the survivors, and the remaining candidates
are the data and the universe rather than the method - "a universe chosen by
economic relationship rather than by running every pair through a test". The
one proxy for an economic relationship this project has is GICS sector, and
so far it has only been read on the 930 survivors, where the 119 same-sector
pairs did a little worse than the 811 cross-sector ones. That is the screen's
sample, not the sector's. This reads the sector on all 4,950 pairs, with the
same backtest control.py ran on the rejects, so that "trade same-sector pairs
and skip the test" is a book that can be compared to the screened one.

Four predictions, written before the backtest ran (the pass rates were
visible while sizing the run and are reported, not predicted):

  1. Unscreened, same-sector pairs are no better than cross-sector ones:
     the gap in mean out-of-sample Sharpe over all 4,950 pairs is inside
     permutation noise, p > 0.10.
  2. Same-sector pairs round-trip more often - two names with a common
     factor have a spread that crosses its mean more - and the extra trips
     cost more than they make at 5bps a side, so whatever gap there is in
     Sharpe is at or below zero.
  3. The same-sector book with no screen does at least as well as the
     screened 930, and neither beats the everything book by more than
     noise: the sector is not a worse filter than the test, and no filter
     here is a good one.
  4. Inside the 585 same-sector pairs the p-value still does not rank:
     Spearman of formation p-value against out-of-sample Sharpe within
     +/-0.05 of zero, as it was over the whole universe.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sleeves.statarb.control import backtest_group, book_stats, permutation_test
from sleeves.statarb.scan import ZSCORE_WINDOW, rolling_zscore

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "sector_study.json"


def group_row(stats: pd.DataFrame, returns: pd.DataFrame, mask: pd.Series) -> dict:
    s = stats[mask]
    return {"pairs": int(len(s)),
            "pass_rate": float(s["passed"].mean()),
            "mean_sharpe": float(s["sharpe"].mean()),
            "median_sharpe": float(s["sharpe"].median()),
            "share_profitable": float((s["total_return"] > 0).mean()),
            "median_return": float(s["total_return"].median()),
            "round_trips": float(s["n_round_trips"].mean()),
            "book": book_stats(returns.loc[:, mask.to_numpy()])}


def by_sector(stats: pd.DataFrame) -> list[dict]:
    """The same-sector pairs, sector by sector: how often the screen passed
    them and how they traded. Small cells are small cells; n is printed."""
    rows = []
    same = stats[stats["same_sector"]]
    for sector, g in same.groupby("sector_a"):
        rows.append({"sector": sector, "pairs": int(len(g)),
                     "pass_rate": float(g["passed"].mean()),
                     "mean_sharpe": float(g["sharpe"].mean()),
                     "share_profitable": float((g["total_return"] > 0).mean()),
                     "round_trips": float(g["n_round_trips"].mean())})
    return sorted(rows, key=lambda r: -r["pairs"])


def after_entry(scan: pd.DataFrame, trading: pd.DataFrame, horizons=(5, 10, 20),
                entry: float = 2.0) -> dict:
    """What the spread does after the signal fires, by group. Post hoc: this
    was written after the backtest came out, to find out WHY.

    For every fresh crossing of |z| = 2 in the trading window, the spread's
    move over the next h days, signed so that positive means it kept going
    the way it had gone, in units of the 60-day spread std the z-score was
    built on. A reverter is paid when this is negative. It is the trade's
    edge measured directly, without the exit rule or the costs in the way.
    """
    moves = {True: {h: [] for h in horizons}, False: {h: [] for h in horizons}}
    for row in scan.itertuples():
        spread = trading[row.a] - row.beta * trading[row.b]
        z = rolling_zscore(spread, ZSCORE_WINDOW)
        sd = spread.rolling(ZSCORE_WINDOW).std()
        fresh = (z.abs() >= entry) & (z.abs().shift(1) < entry)
        idx = np.where(fresh.to_numpy())[0]
        s, zz, sdv = spread.to_numpy(), z.to_numpy(), sd.to_numpy()
        for h in horizons:
            ok = idx[idx + h < len(s)]
            moves[row.same_sector][h].extend(
                (np.sign(zz[ok]) * (s[ok + h] - s[ok]) / sdv[ok]).tolist())
    return {("same sector" if same else "cross sector"): {
                str(h): {"entries": len(v), "mean_move": float(np.mean(v)),
                         "share_continued": float(np.mean(np.asarray(v) > 0))}
                for h, v in by_h.items()}
            for same, by_h in moves.items()}


def verdicts(groups: dict, tests: dict, within: float) -> list[dict]:
    same, cross = groups["same sector"], groups["cross sector"]
    screened, everything = groups["screened"], groups["everything"]
    gap = same["mean_sharpe"] - cross["mean_sharpe"]
    return [
        {"claim": "unscreened, same-sector is no better than cross-sector: p > 0.10",
         "measured": f"gap {gap:+.3f}, p = {tests['same_vs_cross']['p_value']:.3f}",
         "held": tests["same_vs_cross"]["p_value"] > 0.10},
        {"claim": "same-sector pairs round-trip more and the gap in Sharpe is at or below zero",
         "measured": f"round trips {same['round_trips']:.1f} vs {cross['round_trips']:.1f}, gap {gap:+.3f}",
         "held": same["round_trips"] > cross["round_trips"] and gap <= 0.0},
        {"claim": "the same-sector book is at least the screened book, and neither beats everything by more than noise",
         "measured": f"books: same-sector {same['book']['sharpe']:+.2f}, screened {screened['book']['sharpe']:+.2f}, "
                     f"everything {everything['book']['sharpe']:+.2f}",
         "held": (same["book"]["sharpe"] >= screened["book"]["sharpe"]
                  and abs(same["book"]["sharpe"] - everything["book"]["sharpe"]) < 0.25
                  and abs(screened["book"]["sharpe"] - everything["book"]["sharpe"]) < 0.25)},
        {"claim": "inside the same-sector pairs the p-value still does not rank: |rho| < 0.05",
         "measured": f"Spearman {within:+.3f}",
         "held": abs(within) < 0.05},
    ]


def main() -> None:
    scan = pd.read_parquet(RESULTS / "scan.parquet")
    trading = pd.read_parquet(RESULTS / "trading.parquet")
    print(f"{len(scan):,} pairs, {int(scan['same_sector'].sum())} of them in the same GICS sector. "
          f"Backtesting every pair with the control group's rules...")
    stats = backtest_group(scan, trading)
    returns = stats.attrs["returns"]
    stats["same_sector"] = scan["same_sector"].to_numpy()
    stats["sector_a"] = scan["sector_a"].to_numpy()

    masks = {
        "same sector": stats["same_sector"],
        "cross sector": ~stats["same_sector"],
        "same sector, passed": stats["same_sector"] & stats["passed"],
        "same sector, rejected": stats["same_sector"] & ~stats["passed"],
        "cross sector, passed": ~stats["same_sector"] & stats["passed"],
        "cross sector, rejected": ~stats["same_sector"] & ~stats["passed"],
        "screened": stats["passed"],
        "everything": pd.Series(True, index=stats.index),
    }
    groups = {label: group_row(stats, returns, mask) for label, mask in masks.items()}

    print(f"\n{'group':<24} {'pairs':>6} {'passed':>7} {'mean Sh':>8} {'median':>7} {'profit':>7} "
          f"{'trips':>6} {'book Sh':>8} {'book ret':>9}")
    for label, g in groups.items():
        print(f"{label:<24} {g['pairs']:>6,} {g['pass_rate']:>7.1%} {g['mean_sharpe']:>+8.3f} "
              f"{g['median_sharpe']:>+7.3f} {g['share_profitable']:>7.1%} {g['round_trips']:>6.1f} "
              f"{g['book']['sharpe']:>+8.2f} {g['book']['total_return']:>+9.2%}")

    tests = {
        "same_vs_cross": permutation_test(stats.loc[masks["same sector"], "sharpe"].to_numpy(),
                                          stats.loc[masks["cross sector"], "sharpe"].to_numpy()),
        "same_passed_vs_rejected": permutation_test(
            stats.loc[masks["same sector, passed"], "sharpe"].to_numpy(),
            stats.loc[masks["same sector, rejected"], "sharpe"].to_numpy()),
    }
    print(f"\nSame against cross, all pairs: {tests['same_vs_cross']['difference']:+.4f} in mean Sharpe, "
          f"p = {tests['same_vs_cross']['p_value']:.3f}")
    print(f"Inside same-sector, passed against rejected: {tests['same_passed_vs_rejected']['difference']:+.4f}, "
          f"p = {tests['same_passed_vs_rejected']['p_value']:.3f}")

    same = stats[stats["same_sector"]]
    within = float(same["pvalue"].corr(same["sharpe"], method="spearman"))
    overall = float(stats["pvalue"].corr(stats["sharpe"], method="spearman"))
    # the sector as a ranker, on the same footing as the p-value: a rank
    # correlation of the dummy with the Sharpe
    sector_rho = float(stats["same_sector"].astype(float).corr(stats["sharpe"], method="spearman"))
    print(f"\nRank correlation of out-of-sample Sharpe with: formation p-value {overall:+.3f} (all), "
          f"{within:+.3f} (same-sector only); same-sector dummy {sector_rho:+.3f}")

    sectors = by_sector(stats)
    print(f"\n{'sector':<24} {'pairs':>6} {'passed':>7} {'mean Sh':>8} {'profit':>7} {'trips':>6}")
    for r in sectors:
        print(f"{r['sector']:<24} {r['pairs']:>6} {r['pass_rate']:>7.1%} {r['mean_sharpe']:>+8.3f} "
              f"{r['share_profitable']:>7.1%} {r['round_trips']:>6.1f}")

    entries = after_entry(scan, trading)
    print("\nAfter a fresh |z| = 2 entry, the spread's move over the next h days, in 60-day sigmas, "
          "signed so positive = kept going (post hoc):")
    print(f"{'group':<14} " + " ".join(f"{'h=' + h:>8} {'continued':>9}" for h in entries["cross sector"]))
    for label, by_h in entries.items():
        print(f"{label:<14} " + " ".join(f"{m['mean_move']:>+8.3f} {m['share_continued']:>9.1%}"
                                         for m in by_h.values()))

    scored = verdicts(groups, tests, within)
    print("\nPredictions, scored:")
    for v in scored:
        print(f"  [{'held' if v['held'] else 'FAILED'}] {v['claim']}\n         measured: {v['measured']}")

    payload = {"groups": groups, "tests": tests, "spearman": {"pvalue_all": overall,
               "pvalue_same_sector": within, "same_sector_dummy": sector_rho},
               "sectors": sectors, "after_entry": entries, "verdicts": scored}
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
