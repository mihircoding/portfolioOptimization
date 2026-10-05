"""Trade the pairs the screen threw away, and see if they do any worse.

Usage:  python control.py        (reads results/, no network)

Every result in this project so far describes the 930 pairs that passed the
cointegration test. The obvious question has never been answered directly:
compared to what? A screen that keeps 930 of 4,950 pairs is only worth running
if the 4,020 it rejected would have done worse. Nobody ever checks, because
checking means backtesting the rejects, and the rejects are supposed to be the
ones you don't trade.

So: same trading window, same hedge ratios from the same formation regression,
same 60-day z-score, same 2.0/0.5 thresholds, same 5bps a side. The only
difference between the two groups is which side of p = 0.05 they landed on.

Two ways of asking it, because they answer different things:

  - Per pair. Is the distribution of out-of-sample Sharpe different between the
    groups? Tested with a permutation test rather than a t-test: pair returns
    are heavily overlapping (each ticker appears in ~99 pairs), so the usual
    standard errors are wrong in a direction that flatters the screen.

  - As a book. Equal-weight everything that passed against equal-weight
    everything that didn't. This is the version a trader would actually care
    about, and it is not the same question - a group can have a worse average
    pair and a better book if its pairs are less correlated with each other.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scan import ZSCORE_WINDOW, run_pair, sharpe

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SITE_OUT = ROOT.parent / "docs" / "control.js"
N_PERMUTATIONS = 10_000


def backtest_group(scan: pd.DataFrame, trading: pd.DataFrame) -> pd.DataFrame:
    """Run every pair in `scan` through the same rules the survivors got."""
    rows, curves = [], {}
    for row in scan.itertuples():
        y, x = trading[row.a], trading[row.b]
        result, _, _, stats = run_pair(y, x, row.beta)
        rows.append({"a": row.a, "b": row.b, "pvalue": row.pvalue,
                     "passed": row.pvalue <= 0.05, **stats})
        curves[f"{row.a}/{row.b}"] = result["ret_net"]
    out = pd.DataFrame(rows)
    out.attrs["returns"] = pd.DataFrame(curves)
    return out


def permutation_test(a: np.ndarray, b: np.ndarray, n: int = N_PERMUTATIONS,
                     seed: int = 0) -> dict:
    """P(difference in means this large by chance), labels shuffled.

    Makes no assumption about the shape of either distribution, which matters
    here: pair Sharpes are fat-tailed and the two groups are not independent
    samples of anything - they share tickers.
    """
    rng = np.random.default_rng(seed)
    pooled = np.concatenate([a, b])
    observed = a.mean() - b.mean()
    n_a = len(a)
    count = 0
    for _ in range(n):
        rng.shuffle(pooled)
        if abs(pooled[:n_a].mean() - pooled[n_a:].mean()) >= abs(observed):
            count += 1
    return {"difference": float(observed), "p_value": (count + 1) / (n + 1)}


def book_stats(returns: pd.DataFrame) -> dict:
    """Equal-weight book of a set of pairs, rebalanced daily."""
    if returns.empty:
        return {"total_return": 0.0, "sharpe": 0.0, "n": 0}
    daily = returns.mean(axis=1)
    equity = (1.0 + daily).cumprod()
    return {"total_return": float(equity.iloc[-1] - 1.0),
            "sharpe": sharpe(daily), "n": returns.shape[1]}


def decile_table(stats: pd.DataFrame) -> pd.DataFrame:
    """Out-of-sample Sharpe by formation p-value decile, across ALL pairs.

    The existing rank-correlation check only ever saw p-values below 0.05,
    which is a tenth of the range the test can produce. If the screen ranks
    anything, it should show up over the full range or nowhere.
    """
    d = stats.copy()
    d["decile"] = pd.qcut(d["pvalue"], 10, labels=False, duplicates="drop")
    g = d.groupby("decile")
    return pd.DataFrame({
        "pvalue_lo": g["pvalue"].min(), "pvalue_hi": g["pvalue"].max(),
        "pairs": g.size(), "mean_sharpe": g["sharpe"].mean(),
        "median_return": g["total_return"].median(),
        "share_profitable": g["total_return"].apply(lambda s: float((s > 0).mean())),
    }).reset_index()


def main() -> None:
    scan = pd.read_parquet(RESULTS / "scan.parquet")
    trading = pd.read_parquet(RESULTS / "trading.parquet")

    passed = scan[scan["pvalue"] <= 0.05]
    rejected = scan[scan["pvalue"] > 0.05]
    print(f"{len(scan):,} pairs tested on formation data: {len(passed)} passed "
          f"at 5%, {len(rejected):,} rejected.")
    print(f"Trading window {trading.index[0].date()} -> {trading.index[-1].date()}, "
          f"{len(trading):,} days. Same rules for both groups: {ZSCORE_WINDOW}-day "
          f"z-score,\nentry 2.0, exit 0.5, 5bps a side, hedge ratio from formation.\n")

    print("Backtesting every pair, including the rejects (the slow part)...")
    stats = backtest_group(scan, trading)
    returns = stats.attrs["returns"]

    for label, mask in (("passed the screen", stats["passed"]),
                        ("rejected", ~stats["passed"])):
        s = stats[mask]
        print(f"\n{label}: {len(s):,} pairs")
        print(f"  mean Sharpe          {s['sharpe'].mean():>8.3f}")
        print(f"  median Sharpe        {s['sharpe'].median():>8.3f}")
        print(f"  median total return  {s['total_return'].median():>8.2%}")
        print(f"  share profitable     {(s['total_return'] > 0).mean():>8.1%}")
        print(f"  mean round trips     {s['n_round_trips'].mean():>8.1f}")

    test = permutation_test(stats.loc[stats["passed"], "sharpe"].to_numpy(),
                            stats.loc[~stats["passed"], "sharpe"].to_numpy())
    print(f"\nDifference in mean Sharpe (passed - rejected): {test['difference']:+.4f}")
    print(f"Permutation test over {N_PERMUTATIONS:,} relabellings: "
          f"p = {test['p_value']:.3f}")

    print("\nAs an equal-weight book:")
    print(f"{'book':<22} {'pairs':>7} {'total return':>14} {'sharpe':>8}")
    for label, mask in (("screened (passed)", stats["passed"].to_numpy()),
                        ("rejected", ~stats["passed"].to_numpy()),
                        ("everything", np.ones(len(stats), dtype=bool))):
        b = book_stats(returns.loc[:, mask])
        print(f"{label:<22} {b['n']:>7,} {b['total_return']:>14.2%} "
              f"{b['sharpe']:>8.2f}")

    print("\nOut-of-sample Sharpe by formation p-value decile, all 4,950 pairs:")
    table = decile_table(stats)
    print(f"{'decile':>7} {'p-value range':>22} {'pairs':>7} {'mean sharpe':>12} "
          f"{'median ret':>11} {'profitable':>11}")
    for _, r in table.iterrows():
        print(f"{int(r['decile']) + 1:>7} {r['pvalue_lo']:>10.4f}-{r['pvalue_hi']:<11.4f} "
              f"{int(r['pairs']):>7,} {r['mean_sharpe']:>12.3f} "
              f"{r['median_return']:>11.2%} {r['share_profitable']:>11.1%}")

    payload = {
        "n_tested": int(len(scan)), "n_passed": int(len(passed)),
        "n_rejected": int(len(rejected)),
        "window": [str(trading.index[0].date()), str(trading.index[-1].date())],
        "groups": {
            label: {
                "pairs": int(mask.sum()),
                "mean_sharpe": round(float(stats.loc[mask, "sharpe"].mean()), 4),
                "median_sharpe": round(float(stats.loc[mask, "sharpe"].median()), 4),
                "median_return": round(float(stats.loc[mask, "total_return"].median()), 5),
                "share_profitable": round(float(
                    (stats.loc[mask, "total_return"] > 0).mean()), 4),
                "round_trips": round(float(stats.loc[mask, "n_round_trips"].mean()), 2),
                "book": {k: (round(v, 5) if isinstance(v, float) else v)
                         for k, v in book_stats(returns.loc[:, mask.to_numpy()]).items()},
            }
            for label, mask in (("passed", stats["passed"]),
                                ("rejected", ~stats["passed"]))
        },
        "permutation": {k: round(float(v), 5) for k, v in test.items()},
        "deciles": [{k: (round(float(v), 5) if isinstance(v, float) else int(v))
                     for k, v in row.items()} for row in table.to_dict("records")],
        # Sharpe histograms, same bins for both groups so the chart overlays
        "histogram": None,
    }
    edges = np.histogram_bin_edges(stats["sharpe"], bins=36,
                                   range=(-1.5, 1.5))
    payload["histogram"] = {
        "edges": [round(float(e), 3) for e in edges],
        "passed": [int(n) for n in np.histogram(
            stats.loc[stats["passed"], "sharpe"], bins=edges)[0]],
        "rejected": [int(n) for n in np.histogram(
            stats.loc[~stats["passed"], "sharpe"], bins=edges)[0]],
    }

    rho = stats["pvalue"].corr(stats["sharpe"], method="spearman")
    payload["spearman"] = round(float(rho), 4)
    payload["book_all"] = {k: (round(v, 5) if isinstance(v, float) else v)
                           for k, v in book_stats(returns).items()}

    if SITE_OUT.parent.exists():
        SITE_OUT.write_text(
            "window.CONTROL = " + json.dumps(payload, separators=(",", ":")) + ";\n",
            encoding="utf-8")
        print(f"\nwrote {SITE_OUT} ({SITE_OUT.stat().st_size / 1024:.0f} KB)")

    print(f"\nSpearman correlation, formation p-value vs out-of-sample Sharpe, "
          f"all pairs: {rho:+.3f}")
    print("A screen that works would make this strongly negative: lower p-value,")
    print("higher Sharpe. Over the full range of the test, not just the tenth of it")
    print("the survivors occupy.")


if __name__ == "__main__":
    main()
