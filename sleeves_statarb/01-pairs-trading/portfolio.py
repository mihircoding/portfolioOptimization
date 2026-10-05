

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from scan import ROOT, RESULTS, TRADING_DAYS, max_drawdown, run_pair, sharpe

OUT = RESULTS / "portfolio.json"
# The GitHub Pages site reads its numbers from files this repo generates,
# so the portfolio section gets written out here rather than typed into the
# HTML. Loaded after data.js, which the scan produces.
SITE_OUT = ROOT.parent / "docs" / "portfolio.js"

# Book sizes to compare. The point of the sweep is the shape, not any one
# number: if selecting harder helped, Sharpe would fall as N grows.
SIZES = (1, 5, 10, 25, 50, 100, 250, 500, 930)


def pair_returns(results: pd.DataFrame, trading: pd.DataFrame) -> pd.DataFrame:
    
    series = {}
    for row in results.itertuples():
        result, _, _, _ = run_pair(trading[row.a], trading[row.b], row.beta)
        series[f"{row.a}/{row.b}"] = result["ret_net"]
    return pd.DataFrame(series)


def book(returns: pd.DataFrame, equal_risk: bool = False) -> dict:
    
    if equal_risk:
        vol = returns.std(ddof=1).replace(0.0, np.nan)
        daily = (returns / vol).mean(axis=1)
        daily = daily / daily.std(ddof=1) * returns.mean(axis=1).std(ddof=1)
    else:
        daily = returns.mean(axis=1)
    equity = (1.0 + daily).cumprod()
    return {
        "n_pairs": int(returns.shape[1]),
        "total_return": float(equity.iloc[-1] - 1.0),
        "sharpe": sharpe(daily),
        "max_drawdown": max_drawdown(equity),
        "ann_return": float(daily.mean() * TRADING_DAYS),
        "ann_vol": float(daily.std(ddof=1) * np.sqrt(TRADING_DAYS)),
    }


def average_correlation(returns: pd.DataFrame, sample: int = 200,
                        seed: int = 0) -> float:
    """Mean off-diagonal correlation of the pair return series.

    Sampled rather than computed over all 930 columns: a 930x930 correlation
    matrix is 432,000 pairwise correlations of 1,254 observations each, and
    the mean of a 200-column sample is the same number to two decimals for a
    fraction of the work.
    """
    rng = np.random.default_rng(seed)
    cols = returns.columns
    if len(cols) > sample:
        cols = rng.choice(cols, size=sample, replace=False)
    corr = returns[cols].corr().to_numpy()
    off = corr[~np.eye(len(cols), dtype=bool)]
    return float(np.nanmean(off))


def diversification_check(returns: pd.DataFrame, stats: dict) -> dict:
    """Does the book's Sharpe match what averaging N correlated series gives?

    For equally-weighted series with mean pairwise correlation rho, the
    portfolio Sharpe is the average single-name Sharpe times

        sqrt(N) / sqrt(1 + (N - 1) * rho)

    which for large N stops growing and tends to 1/sqrt(rho). Comparing the
    predicted number to the realized one is a check that the book's result
    is diversification arithmetic rather than something the pairs are doing
    together - and if they diverge, the divergence is the finding.
    """
    n = returns.shape[1]
    rho = average_correlation(returns)
    per_pair = np.array([sharpe(returns[c]) for c in returns.columns])
    mean_sharpe = float(np.nanmean(per_pair))
    predicted = mean_sharpe * np.sqrt(n) / np.sqrt(1 + (n - 1) * rho) if n > 1 else mean_sharpe
    # The sqrt(N) formula assumes every series contributes the same risk.
    # Equal DOLLAR weighting doesn't do that - these pairs' volatilities
    # span a factor of six - so the equal-risk book is the one the formula
    # is actually a prediction about, and the gap between the two is the
    # volatile pairs' contribution.
    vol = returns.std(ddof=1)
    vol_sharpe_corr = float(np.corrcoef(vol.to_numpy(), per_pair)[0, 1])

    return {
        "mean_pair_sharpe": mean_sharpe,
        "median_pair_sharpe": float(np.nanmedian(per_pair)),
        "best_pair_sharpe": float(np.nanmax(per_pair)),
        "worst_pair_sharpe": float(np.nanmin(per_pair)),
        "share_positive_sharpe": float((per_pair > 0).mean()),
        "share_profitable": float(
            (returns.add(1.0).prod() - 1.0 > 0).mean()),
        "avg_correlation": rho,
        "predicted_book_sharpe": float(predicted),
        "realized_book_sharpe": stats["sharpe"],
        "equal_risk_book_sharpe": book(returns, equal_risk=True)["sharpe"],
        "corr_pair_vol_vs_sharpe": vol_sharpe_corr,
        "vol_ratio_max_to_min": float(vol.max() / vol.min()),
    }


def does_the_screen_predict(results: pd.DataFrame, returns: pd.DataFrame,
                            n_buckets: int = 5) -> pd.DataFrame:
    """Sort the survivors by formation p-value, then compare book by bucket.

    This is the question the scan exists to answer. A lower p-value is a
    stronger claim that the pair was cointegrated in 2013-2020. If that
    claim carries information about 2021-2025, the strongest bucket should
    out-trade the weakest. Each bucket is traded as its own equal-weight
    book so the comparison is between portfolios, not between single pairs.
    """
    ranked = results.sort_values("pvalue").reset_index(drop=True)
    edges = np.linspace(0, len(ranked), n_buckets + 1).astype(int)

    rows = []
    for i in range(n_buckets):
        chunk = ranked.iloc[edges[i]:edges[i + 1]]
        cols = [f"{r.a}/{r.b}" for r in chunk.itertuples()]
        stats = book(returns[cols])
        rows.append({
            "bucket": f"{i + 1} ({'strongest' if i == 0 else 'weakest' if i == n_buckets - 1 else 'middle'})",
            "pvalue_from": float(chunk["pvalue"].min()),
            "pvalue_to": float(chunk["pvalue"].max()),
            **stats,
        })
    return pd.DataFrame(rows)


def rank_correlation(results: pd.DataFrame) -> dict:
    """Spearman rank correlation between formation p-value and OOS Sharpe.

    The bucket table above, reduced to one number. Rank rather than Pearson
    because p-values are wildly non-normal and a handful of very small ones
    would otherwise decide the answer on their own.
    """
    x = results["pvalue"].rank()
    y = results["sharpe"].rank()
    rho = float(np.corrcoef(x, y)[0, 1])
    n = len(results)
    # Standard error of a Spearman correlation under the null of no
    # association, so the number can be read against something.
    se = 1.0 / np.sqrt(n - 1)
    return {"spearman": rho, "n": int(n), "se_under_null": float(se),
            "z": float(rho / se)}


def leg_concentration(results: pd.DataFrame) -> pd.DataFrame:
    
    legs = pd.concat([results["a"], results["b"]]).value_counts()
    total = int(legs.sum())
    return pd.DataFrame({
        "ticker": legs.index,
        "n_legs": legs.to_numpy(),
        "share_of_legs": legs.to_numpy() / total,
    })


def main() -> None:
    results = pd.read_parquet(RESULTS / "results.parquet")
    trading = pd.read_parquet(RESULTS / "trading.parquet")
    print(f"{len(results)} survivors, trading window {trading.index[0].date()} "
          f"-> {trading.index[-1].date()}")

    print("Re-running every survivor for its return series...")
    returns = pair_returns(results, trading)

    print("\nEqual-weight book, top N by formation p-value")
    print(f"{'N pairs':>8} {'return':>9} {'sharpe':>8} {'max dd':>9} {'ann vol':>9}")
    ranked = results.sort_values("pvalue").reset_index(drop=True)
    sweep = []
    for n in SIZES:
        if n > len(ranked):
            continue
        cols = [f"{r.a}/{r.b}" for r in ranked.head(n).itertuples()]
        stats = book(returns[cols])
        sweep.append(stats)
        print(f"{n:>8} {stats['total_return']:>9.2%} {stats['sharpe']:>8.2f} "
              f"{stats['max_drawdown']:>9.2%} {stats['ann_vol']:>9.2%}")

    full = book(returns)
    div = diversification_check(returns, full)
    print(f"\nAll {full['n_pairs']} pairs, equal weight")
    print(f"  mean single-pair Sharpe   {div['mean_pair_sharpe']:>7.3f}")
    print(f"  median single-pair Sharpe {div['median_pair_sharpe']:>7.3f}")
    print(f"  share with positive Sharpe {div['share_positive_sharpe']:>6.1%}")
    print(f"  share that made money      {div['share_profitable']:>6.1%}")
    print(f"  average pair correlation  {div['avg_correlation']:>7.3f}")
    print(f"  book Sharpe predicted     {div['predicted_book_sharpe']:>7.2f}")
    print(f"  book Sharpe realized      {div['realized_book_sharpe']:>7.2f}")
    print(f"  same book, equal RISK     {div['equal_risk_book_sharpe']:>7.2f}")
    print(f"  corr(pair vol, pair Sharpe) {div['corr_pair_vol_vs_sharpe']:>5.2f}"
          f"  (pair vols span {div['vol_ratio_max_to_min']:.1f}x)")

    print("\nDoes the formation p-value predict out-of-sample Sharpe?")
    buckets = does_the_screen_predict(results, returns)
    print(f"{'bucket':<16} {'p-value range':>24} {'return':>9} {'sharpe':>8}")
    for row in buckets.itertuples():
        rng_txt = f"{row.pvalue_from:.1e} - {row.pvalue_to:.1e}"
        print(f"{row.bucket:<16} {rng_txt:>24} {row.total_return:>9.2%} "
              f"{row.sharpe:>8.2f}")
    rc = rank_correlation(results)
    print(f"\nSpearman(p-value, OOS Sharpe) = {rc['spearman']:+.3f} "
          f"over {rc['n']} pairs (SE under the null {rc['se_under_null']:.3f}, "
          f"z = {rc['z']:+.2f})")

    conc = leg_concentration(results)
    top = conc.head(5)
    print(f"\nLeg concentration: {len(conc)} distinct tickers across "
          f"{len(results) * 2} legs")
    for row in top.itertuples():
        print(f"  {row.ticker:<6} {row.n_legs:>4} legs  {row.share_of_legs:>6.2%} of the book")

    payload = {
        "sweep": sweep,
        "full_book": full,
        "diversification": div,
        "buckets": buckets.to_dict("records"),
        "rank_correlation": rc,
        "concentration": conc.head(15).to_dict("records"),
        "equity": [[d.strftime("%Y-%m-%d"), round(float(v), 5)]
                   for d, v in (1.0 + returns.mean(axis=1)).cumprod().items()],
    }
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"\nwrote {OUT}")

    if SITE_OUT.parent.exists():
        SITE_OUT.write_text(
            "window.PORTFOLIO = " + json.dumps(payload, separators=(",", ":")) + ";\n",
            encoding="utf-8")
        print(f"wrote {SITE_OUT} ({SITE_OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
