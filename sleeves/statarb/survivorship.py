"""Putting a number on the one caveat this project has only ever admitted.

Usage:  python survivorship.py [--quick]      (reads results/, one small download)

Every page of this project carries the same warning: the universe is the
S&P 100 as it stands TODAY. Membership is awarded for having already grown
large, so scanning it from 2013 asks how today's winners behaved on their way
to winning, and firms that were in the index and then collapsed or were bought
are simply absent. The warning has always ended by saying the fix needs a
point-in-time constituent list, which free data does not provide.

That is true, and it has been used as a reason to leave the size of the bias
unstated, which is a different thing. An unmeasured caveat is indistinguishable
from a large one. So this file does what can be done without the data that
does not exist, in two directions.

WHAT THE BIAS IS, MEASURED. The selection is observable from the inside, though
not in the place people usually look. The median surviving name roughly matched
SPY over the formation window and fewer than half of them beat it - which is
nearly circular, since SPY is cap-weighted and largely made of these same
companies. The selection shows in the LEFT TAIL: the worst name in this list
over ten years returned -52%, and a genuine 2013 large-cap universe followed to
2025 contains companies that went to zero, were bought at a premium, or shrank
out of the index entirely. What the mean does show is the right tail - one name
returned 3,636%. The question that matters for a cointegration screen is whether
that shared drift is what the screen is picking up. Two stocks that both tripled between 2013 and 2020 have a strong
common trend, and a common trend is exactly what makes an ADF test on a fitted
residual reject when it should not - which section 4 of RESULTS.md already
showed it does, three times too often. So: split the 4,950 pairs by how much
both legs outperformed during formation and look at the pass rate. If
co-winners pass more often, some of the 930 is selection rather than structure,
and the amount is readable off the table.

WHAT THE BIAS COSTS, BOUNDED. The event a survivors-only universe can never
contain is a leg being acquired. That is not a small tail: a target jumps to
near the offer price in one print, then stops moving, then delists - so a
cointegrated spread gaps and never mean-reverts, and the strategy, which makes
small money on reversion and has nothing to cap a gap, is forced to realise
the whole loss. There is no way to observe those events in this universe. There
is a way to inject them at a stated rate and measure what they cost, which
turns "unknown" into a range with assumptions attached. The rate and the
premium below are external estimates, not measurements, and the table sweeps
both so no conclusion rests on one guess.

The point of the second half is not the exact number. It is that a strategy
whose per-pair edge is already indistinguishable from zero does not have room
to absorb an unmodelled loss of that size, and saying so with a figure is
worth more than a warning box.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from sleeves.statarb.scan import COST_BPS, TRADING_DAYS, run_pair, sharpe

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

# M&A base rate. S&P 500 sees roughly 20-25 constituent changes a year and
# about half are mergers or acquisitions; the S&P 100 is more stable in
# proportion, but 1-2 of its hundred names a year leaving for a deal is the
# right order. The grid brackets that rather than betting on it.
DEAL_RATE_GRID = (0.005, 0.01, 0.02, 0.04)
PREMIUM_GRID = (0.20, 0.30)
DAYS_TO_CLOSE = 110          # ~5 months, typical for a large US cash deal
DEAL_VOL = 0.0015            # daily vol of a target between announce and close
N_TRIALS = 12           # each trial re-backtests every pair a deal touched, twice
BENCH = "SPY"


def load_cached() -> dict:
    scan = pd.read_parquet(RESULTS / "scan.parquet")
    results = pd.read_parquet(RESULTS / "results.parquet")
    formation = pd.read_parquet(RESULTS / "formation.parquet")
    trading = pd.read_parquet(RESULTS / "trading.parquet")
    meta = json.loads((RESULTS / "meta.json").read_text())
    return {"scan": scan, "results": results, "formation": formation,
            "trading": trading, "meta": meta}


def benchmark_return(start, end) -> float | None:
    """SPY's total return over the formation window, or None if offline.

    One number, one download, and the study still runs without it - the
    within-universe comparison is what the tables turn on, and the benchmark
    only sets the scene.
    """
    try:
        import yfinance as yf
        px = yf.download(BENCH, start=str(start.date()), end=str(end.date()),
                         auto_adjust=True, progress=False)["Close"].dropna()
        return float(px.iloc[-1].item() / px.iloc[0].item() - 1.0)
    except Exception:
        return None


def formation_performance(formation: pd.DataFrame) -> pd.Series:
    """Each surviving name's total return over the formation window."""
    return (formation.iloc[-1] / formation.iloc[0] - 1.0).sort_values()


def pass_rate_by_performance(scan: pd.DataFrame, perf: pd.Series,
                            n_buckets: int = 4) -> pd.DataFrame:
    """Does the screen pass co-winners more often than it passes the rest?

    Each leg gets a quartile by formation return, and a pair is bucketed by the
    LOWER of its two, so "top quartile" means both legs were top-quartile
    winners. The bottom bucket is pairs where at least one leg was, relatively,
    a laggard - which is as close as this universe gets to containing a name
    that did not deserve its place in it.
    """
    ranks = perf.rank(pct=True)
    out = []
    lo_edges = np.linspace(0, 1, n_buckets + 1)
    bucket_of = {}
    for ticker, r in ranks.items():
        bucket_of[ticker] = min(int(r * n_buckets - 1e-9), n_buckets - 1)

    weaker = scan.apply(lambda r: min(bucket_of.get(r["a"], 0),
                                      bucket_of.get(r["b"], 0)), axis=1)
    for b in range(n_buckets):
        sub = scan[weaker == b]
        if sub.empty:
            continue
        names = [t for t, v in bucket_of.items() if v == b]
        out.append({
            "bucket": b + 1,
            "label": f"{lo_edges[b]:.0%}-{lo_edges[b + 1]:.0%}",
            "pairs": len(sub),
            "pass_rate": float((sub["pvalue"] <= 0.05).mean()),
            "median_p": float(sub["pvalue"].median()),
            "mean_formation_return": float(perf[names].mean()),
        })
    return pd.DataFrame(out)


def oos_by_performance(results: pd.DataFrame, perf: pd.Series,
                       n_buckets: int = 4) -> pd.DataFrame:
    """And are the co-winner pairs any better out of sample? (No.)"""
    ranks = perf.rank(pct=True)
    bucket_of = {t: min(int(r * n_buckets - 1e-9), n_buckets - 1)
                 for t, r in ranks.items()}
    weaker = results.apply(lambda r: min(bucket_of.get(r["a"], 0),
                                         bucket_of.get(r["b"], 0)), axis=1)
    rows = []
    for b in range(n_buckets):
        sub = results[weaker == b]
        if sub.empty:
            continue
        rows.append({"bucket": b + 1, "pairs": len(sub),
                     "mean_sharpe": float(sub["sharpe"].mean()),
                     "median_sharpe": float(sub["sharpe"].median()),
                     "share_profitable": float((sub["total_return"] > 0).mean())})
    return pd.DataFrame(rows)


def inject_deals(trading: pd.DataFrame, rng, rate: float, premium: float,
                 days_to_close: int = DAYS_TO_CLOSE) -> tuple[pd.DataFrame, list]:
    """Acquire some names mid-window and rewrite their prices accordingly.

    For each name drawn, pick an announcement day uniformly over the window,
    then replace everything from that day on with

        one gap of +premium, then a near-flat walk, then nothing at all

    which is what a target actually does: it stops being a stock and becomes a
    bond paying the offer price, and then it delists. Truncating the series is
    the part that does the damage to a pairs book, because a position in a
    spread that gapped and stopped reverting has to be closed at whatever it is
    worth on the last day rather than waited out.

    The number of deals is Poisson with mean rate * n_names * years, so a run
    can draw none, and the spread across trials is part of the answer.
    """
    years = len(trading) / TRADING_DAYS
    expected = rate * trading.shape[1] * years
    n_deals = int(rng.poisson(expected))
    if n_deals == 0:
        return trading, []

    names = list(rng.choice(trading.columns, size=min(n_deals,
                                                      trading.shape[1]),
                            replace=False))
    out = trading.copy()
    events = []
    # Leave room for the whole deal to play out inside the window sometimes and
    # not others; a deal announced near the end simply truncates earlier.
    latest = len(trading) - 2
    for name in names:
        i = int(rng.integers(30, latest))
        offer = float(out[name].iloc[i - 1]) * (1.0 + premium)
        close_i = min(i + days_to_close, len(out))
        n = close_i - i
        drift = rng.normal(0.0, DEAL_VOL, size=n).cumsum()
        out.iloc[i:close_i, out.columns.get_loc(name)] = offer * np.exp(drift)
        if close_i < len(out):
            out.iloc[close_i:, out.columns.get_loc(name)] = np.nan
        events.append({"ticker": name, "day": int(i),
                       "date": str(out.index[i].date()),
                       "closed": bool(close_i < len(out))})
    return out, events


def book_return(pairs: pd.DataFrame, trading: pd.DataFrame) -> dict:
    """Equal-weight book over a set of pairs, on a given price history.

    A pair whose legs do not overlap for long enough to form a z-score simply
    contributes nothing, which is the honest treatment: a book cannot trade a
    spread one of whose legs has delisted.
    """
    curves = {}
    for row in pairs.itertuples():
        y, x = trading[row.a].dropna(), trading[row.b].dropna()
        idx = y.index.intersection(x.index)
        if len(idx) < 120:
            continue
        result, _, _, _ = run_pair(y.loc[idx], x.loc[idx], row.beta)
        curves[f"{row.a}/{row.b}"] = result["ret_net"]

    book = pd.DataFrame(curves).mean(axis=1).reindex(trading.index).fillna(0.0)
    equity = (1.0 + book).cumprod()
    return {"total_return": float(equity.iloc[-1] - 1.0),
            "sharpe": sharpe(book),
            "max_drawdown": float((equity / equity.cummax() - 1.0).min()),
            "pairs_traded": len(curves)}


def pair_return(trading: pd.DataFrame, a: str, b: str, beta: float,
                index=None) -> float | None:
    """One pair's total return, optionally restricted to a given set of dates.

    `index` is what makes the comparison in deal_impact() a controlled one. A
    takeover shortens a pair's history, and a shorter history changes the return
    all by itself, so the only way to isolate the gap is to score the real
    prices over exactly the days the modified history has. Passing the modified
    pair's index does that.
    """
    y, x = trading[a].dropna(), trading[b].dropna()
    idx = y.index.intersection(x.index)
    if index is not None:
        idx = idx.intersection(index)
    if len(idx) < 120:
        return None
    _, _, _, stats = run_pair(y.loc[idx], x.loc[idx], beta)
    return stats["total_return"]


def pair_returns(pairs: pd.DataFrame, trading: pd.DataFrame) -> dict:
    """Every pair's total return on a given price history, keyed by (a, b)."""
    out = {}
    for row in pairs.itertuples():
        r = pair_return(trading, row.a, row.b, row.beta)
        if r is not None:
            out[(row.a, row.b)] = r
    return out


def deal_impact(pairs: pd.DataFrame, trading: pd.DataFrame,
                n_trials: int = N_TRIALS, rates=DEAL_RATE_GRID,
                premiums=PREMIUM_GRID, seed: int = 0) -> pd.DataFrame:
    """What a takeover does to the pairs it touches, matched pair by pair.

    The naive version of this measurement - book return with deals minus book
    return without - is not interpretable, and it is worth saying why rather
    than quietly not doing it. Acquiring a name TRUNCATES its series, so every
    pair holding it trades fewer days; this book loses money, so trading fewer
    days improves it. The two effects are opposite in sign and the difference
    of the totals mixes them.

    So the comparison here is matched: for each pair that a deal touched, its
    return on the modified history against its return on the real one, over
    whatever days it had. Same pair, same hedge ratio, same rules; the only
    difference is the takeover. That isolates the thing being measured and
    leaves the truncation out of it.

    The window is matched too, and that is the part it is easy to get wrong. A
    takeover TRUNCATES the leg's history, so an affected pair trades fewer days
    even before the gap is considered; this book loses money, so fewer days
    improves it, and comparing a short modified history against a full real one
    measures that instead of the merger. Every delta below scores the real
    prices over exactly the dates the modified pair had, so the only difference
    between the two numbers is what the acquired leg did.

    A pair left with fewer than 120 usable days cannot form a z-score at all and
    is counted in 'killed' rather than given a delta: it earns zero, and zero is
    BETTER than this book's average pair, so folding those into the mean would
    make takeovers look profitable. That is a fact about the strategy being a
    loser, not about mergers.
    """
    rows = []
    for premium in premiums:
        for rate in rates:
            rng = np.random.default_rng(seed)
            deltas, n_deals, affected_share, killed, total = [], [], [], 0, 0
            for _ in range(n_trials):
                modified, events = inject_deals(trading, rng, rate, premium)
                n_deals.append(len(events))
                if not events:
                    affected_share.append(0.0)
                    continue
                touched = {e["ticker"] for e in events}
                subset = pairs[pairs["a"].isin(touched)
                               | pairs["b"].isin(touched)]
                affected_share.append(len(subset) / len(pairs))

                for row in subset.itertuples():
                    total += 1
                    idx = (modified[row.a].dropna().index
                           .intersection(modified[row.b].dropna().index))
                    after = pair_return(modified, row.a, row.b, row.beta)
                    if after is None:
                        killed += 1
                        continue
                    before = pair_return(trading, row.a, row.b, row.beta,
                                         index=idx)
                    if before is not None:
                        deltas.append(after - before)

            d = np.asarray(deltas) if deltas else np.array([0.0])
            share = float(np.mean(affected_share))
            rows.append({
                "premium": premium, "rate": rate,
                "mean_deals": float(np.mean(n_deals)),
                "affected_share": share,
                "mean_delta": float(d.mean()),
                "median_delta": float(np.median(d)),
                "p5_delta": float(np.percentile(d, 5)),
                "worst_delta": float(d.min()),
                "share_worse": float((d < 0).mean()),
                "share_bad": float((d < -0.10).mean()),
                "kill_rate": killed / total if total else 0.0,
            })
    out = pd.DataFrame(rows)
    out.attrs["base"] = book_return(pairs, trading)
    out.attrs["n_baseline"] = len(pairs)
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="a 120-pair book and 20 trials; a smoke test")
    args = parser.parse_args()

    data = load_cached()
    scan, results = data["scan"], data["results"]
    formation, trading = data["formation"], data["trading"]

    perf = formation_performance(formation)
    bench = benchmark_return(formation.index[0], formation.index[-1])

    print(f"Formation window as cached: {formation.index[0].date()} to "
          f"{formation.index[-1].date()}, {formation.shape[1]} names\n")
    print("1. What the selection looks like from the inside\n")
    print(f"  median surviving name's formation return   "
          f"{perf.median():>8.0%}")
    print(f"  mean                                        "
          f"{perf.mean():>8.0%}")
    if bench is not None:
        print(f"  {BENCH} over the same window                    "
              f"{bench:>8.0%}")
        beat = float((perf > bench).mean())
        print(f"  share of the universe that beat it          {beat:>8.0%}")
    print(f"  worst / best                               "
          f"{perf.iloc[0]:>8.0%} / {perf.iloc[-1]:.0%}")
    print(f"  names in the list at all:                  {perf.index[0]} "
          f"is the weakest, {perf.index[-1]} the strongest")
    print("\n  The tell is the MINIMUM, not the mean. The median surviving name "
          "roughly\n  matched SPY, and fewer than half beat it - which is not "
          "surprising, since SPY\n  is cap-weighted and largely made of these "
          "same companies, so the comparison is\n  close to circular. What a "
          "true 2013 large-cap universe followed to 2025 would\n  contain, and "
          "this one does not, is the other end: companies that went to zero, "
          "got\n  bought at a premium, or shrank out of the index. The worst "
          f"case in this list is\n  {perf.index[0]} at {perf.iloc[0]:.0%}. "
          "Survivorship bias is not mostly a story about the\n  average being "
          "too high; it is a story about the left tail being absent.")

    print("\n\n2. Does the screen pass co-winners more often?\n")
    table = pass_rate_by_performance(scan, perf)
    print(f"{'both legs at least':<20} {'pairs':>7} {'pass rate':>10} "
          f"{'median p':>9} {'mean formation ret':>19}")
    for _, r in table.iterrows():
        print(f"{r['label']:<20} {r['pairs']:>7,} {r['pass_rate']:>10.1%} "
              f"{r['median_p']:>9.3f} {r['mean_formation_return']:>18.0%}")
    span = table["pass_rate"].max() - table["pass_rate"].min()
    print(f"\n  Spread in pass rate across buckets: {span:.1%}.")

    oos = oos_by_performance(results, perf)
    print(f"\n{'bucket':<20} {'pairs':>7} {'mean sharpe':>12} "
          f"{'median':>8} {'profitable':>11}")
    for _, r in oos.iterrows():
        print(f"{'quartile ' + str(int(r['bucket'])):<20} {r['pairs']:>7,} "
              f"{r['mean_sharpe']:>12.3f} {r['median_sharpe']:>8.3f} "
              f"{r['share_profitable']:>11.1%}")

    print("\n\n3. What a leg being acquired would cost, since none ever is\n")
    pairs = results.head(120) if args.quick else results
    trials = 20 if args.quick else N_TRIALS
    sens = deal_impact(pairs, trading, n_trials=trials)
    base = sens.attrs["base"]
    print(f"  the book as the project reports it: {base['pairs_traded']} pairs, "
          f"return {base['total_return']:+.2%}, Sharpe {base['sharpe']:+.2f}")
    print("  each row below is matched pair by pair: the same spread with the "
          "takeover and\n  without it, so the comparison is the event and not "
          "the shortened history.\n")
    print(f"{'premium':>8} {'deals/yr':>9} {'deals':>6} {'% of book':>10} "
          f"{'median':>8} {'mean':>8} {'5th pct':>9} {'worst':>9} "
          f"{'lost >10%':>10} {'killed':>7}")
    for _, r in sens.iterrows():
        print(f"{r['premium']:>7.0%} {r['rate']:>9.1%} "
              f"{r['mean_deals']:>6.1f} {r['affected_share']:>10.0%} "
              f"{r['median_delta']:>8.2%} {r['mean_delta']:>8.2%} "
              f"{r['p5_delta']:>9.2%} {r['worst_delta']:>9.2%} "
              f"{r['share_bad']:>10.0%} {r['kill_rate']:>7.0%}")
    print(f"\n  {trials} trials per row, {sens.attrs['n_baseline']} pairs in "
          "the book. 'deals' is how many of the\n  100 names were acquired "
          "somewhere in the five-year window, Poisson at the stated\n  rate; "
          "'% of book' is the share of pairs holding one of them. The middle "
          "columns\n  are the change in an affected pair's total return, "
          "measured against the same pair\n  on the real history. 'killed' is "
          "the share whose leg delisted early enough that\n  the spread "
          "stopped being tradeable at all - those earn zero rather than a "
          "loss,\n  and are kept out of the other columns so they cannot "
          "flatter the result.")
    print("\n  Read the median against the 5th percentile, because they say "
          "different things.\n  A takeover is not a systematic drag on a pairs "
          "book: the strategy is short one\n  leg and long the other and flips "
          "between them, so a gap is about as likely to\n  land your way as "
          "against you, and the median affected pair is nearly unmoved.\n  What "
          "it is is a TAIL. Nothing in the strategy caps a gap - reversion pays "
          "in\n  small amounts over weeks and a takeover does not revert - so "
          "the bottom of the\n  distribution is far from the middle. For a book "
          "whose per-pair edge is already\n  indistinguishable from zero, a "
          "risk shaped like that is the whole story, and\n  this universe "
          "cannot contain one instance of it.")


if __name__ == "__main__":
    main()
