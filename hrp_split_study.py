"""HRP's worst-looking step is the one holding it together.

Usage:  python hrp_split_study.py [--quick]

src/hrp.py has carried a paragraph of small print since the day it was
written. Lopez de Prado's recursive bisection does not cut the TREE it just
spent three steps building - it cuts the ordered list of assets in half by
count. Where the tree's real boundary is somewhere other than the midpoint,
the algorithm splits a cluster down the middle and weighs half of it against
something it is not related to. The measured consequence, in
tests/test_hrp.py: three copies of one bet plus one independent bet get cut
2-2 instead of 3-1, and the duplicates keep about two thirds of the book
where they should keep a half.

So write the fix. tree_bisection() descends the dendrogram and splits at each
of its actual joins. On the synthetic case it is exactly right - two thirds
becomes 0.502, near enough a half to call it solved.

On real data it is much worse, and the reason is the interesting part.

A correlation dendrogram of financial assets is not a tree, it is a LADDER.
Both the 50 large caps and a 20-asset multi-asset book split one asset
against all the rest at the root, and attach a single asset at three quarters
of their joins. Switching from single linkage to average linkage - the
textbook cure for exactly this chaining - barely moves it, because the cause
is not the update rule. It is that a correlation matrix dominated by one
common factor has no block structure to find, so everything is roughly
equidistant from everything and greedy agglomeration snowballs whichever way
it is told to measure distance. The synthetic control in table 1 confirms the
diagnosis from the other side: build a covariance that really does have eight
blocks in it and the same code produces a balanced tree.

Descending a ladder means the first split is one asset against forty-nine,
and that one asset takes a large share of the book before anything else is
considered. Cutting the list at its midpoint is protected from this, because
a midpoint is balanced whatever the tree looks like - which is not a property
anybody claimed for it, and is apparently what most of the method is.

Table 4 is the consequence taken seriously. If the balance is doing the work
rather than the clustering, then throwing the clustering away entirely -
bisecting a RANDOM order, same midpoint cut - should barely hurt. It barely
does. Averaged over 47 quarterly rebalances, the clustered order's predicted
volatility is 0.19 percentage points below the average random order's on the 50
large caps, which is about two standard deviations of the random distribution
and 1.3% in relative terms; on the 20-asset multi-asset book the gap is 0.05
points, which is a fifth of a standard deviation and indistinguishable from
nothing.

The direction of that second row was a surprise, since the multi-asset book is
where the blocks are most obviously real, and it is worth naming the likeliest
reason rather than leaving it. The split's decision comes from
cluster_variance(), which weights each side by an inverse-variance sub-portfolio.
On a book holding both Treasuries at 4% vol and equities at 18%, those variances
differ by so much that they, and not which assets happen to sit together,
determine where the money goes. Twenty assets is also only nineteen splits.
That is a conjecture; what is measured is the two numbers.

Either way the conclusion holds and it is about the method rather than the fix:
three of HRP's four steps buy about one percent of predicted volatility, and the
fourth - the one usually written up as the wart - is carrying the rest. Being
able to say which of your own steps is load-bearing is worth more than another
row in a table of Sharpes.
"""

import argparse

import numpy as np
import pandas as pd
import yfinance as yf

from src.hrp import (correlation_distance, correlation_from_covariance,
                     hrp_weights, linkage, recursive_bisection, tree_shape)
from src.returns import TRADING_DAYS, daily_returns

STOCKS = [
    "AAPL", "MSFT", "ORCL", "CSCO", "INTC", "IBM", "TXN", "QCOM", "ADBE",
    "JNJ", "MRK", "PFE", "ABT", "UNH", "LLY", "AMGN", "GILD",
    "JPM", "BAC", "GS", "MS", "AXP", "USB",
    "XOM", "CVX",
    "PG", "KO", "PEP", "WMT", "COST", "TGT", "MCD", "SBUX", "NKE", "HD", "LOW",
    "CAT", "BA", "MMM", "HON", "GE", "RTX", "LMT", "UPS",
    "DIS", "CMCSA", "VZ", "T", "NEE", "DUK",
]
# Chosen so the blocks are real: five equity buckets, five in rates and
# credit, commodities, property and the dollar. If clustering ever helps
# anywhere, it helps here.
MULTI = ["SPY", "QQQ", "IWM", "EFA", "EEM", "XLE", "XLF", "XLK", "XLU", "XLP",
         "TLT", "IEF", "LQD", "HYG", "TIP", "GLD", "SLV", "DBC", "VNQ", "UUP"]

START, END = "2010-01-01", "2024-12-31"
HOLD_DAYS = 63
WINDOW_DAYS = 3 * TRADING_DAYS
COMBOS = (("single", "list"), ("single", "tree"),
          ("average", "list"), ("average", "tree"))


def load(tickers: list[str]) -> pd.DataFrame:
    px = yf.download(tickers, start=START, end=END, auto_adjust=True,
                     progress=False)["Close"]
    px = px.dropna(axis=1, how="any")
    return px.loc[:, [t for t in tickers if t in px.columns]].dropna()


def block_covariance(n_blocks: int = 8, per_block: int = 4,
                     seed: int = 1) -> np.ndarray:
    """A covariance that genuinely has blocks, as a control on the diagnosis.

    One market factor plus one factor per block plus idiosyncratic noise. If
    the ladder in table 1 were an artefact of the clustering code rather than
    of the data, this would come out as a ladder too.
    """
    rng = np.random.default_rng(seed)
    market = rng.normal(size=500)
    columns = []
    for _ in range(n_blocks):
        block = rng.normal(size=500)
        for _ in range(per_block):
            columns.append(0.5 * market + 0.7 * block
                           + 0.5 * rng.normal(size=500))
    return np.cov(np.array(columns).T, rowvar=False)


def shapes(cov: np.ndarray) -> list[dict]:
    """Tree shape under each linkage, for one covariance matrix."""
    dist = correlation_distance(correlation_from_covariance(cov))
    return [{"method": m, **tree_shape(linkage(dist, method=m))}
            for m in ("single", "average")]


def duplicate_case() -> list[dict]:
    """Three near-identical bets and one independent one, all the same vol.

    The case the fix was written for. 'right answer' depends on what you think
    HRP is claiming - inverse-variance weighting gives the trio three
    quarters, treating one bet as one bet gives it a half - but the trio
    getting MORE than half is the thing nobody defends.
    """
    corr = np.eye(4)
    for i in range(3):
        for j in range(3):
            if i != j:
                corr[i, j] = 0.99
    cov = corr * 0.04
    out = []
    for method, split in COMBOS:
        w = hrp_weights(cov, split=split, method=method)
        out.append({"method": method, "split": split,
                    "trio": float(w[:3].sum()), "lone": float(w[3])})
    return out


def walk_forward(prices: pd.DataFrame, window: int = WINDOW_DAYS) -> pd.DataFrame:
    """The four combinations through the same rebalance dates, out of sample.

    Same estimation window, same quarterly hold, same everything except which
    linkage builds the tree and where the money is split. Predicted volatility
    is what the method told you at the rebalance; realized is what the next
    quarter actually delivered.
    """
    values = daily_returns(prices).values
    realized: dict[str, list] = {}
    predicted: dict[str, list] = {}
    turnover: dict[str, list] = {}
    effn: dict[str, list] = {}
    prev: dict[str, np.ndarray] = {}
    n_rebal = 0

    for start in range(window, len(values) - HOLD_DAYS + 1, HOLD_DAYS):
        cov = np.cov(values[start - window:start], rowvar=False) * TRADING_DAYS
        test = values[start:start + HOLD_DAYS]
        n_rebal += 1
        for method, split in COMBOS:
            name = f"{method} linkage, {split} split"
            w = hrp_weights(cov, split=split, method=method)
            predicted.setdefault(name, []).append(float(np.sqrt(w @ cov @ w)))
            realized.setdefault(name, []).extend((test @ w).tolist())
            effn.setdefault(name, []).append(1.0 / float(np.sum(w ** 2)))
            if name in prev:
                turnover.setdefault(name, []).append(
                    float(np.abs(w - prev[name]).sum() / 2))
            prev[name] = w

    rows = []
    for name, series in realized.items():
        r = np.asarray(series)
        vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
        ann = float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
        pred = float(np.mean(predicted[name]))
        rows.append({"method": name, "predicted": pred, "realized": vol,
                     "ratio": vol / pred, "ann return": ann,
                     "sharpe": ann / vol if vol else 0.0,
                     "turnover": float(np.mean(turnover.get(name, [0.0]))),
                     "effective n": float(np.mean(effn[name]))})
    out = pd.DataFrame(rows)
    out.attrs["rebalances"] = n_rebal
    return out


def order_ablation(prices: pd.DataFrame, n_draws: int = 200,
                   window: int = WINDOW_DAYS) -> dict:
    """What the clustering is worth: its order against random orders.

    Hold the bisection fixed - always the midpoint cut - and change only the
    order it is applied to. Anything the clustered order achieves over a
    random one is what the first three steps of HRP contributed. The spread
    across random draws is the scale to judge it against, which is the part
    that makes this an answer rather than two numbers.
    """
    values = daily_returns(prices).values
    rng = np.random.default_rng(0)
    clustered, random_mean, random_sd = [], [], []

    for start in range(window, len(values) - HOLD_DAYS + 1, HOLD_DAYS):
        cov = np.cov(values[start - window:start], rowvar=False) * TRADING_DAYS
        n = cov.shape[0]
        w = hrp_weights(cov)
        clustered.append(float(np.sqrt(w @ cov @ w)))

        draws = []
        for _ in range(n_draws):
            rw = recursive_bisection(cov, list(rng.permutation(n)))
            rw = rw / rw.sum()
            draws.append(float(np.sqrt(rw @ cov @ rw)))
        random_mean.append(float(np.mean(draws)))
        random_sd.append(float(np.std(draws)))

    gain = np.array(random_mean) - np.array(clustered)
    return {"clustered": float(np.mean(clustered)),
            "random": float(np.mean(random_mean)),
            "random_sd": float(np.mean(random_sd)),
            "gain": float(gain.mean()),
            "sigmas": float(gain.mean() / np.mean(random_sd)),
            "rebalances": len(clustered)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                        help="40 random orders instead of 200")
    args = parser.parse_args()
    draws = 40 if args.quick else 200

    print("Downloading...")
    stocks, multi = load(STOCKS), load(MULTI)
    print(f"  {len(stocks.columns)} stocks and {len(multi.columns)} "
          f"multi-asset ETFs, {stocks.index[0].date()} to "
          f"{stocks.index[-1].date()}")

    books = [("50 large caps", stocks), ("20 multi-asset ETFs", multi)]

    print("\n\n1. The shape of the tree the fix wants to descend\n")
    print(f"{'book':<24} {'linkage':<9} {'root split':>12} "
          f"{'1-asset joins':>14} {'balance':>8}")
    for label, px in books:
        cov = np.cov(daily_returns(px).values[-WINDOW_DAYS:],
                     rowvar=False) * TRADING_DAYS
        for s in shapes(cov):
            root = f"{s['root'][0]} v {s['root'][1]}"
            print(f"{label:<24} {s['method']:<9} {root:>12} "
                  f"{s['single_asset_joins']:>7}/{s['joins']:<6} "
                  f"{s['balance']:>8.2f}")
    for s in shapes(block_covariance()):
        root = f"{s['root'][0]} v {s['root'][1]}"
        print(f"{'synthetic, 8 real blocks':<24} {s['method']:<9} {root:>12} "
              f"{s['single_asset_joins']:>7}/{s['joins']:<6} "
              f"{s['balance']:>8.2f}")
    print("\nbalance = mean of (smaller branch / larger branch) over every join, "
          "weighted by\ncluster size. 1.0 is a balanced binary tree; a ladder "
          "tends to 2/n. The synthetic\nrow is the control: the same code finds "
          "a balanced tree when there is one to find.")

    print("\n\n2. The case the fix was written for\n")
    print(f"{'linkage':<9} {'split':<6} {'three duplicates':>18} {'lone asset':>12}")
    for r in duplicate_case():
        print(f"{r['method']:<9} {r['split']:<6} {r['trio']:>17.1%} "
              f"{r['lone']:>11.1%}")
    print("\nThe tree split is simply correct here, under either linkage: the "
          "trio drops from\ntwo thirds of the book to a half. Whatever is "
          "wrong with it on real data is not\nthis.")

    print("\n\n3. Out of sample, same dates, only the tree and the cut change")
    for label, px in books:
        table = walk_forward(px)
        print(f"\n{label}  ({table.attrs['rebalances']} quarterly "
              f"rebalances)\n")
        print(f"{'':<28} {'pred':>6} {'real':>6} {'ratio':>6} {'ret':>7} "
              f"{'sharpe':>7} {'turn':>6} {'eff n':>6}")
        for _, r in table.sort_values("realized").iterrows():
            print(f"{r['method']:<28} {r['predicted']:>6.1%} "
                  f"{r['realized']:>6.1%} {r['ratio']:>6.2f} "
                  f"{r['ann return']:>7.2%} {r['sharpe']:>7.2f} "
                  f"{r['turnover']:>6.1%} {r['effective n']:>6.1f}")

    print("\n\n4. So what is the clustering worth? Its order against random "
          "ones\n")
    print(f"{'book':<24} {'clustered':>10} {'random':>10} {'sd':>7} "
          f"{'gain':>7} {'sigmas':>7}")
    for label, px in books:
        a = order_ablation(px, n_draws=draws)
        print(f"{label:<24} {a['clustered']:>10.2%} {a['random']:>10.2%} "
              f"{a['random_sd']:>7.2%} {a['gain']:>7.2%} "
              f"{a['sigmas']:>7.1f}")
    print(f"\nPredicted portfolio volatility, averaged over rebalances, with "
          f"{draws} random orders\nper date. 'gain' is how much lower the "
          "clustered order's volatility is; 'sigmas' is\nthat gain measured "
          "in standard deviations of the random draws.")
    print("\nSo the clustering is worth something on the stock book - two "
          "standard deviations,\nand 1.3% of the volatility it is predicting - "
          "and nothing measurable on the\nmulti-asset one, which is the "
          "opposite of where the blocks are most obviously\nreal. The likely "
          "reason is in cluster_variance(): with Treasuries at 4% vol beside\n"
          "equities at 18%, the variances decide the split rather than which "
          "assets sit\ntogether. Either way, three of HRP's four steps are "
          "worth about one percent here\nand the fourth is carrying the rest.")


if __name__ == "__main__":
    main()
