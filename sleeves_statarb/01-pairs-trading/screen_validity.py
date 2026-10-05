"""Is the screen's p-value a p-value?

Usage:  python screen_validity.py        (reads data/ and results/, no network)

RESULTS.md has spent a lot of words on what the cointegration screen does
not buy you: the 930 survivors do not out-trade the 4,020 rejects, the
p-value does not rank anything, and re-estimating more often does not help.
Every one of those findings takes the screen's own arithmetic at face value
and shows the arithmetic does not pay. This asks the question underneath:
is the number the screen is thresholding on the thing it claims to be?

The screen does what nearly every pairs tutorial does. Regress one log
price on the other, take the residual, and run an augmented Dickey-Fuller
test on it:

    beta = OLS(y ~ x)
    spread = y - a - beta * x
    p = adfuller(spread).pvalue

The residual is not data. It is the output of a regression chosen to make
that residual as small as possible, so it looks more stationary than a
series of the same statistical character would - the OLS fit has already
spent some of the wandering. Dickey-Fuller's tables were computed for a
series nobody fitted anything to, so applying them here produces a number
that is labelled 0.05 and is not 0.05. This is exactly why Engle and
Granger, and then Phillips and Ouliaris, published SEPARATE critical values
for the case where beta was estimated - and why statsmodels ships coint(),
which uses them, alongside adfuller(), which does not.

Three measurements here, in increasing order of how much they hurt.

  1. The size of the test, by simulation. Feed the screen pairs of
     independent random walks - no relationship of any kind - and count how
     often it says "cointegrated at 5%". A correctly sized test says 5%.

  2. The same 4,950 real pairs, re-tested with the correct critical values.

  3. Multiple testing on top of that. meta.json already records that
     Bonferroni leaves one pair standing. Bonferroni is the blunt version;
     Benjamini-Hochberg controls the false discovery rate instead, which is
     the quantity a trader actually cares about - not "is any of this real"
     but "what share of what I am about to trade is noise".
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller, coint

from scan import (ADF_MAXLAG, FORMATION, MIN_FORMATION_DAYS, TRADING,
                  beta_and_pvalue, clean, load_prices, run_pair, sharpe)
from universe import TICKERS

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SITE_OUT = ROOT.parent / "docs" / "validity.js"
ALPHA = 0.05
N_SIM = 2_000
SIM_SEED = 20260928


def engle_granger_pvalue(y: np.ndarray, x: np.ndarray) -> float:
    """The same two series, tested the way the estimated beta requires.

    coint() runs the identical regression and the identical unit-root test
    on the identical residual. The only thing that changes is which table
    the statistic is looked up in.
    """
    return float(coint(y, x, trend="c", maxlag=ADF_MAXLAG, autolag="aic")[1])


def empirical_size(n_obs: int, n_sim: int = N_SIM, seed: int = SIM_SEED) -> dict:
    """How often each test cries cointegration on two unrelated random walks.

    The simulation is the null hypothesis, stated as code: two independent
    driftless random walks of the same length as the formation window. There
    is no relationship to find. Whatever fraction of these the screen calls
    significant at 5% IS the screen's true size, by definition, and it needs
    no theory to interpret.
    """
    rng = np.random.default_rng(seed)
    naive, proper = [], []
    for _ in range(n_sim):
        y = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n_obs)))
        x = 100 * np.exp(np.cumsum(rng.normal(0, 0.012, n_obs)))
        naive.append(beta_and_pvalue(np.log(y), np.log(x))[1])
        proper.append(engle_granger_pvalue(np.log(y), np.log(x)))
    naive = np.asarray(naive)
    proper = np.asarray(proper)
    return {
        "n_sim": n_sim,
        "n_obs": n_obs,
        "naive_size": float(np.mean(naive < ALPHA)),
        "proper_size": float(np.mean(proper < ALPHA)),
        "naive_size_1pct": float(np.mean(naive < 0.01)),
        "proper_size_1pct": float(np.mean(proper < 0.01)),
        # Where the naive p-value would have to be cut for the test to
        # reject 5% of the time on data with nothing in it.
        "honest_threshold": float(np.quantile(naive, ALPHA)),
        "naive_median": float(np.median(naive)),
        "proper_median": float(np.median(proper)),
    }


def benjamini_hochberg(pvalues: np.ndarray, q: float) -> np.ndarray:
    """Step-up FDR control. Returns the boolean mask of discoveries.

    Sort the p-values, find the largest k with p_(k) <= k*q/n, and reject
    everything up to it. Bonferroni asks "could ANY of these be a false
    positive"; this asks "of the ones I take, what share are false", which
    is the question with money attached - a book of 100 pairs where 10 are
    noise is a business, and Bonferroni would have refused to open it.
    """
    p = np.asarray(pvalues, dtype=float)
    n = len(p)
    order = np.argsort(p)
    thresholds = (np.arange(1, n + 1) / n) * q
    passing = p[order] <= thresholds
    mask = np.zeros(n, dtype=bool)
    if passing.any():
        cutoff = np.max(np.flatnonzero(passing))
        mask[order[: cutoff + 1]] = True
    return mask


def storey_pi0(pvalues: np.ndarray, lam: float = 0.5) -> float:
    """Share of tested pairs with nothing going on, from the p-value histogram.

    Under the null, p-values are uniform, so the ones above lam are all
    null in expectation and there are (1 - lam) * pi0 * n of them. Invert
    that. The estimate is only meaningful on p-values that ARE uniform under
    the null, which is precisely what measurement 1 is checking - so this is
    reported for the corrected p-values and not for the naive ones.
    """
    p = np.asarray(pvalues, dtype=float)
    return float(min(1.0, np.mean(p > lam) / (1.0 - lam)))


def recheck_pairs(prices: pd.DataFrame, scan: pd.DataFrame) -> pd.DataFrame:
    """Re-test every pair in the scan with the correct critical values."""
    logs = {c: np.log(prices[c].to_numpy(float)) for c in prices.columns}
    out = []
    t0 = time.time()
    for i, row in enumerate(scan.itertuples(), 1):
        if row.a not in logs or row.b not in logs:
            continue
        out.append(engle_granger_pvalue(logs[row.a], logs[row.b]))
        if i % 1000 == 0:
            print(f"    {i}/{len(scan)} ({time.time() - t0:.0f}s)")
    res = scan.iloc[: len(out)].copy()
    res["eg_pvalue"] = out
    return res


def trade_survivors(rechecked: pd.DataFrame, q: float = 0.10) -> dict:
    """Trade the FDR survivors against everything else the screen kept.

    control.py already showed the 930 that passed do no better than the 4,020
    that did not. The obvious follow-up: if the screen's problem is that it
    was reading the wrong table, then the handful of pairs that survive a
    correctly sized test AND a false-discovery-rate correction are the only
    ones that were ever evidence of anything, and they should be the ones
    that trade.
    """
    trading = clean(load_prices(TICKERS, *TRADING), 250)
    keep = benjamini_hochberg(rechecked["eg_pvalue"].to_numpy(), q)
    groups = {
        f"survive BH at {q:.0%}": rechecked[keep],
        "passed the old screen, not BH": rechecked[~keep & (rechecked["pvalue"] < ALPHA)],
        "rejected by the old screen": rechecked[rechecked["pvalue"] >= ALPHA],
    }

    out = {}
    for name, group in groups.items():
        sharpes, curves = [], {}
        for row in group.itertuples():
            if row.a not in trading.columns or row.b not in trading.columns:
                continue
            result, _, _, stats = run_pair(trading[row.a], trading[row.b], row.beta)
            sharpes.append(stats["sharpe"])
            curves[f"{row.a}/{row.b}"] = result["ret_net"]
        if not sharpes:
            continue
        book = pd.DataFrame(curves).mean(axis=1)
        out[name] = {"n": len(sharpes),
                     "mean_sharpe": float(np.mean(sharpes)),
                     "book_sharpe": float(sharpe(book))}
    return out


def write_site(payload: dict, rechecked: pd.DataFrame) -> None:
    """docs/validity.js, the way control.py writes docs/control.js.

    The page gets the p-value histograms as well as the summary, because the
    shape is the argument: if the screen were correctly sized, the naive and
    corrected histograms would sit on top of each other.
    """
    edges = np.linspace(0.0, 1.0, 21)
    payload = dict(payload)
    payload["hist"] = {
        "edges": [round(float(e), 3) for e in edges],
        "naive": [int(v) for v in np.histogram(rechecked["pvalue"], bins=edges)[0]],
        "proper": [int(v) for v in np.histogram(rechecked["eg_pvalue"], bins=edges)[0]],
    }
    SITE_OUT.parent.mkdir(exist_ok=True)
    SITE_OUT.write_text("window.VALIDITY = " + json.dumps(payload,
                        separators=(",", ":")) + ";\n", encoding="utf-8")
    print(f"wrote {SITE_OUT}")


def main() -> None:
    scan = pd.read_parquet(RESULTS / "scan.parquet")
    prices = clean(load_prices(TICKERS, *FORMATION), MIN_FORMATION_DAYS)
    n_obs = len(prices)
    print(f"{len(scan):,} pairs, {n_obs:,} formation observations\n")

    print(f"1. Size of the test, {N_SIM:,} simulated pairs of unrelated "
          f"random walks")
    size = empirical_size(n_obs)
    print(f"   {'test':<44} {'rejects at 5%':>14} {'at 1%':>8}")
    print(f"   {'adfuller on the OLS residual (the screen)':<44} "
          f"{size['naive_size']:>13.1%} {size['naive_size_1pct']:>8.1%}")
    print(f"   {'Engle-Granger critical values (coint)':<44} "
          f"{size['proper_size']:>13.1%} {size['proper_size_1pct']:>8.1%}")
    print(f"\n   A 5% test rejects 5% of the time on data with nothing in it.")
    print(f"   The screen's own p-value of 0.05 corresponds to a true 5% test at "
          f"{size['honest_threshold']:.4f}.")

    print(f"\n2. The same {len(scan):,} pairs, correct critical values")
    rechecked = recheck_pairs(prices, scan)
    naive_pass = int((rechecked["pvalue"] < ALPHA).sum())
    proper_pass = int((rechecked["eg_pvalue"] < ALPHA).sum())
    both = int(((rechecked["pvalue"] < ALPHA)
                & (rechecked["eg_pvalue"] < ALPHA)).sum())
    print(f"   passed the screen as written      {naive_pass:>6,}  "
          f"({naive_pass / len(rechecked):.1%})")
    print(f"   pass with the right table         {proper_pass:>6,}  "
          f"({proper_pass / len(rechecked):.1%})")
    print(f"   pass both                         {both:>6,}")
    print(f"   rank correlation of the two p-values "
          f"{rechecked['pvalue'].corr(rechecked['eg_pvalue'], method='spearman'):.3f}")

    print(f"\n3. Multiple testing, on the corrected p-values")
    eg = rechecked["eg_pvalue"].to_numpy()
    pi0 = storey_pi0(eg)
    print(f"   estimated share of pairs with nothing there   {pi0:>6.1%}")
    print(f"   expected false positives at a flat 5%         "
          f"{pi0 * len(eg) * ALPHA:>6.0f}")
    for q in (0.05, 0.10, 0.20):
        keep = benjamini_hochberg(eg, q)
        print(f"   survive Benjamini-Hochberg at FDR {q:.0%}          "
              f"{int(keep.sum()):>6,}")
    bonf = int((eg < ALPHA / len(eg)).sum())
    print(f"   survive Bonferroni                            {bonf:>6,}")

    print(f"\n4. Do the survivors trade any better?")
    traded = trade_survivors(rechecked)
    print(f"   {'group':<36} {'pairs':>6} {'mean sharpe':>12} {'book sharpe':>12}")
    for name, row in traded.items():
        print(f"   {name:<36} {row['n']:>6} {row['mean_sharpe']:>12.2f} "
              f"{row['book_sharpe']:>12.2f}")

    payload = {
        "trading": traded,
        "size": size,
        "naive_pass": naive_pass,
        "proper_pass": proper_pass,
        "pass_both": both,
        "pi0": pi0,
        "bh": {f"{q:.2f}": int(benjamini_hochberg(eg, q).sum())
               for q in (0.05, 0.10, 0.20)},
        "bonferroni": bonf,
        "n_pairs": len(rechecked),
        "n_obs": n_obs,
    }
    (RESULTS / "screen_validity.json").write_text(json.dumps(payload, indent=2))
    write_site(payload, rechecked)
    rechecked[["a", "b", "pvalue", "eg_pvalue"]].to_parquet(
        RESULTS / "screen_validity.parquet", index=False)
    print(f"\nwrote results/screen_validity.json and .parquet")


if __name__ == "__main__":
    main()
