"""Does a better covariance matrix matter? Only when N stops being small.

Usage:  python factor_study.py

RESULTS.md has said from the start that five assets is an easy problem, and
that shrinkage and factor models earn their keep somewhere else. This script is
"somewhere else": 50 US large caps, so the 1,275 free parameters in a
covariance matrix are finally comparable to the amount of data estimating them.

The test is deliberately not "which portfolio earned more." Returns over any
one sample are mostly luck. The test is whether the risk number each estimator
printed at the time turned out to be true - that is measurable at every
rebalance, and it is the number a risk manager is actually paid to get right.

Everything is out of sample: estimate on a trailing window, hold for a quarter,
never look forward.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from src.costs import COST_LEVELS_BPS, DEFAULT_COST_BPS, hold, net_returns
from src.factor_model import (factor_variance_share, marchenko_pastur_edge,
                              num_significant_factors, pca_factor_covariance)
from src.optimizer import min_variance_weights, min_variance_weights_analytic
from src.returns import TRADING_DAYS, daily_returns
from src.risk_parity import ledoit_wolf_alpha, shrink_covariance

# 50 US large caps, all trading since 2006 so there are no survivorship gaps
# inside the sample. Spread across sectors on purpose: a single-sector universe
# is effectively a one-factor universe, which would hand the factor model a win
# it hadn't earned.
UNIVERSE = [
    "AAPL", "MSFT", "ORCL", "CSCO", "INTC", "IBM", "TXN", "QCOM", "ADBE",
    "JNJ", "MRK", "PFE", "ABT", "UNH", "LLY", "AMGN", "GILD",
    "JPM", "BAC", "GS", "MS", "AXP", "USB",
    "XOM", "CVX",
    "PG", "KO", "PEP", "WMT", "COST", "TGT", "MCD", "SBUX", "NKE", "HD", "LOW",
    "CAT", "BA", "MMM", "HON", "GE", "RTX", "LMT", "UPS",
    "DIS", "VZ", "T",
    "SO", "DUK", "NEE",
]
SMALL_UNIVERSE = ["SPY", "EFA", "AGG", "GLD", "VNQ"]   # the rest of the repo

START, END = "2006-01-01", "2024-12-31"
HOLD_DAYS = 63            # one quarter
WINDOWS = [126, 252, 504, 1008]
CONTROL = "Equal weight"

# No-trade bands for section 6, in weight points. 1pt is half the average 1/N
# weight, the obvious first guess; 5pt is about the size of a typical
# unconstrained min-variance position. BAND_SWEEP shows everything in between
# and past it, so neither choice has to be taken on trust.
BANDS = [0.0, 0.01, 0.05]
BAND_SWEEP = [0.0, 0.005, 0.01, 0.02, 0.05, 0.10, 0.20]


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def load(tickers: list[str]) -> pd.DataFrame:
    px = yf.download(tickers, start=START, end=END, progress=False,
                     auto_adjust=True)["Close"]
    px = px[tickers].dropna()
    if px.empty:
        raise RuntimeError("no price data came back")
    return px


def standard_estimators(train: np.ndarray) -> dict:
    """The four covariance estimates compared throughout, annualized.

    The factor model's k comes from the Marchenko-Pastur cutoff on this window's
    own data, so it is re-chosen at every rebalance and never hand-picked.
    """
    sample = np.cov(train, rowvar=False)
    return {
        "Equal weight (control)": sample * TRADING_DAYS,
        "Sample": sample * TRADING_DAYS,
        "Shrunk (Ledoit-Wolf)": shrink_covariance(
            sample, ledoit_wolf_alpha(train)) * TRADING_DAYS,
        "Factor (k from MP)": pca_factor_covariance(
            train, num_significant_factors(train)) * TRADING_DAYS,
        "Diagonal only": np.diag(np.diag(sample)) * TRADING_DAYS,
    }


def k_sweep_estimators(ks):
    """Estimator set for the k sweep: the same factor model at several k."""
    def build(train: np.ndarray) -> dict:
        return {f"k = {k:>2}": pca_factor_covariance(train, k) * TRADING_DAYS
                for k in ks}
    return build


def backtest(prices: pd.DataFrame, window: int, build=standard_estimators,
             long_only: bool = False) -> pd.DataFrame:
    """Roll a minimum-variance portfolio under each covariance estimate.

    Records, per rebalance, the volatility the estimate *predicted* and the
    returns actually realized over the following quarter. Those two columns are
    the point. An estimate that predicts 7% and delivers 15% has not made a
    small error; it has understated risk by a factor of two, and it did so
    because the optimizer deliberately sought out the directions in which it was
    most wrong. A min-variance portfolio is the most unforgiving possible test
    of a covariance matrix for exactly that reason.
    """
    rets = daily_returns(prices)
    values = rets.values

    realized: dict[str, list] = {}
    predicted: dict[str, list] = {}
    turnover: dict[str, list] = {}
    shorts: dict[str, list] = {}
    weights: dict[str, list] = {}       # (start index, w) per rebalance
    prev: dict[str, np.ndarray] = {}
    k_chosen: list[int] = []

    for start in range(window, len(values) - HOLD_DAYS + 1, HOLD_DAYS):
        train = values[start - window:start]
        test = values[start:start + HOLD_DAYS]
        k_chosen.append(num_significant_factors(train))

        for name, cov in build(train).items():
            if name.startswith(CONTROL):
                # Fixed weights, priced with the sample covariance. This row
                # does no optimization at all, so whatever gap it shows between
                # predicted and realized is not estimation error being exploited
                # - it is volatility clustering, the fact that the quarter after
                # a calm window is often not calm. Every other row has to beat
                # this one before any of its miss can be blamed on the matrix.
                w = np.full(cov.shape[0], 1.0 / cov.shape[0])
            else:
                try:
                    w = (min_variance_weights(cov, long_only=True) if long_only
                         else min_variance_weights_analytic(cov))
                except (np.linalg.LinAlgError, RuntimeError):
                    continue
            predicted.setdefault(name, []).append(float(np.sqrt(w @ cov @ w)))
            realized.setdefault(name, []).extend((test @ w).tolist())
            shorts.setdefault(name, []).append(float(-np.minimum(w, 0).sum()))
            weights.setdefault(name, []).append((start, w))
            if name in prev:
                turnover.setdefault(name, []).append(
                    float(np.abs(w - prev[name]).sum() / 2))
            prev[name] = w

    rows = []
    for name, series in realized.items():
        r = np.asarray(series)
        rows.append({
            "estimator": name,
            "predicted vol": float(np.mean(predicted[name])),
            "realized vol": float(r.std(ddof=1) * np.sqrt(TRADING_DAYS)),
            "ann return": float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1),
            "turnover": float(np.mean(turnover.get(name, [0.0]))),
            "short": float(np.mean(shorts[name])),
        })
    out = pd.DataFrame(rows)
    out["ratio"] = out["realized vol"] / out["predicted vol"]
    control = out.loc[out["estimator"].str.startswith(CONTROL), "ratio"]
    out.attrs["control ratio"] = float(control.iloc[0]) if len(control) else None
    out.attrs["k"] = k_chosen
    out.attrs["rebalances"] = len(k_chosen)
    out.attrs["weights"] = weights
    return out


def cost_study(prices: pd.DataFrame, table: pd.DataFrame, bands=BANDS,
               names=None, suffix: str = "") -> pd.DataFrame:
    """Re-run backtest()'s target weights with drift, real trades and costs.

    backtest() multiplies each day's returns by the target weights, which is a
    book reset to target every day for free, and measures turnover target to
    target. Here the book drifts through the quarter, each rebalance trades
    from where it drifted to, and the cost of that trade is charged at
    each of COST_LEVELS_BPS. Each target is also run through a no-trade band
    (src/costs.py :: band_rebalance) at every width in `bands`.

    Sharpe is annualized from daily returns, risk-free rate zero. Drag is gross
    minus net annual return. Turnover is one-way, per quarterly rebalance, so
    it reads against backtest()'s column.
    """
    values = daily_returns(prices).values
    rows = []
    for name, schedule in table.attrs["weights"].items():
        if names is not None and name not in names:
            continue
        periods = [(w, values[s:s + HOLD_DAYS]) for s, w in schedule]
        for band in bands:
            gross, traded, starts = hold(periods, band=band)
            row = {"estimator": name + suffix, "band": band,
                   "turnover": float(traded.mean() / 2)}
            for c in (0,) + tuple(COST_LEVELS_BPS):
                r = net_returns(gross, traded, starts, c) if c else gross
                vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
                ann = float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
                row[f"sharpe {c}"] = float(r.mean() * TRADING_DAYS / vol)
                row[f"ann return {c}"] = ann
                if c == 0:
                    row["realized vol"] = vol
                else:
                    row[f"drag {c}"] = row["ann return 0"] - ann
            rows.append(row)
    return pd.DataFrame(rows)


def cost_tables(prices, unconstrained, long_only, small=None, small_table=None):
    """Section 6's main table: the rows worth charging for, at each of BANDS.

    Arguments are the 252-day backtest() tables section 2, 4 and 5 already
    computed, so nothing is re-optimized here.
    """
    parts = [
        cost_study(prices, unconstrained,
                   names=["Equal weight (control)", "Sample", "Shrunk (Ledoit-Wolf)"]),
        cost_study(prices, long_only, names=["Sample"], suffix=", long-only"),
    ]
    if small is not None:
        parts.append(cost_study(small, small_table, names=["Sample"],
                                suffix=", 5 ETFs"))
    return pd.concat(parts, ignore_index=True)


def band_label(band: float) -> str:
    return "to target" if band == 0 else f"{band * 100:g}pt band"


def show_costs(table: pd.DataFrame) -> None:
    print(f"{'estimator':<24} {'rebalance':<16} {'turnover':>9} {'real vol':>9} "
          f"{'gross':>6} " + " ".join(f"{f'@{c}bp':>6}" for c in COST_LEVELS_BPS)
          + f" {'drag@' + f'{DEFAULT_COST_BPS:g}' + 'bp':>11}")
    for _, r in table.iterrows():
        print(f"{r['estimator']:<24} {band_label(r['band']):<16} "
              f"{r['turnover']:>9.1%} {r['realized vol']:>9.2%} {r['sharpe 0']:>6.2f} "
              + " ".join(f"{r[f'sharpe {c}']:>6.2f}" for c in COST_LEVELS_BPS)
              + f" {r[f'drag {DEFAULT_COST_BPS:g}'] * 1e4:>9.0f}bp")


def show(table: pd.DataFrame) -> None:
    print(f"{'estimator':<22} {'predicted':>10} {'realized':>9} {'ratio':>7} "
          f"{'ann ret':>8} {'turnover':>9} {'short':>7}")
    for _, r in table.iterrows():
        print(f"{r['estimator']:<22} {r['predicted vol']:>10.2%} "
              f"{r['realized vol']:>9.2%} {r['ratio']:>7.2f} "
              f"{r['ann return']:>8.2%} {r['turnover']:>9.1%} {r['short']:>7.0%}")


def main() -> None:
    prices = load(UNIVERSE)
    rets = daily_returns(prices)
    n = prices.shape[1]

    section(f"1. The counting problem ({n} assets, "
            f"{prices.index[0].date()} -> {prices.index[-1].date()})")
    print(f"free parameters in a {n}x{n} covariance matrix: {n * (n + 1) // 2}\n")
    print(f"{'window':>8} {'obs/parameter':>14} {'noise edge':>11} "
          f"{'factors above':>14} {'their share of var':>19}")
    for w in WINDOWS:
        k = num_significant_factors(rets.values[-w:])
        print(f"{w:>7}d {w * n / (n * (n + 1) / 2):>14.1f} "
              f"{marchenko_pastur_edge(n, w):>11.2f} {k:>14} "
              f"{factor_variance_share(rets.values[-w:], k):>19.0%}")
    print("\nThe noise edge is Marchenko-Pastur: the largest eigenvalue a correlation")
    print("matrix of this shape produces when the data has no structure at all. At a")
    print("126-day window it sits near 2, so an eigenvalue of 2 means nothing there,")
    print("and means a real factor at 1008 days. Same eigenvalue, opposite verdict,")
    print("purely because of how much data went into the matrix.")

    section("2. Minimum variance, unconstrained, rebalanced quarterly")
    print("'predicted' is the volatility the estimate claimed at the rebalance.")
    print("'realized' is what the portfolio then did. ratio > 1 means the estimate")
    print("understated risk. 'short' is gross short exposure as a share of capital.")
    print("The first row never optimizes - it is the same 1/N portfolio every quarter,")
    print("priced with the sample covariance - so its ratio is the part of the miss")
    print("that has nothing to do with the matrix.\n")
    tables = {}
    for w in WINDOWS:
        table = tables[w] = backtest(prices, w)
        print(f"window = {w} days ({table.attrs['rebalances']} rebalances, "
              f"MP chose k = {min(table.attrs['k'])}-{max(table.attrs['k'])})")
        show(table)
        print()

    section("3. How many factors? The bias-variance tradeoff, measured")
    print("Same test, same windows, only k changes. k = 50 is the sample covariance")
    print("exactly (the model then has no residual left to diagonalize), so this table")
    print("contains section 2's first row as its last row and is a continuum, not a")
    print("set of alternatives.\n")
    for w in (252, 1008):
        table = backtest(prices, w, build=k_sweep_estimators([1, 2, 3, 5, 10, 20, 50]))
        print(f"window = {w} days")
        show(table)
        print()

    section("4. The same test, long-only (Jagannathan-Ma)")
    print("No shorting. Jagannathan and Ma (2003) showed that a no-short constraint")
    print("is algebraically equivalent to shrinking the largest covariance entries -")
    print("so the constraint is itself a risk model, and a crude one applied to a bad")
    print("matrix often beats a good matrix used without it.\n")
    long_only = backtest(prices, 252, long_only=True)
    print("window = 252 days")
    show(long_only)
    print()

    section("5. The same five ETFs this repo uses everywhere else")
    print("Five assets means 15 parameters, which 126 days estimates comfortably, so")
    print("there should be nothing here for any of this to fix.\n")
    small = load(SMALL_UNIVERSE)
    small_tables = {}
    for w in (126, 252):
        table = small_tables[w] = backtest(small, w)
        print(f"window = {w} days")
        show(table)
        print()

    section("6. What the turnover costs, and a no-trade band")
    print("Sections 2-5 hold the target weights fixed for the whole quarter, which is")
    print("a book reset to target every day at no cost. Here the book drifts through")
    print("each quarter, each rebalance trades from the drifted weights, and every")
    print("dollar traded pays the cost shown. A band leaves any position within that")
    print("many weight points of its target alone and trades the rest only back to")
    print("the band edge. 252-day window throughout.\n")
    show_costs(cost_tables(prices, tables[252], long_only, small, small_tables[252]))
    print("\nturnover is one-way per quarter; Sharpe from daily returns, rf = 0;")
    print(f"drag is gross minus net annual return at {DEFAULT_COST_BPS:g} bps.\n")
    print("Band width, sample covariance, unconstrained, 50 stocks:")
    show_costs(cost_study(prices, tables[252], bands=BAND_SWEEP, names=["Sample"]))


if __name__ == "__main__":
    main()
