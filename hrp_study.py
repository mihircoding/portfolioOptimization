"""HRP against the optimizers, where inverting the matrix starts to hurt.

Usage:  python hrp_study.py

Section 9 of RESULTS.md ends on a result that reads like a shrug: on 50 US
large caps, three different covariance estimators barely separate, and the
constraint - forbidding short sales - does more for realized risk than any
of them. The reason given there is that the damage is done by the
INVERSION, not by the matrix, and that a constraint helps because it stops
the optimizer acting on the inverse's worst directions.

If that explanation is right, it predicts something specific and testable:
a method that never inverts the covariance at all should land near the
long-only optimizers on realized risk without being told to. Hierarchical
risk parity (src/hrp.py) is that method. It uses the correlations to build
a tree and splits money down it, and there is no Sigma inverse anywhere in
the calculation.

Everything is out of sample: a trailing estimation window, held for a
quarter, rolled. The comparisons that matter are

  - predicted vs realized volatility, which is the risk manager's question
    and the one section 9 is built on,
  - turnover, because an allocation that rewrites itself every quarter is
    expensive whatever it predicted,
  - and how each method holds up as the universe grows from 5 to 50, which
    is the axis the whole argument is about.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from src.costs import DEFAULT_COST_BPS, hold, net_returns
from src.hrp import hrp_weights
from src.optimizer import min_variance_weights, min_variance_weights_analytic
from src.returns import TRADING_DAYS, daily_returns
from src.risk_parity import (equal_risk_contribution_weights,
                             inverse_vol_weights, ledoit_wolf_alpha,
                             shrink_covariance)

# The same 50 names as factor_study.py, so the two sections are talking
# about the same universe, plus the 5-ETF book from RESULTS.md's main table.
LARGE = [
    "AAPL", "MSFT", "ORCL", "CSCO", "INTC", "IBM", "TXN", "QCOM", "ADBE",
    "JNJ", "MRK", "PFE", "ABT", "UNH", "LLY", "AMGN", "GILD",
    "JPM", "BAC", "GS", "MS", "AXP", "USB",
    "XOM", "CVX",
    "PG", "KO", "PEP", "WMT", "COST", "TGT", "MCD", "SBUX", "NKE", "HD", "LOW",
    "CAT", "BA", "MMM", "HON", "GE", "RTX", "LMT", "UPS",
    "DIS", "CMCSA", "VZ", "T", "NEE", "DUK",
]
SMALL = ["SPY", "EFA", "AGG", "GLD", "VNQ"]
START, END = "2006-01-01", "2024-12-31"
HOLD_DAYS = 63                      # one quarter
WINDOW_DAYS = 3 * TRADING_DAYS      # three years of trailing data


def load(tickers: list[str]) -> pd.DataFrame:
    px = yf.download(tickers, start=START, end=END, auto_adjust=True,
                     progress=False)["Close"]
    return px.dropna(axis=1, how="any").loc[:, [t for t in tickers
                                                if t in px.columns]].dropna()


def methods(cov: np.ndarray) -> dict:
    """Every allocation the comparison needs, from one covariance matrix.

    Three of these invert it (explicitly or through a solver) and three do
    not. That split, not the names, is what the table is about.
    """
    n = cov.shape[0]
    return {
        "Equal weight": lambda: np.full(n, 1 / n),           # no Sigma at all
        "Inverse vol": lambda: inverse_vol_weights(cov),     # diagonal only
        "Hierarchical risk parity": lambda: hrp_weights(cov),  # correlations
        "Equal risk contribution": lambda: equal_risk_contribution_weights(cov),
        "Min variance (long only)": lambda: min_variance_weights(cov,
                                                                 long_only=True),
        "Min variance (unconstrained)": lambda: min_variance_weights_analytic(cov),
    }


def backtest(prices: pd.DataFrame, window: int = WINDOW_DAYS,
             shrink: float | None = None) -> pd.DataFrame:
    """Roll every method through the same rebalance dates and score them.

    shrink=None uses the sample covariance. Passing a number shrinks towards
    a constant-correlation target first, which is the fair fight for the
    optimizers: the point of this table is not to beat min-variance by
    handing it a bad matrix.
    """
    rets = daily_returns(prices)
    values = rets.values

    realized: dict[str, list] = {}
    predicted: dict[str, list] = {}
    turnover: dict[str, list] = {}
    weights: dict[str, list] = {}
    skipped: dict[str, int] = {}
    prev: dict[str, np.ndarray] = {}
    n_rebal = 0

    for start in range(window, len(values) - HOLD_DAYS + 1, HOLD_DAYS):
        train = values[start - window:start]
        test = values[start:start + HOLD_DAYS]
        sample = np.cov(train, rowvar=False) * TRADING_DAYS
        cov = sample if shrink is None else shrink_covariance(sample, shrink)
        n_rebal += 1

        for name, solve in methods(cov).items():
            try:
                w = solve()
            except (np.linalg.LinAlgError, RuntimeError, ValueError):
                skipped[name] = skipped.get(name, 0) + 1
                continue
            predicted.setdefault(name, []).append(float(np.sqrt(w @ cov @ w)))
            realized.setdefault(name, []).extend((test @ w).tolist())
            weights.setdefault(name, []).append((start, w))
            if name in prev:
                turnover.setdefault(name, []).append(
                    float(np.abs(w - prev[name]).sum() / 2))
            prev[name] = w

    rows = []
    for name, series in realized.items():
        r = np.asarray(series)
        gross = float((1 + r).prod() ** (TRADING_DAYS / len(r)) - 1)
        vol = float(r.std(ddof=1) * np.sqrt(TRADING_DAYS))
        turn = float(np.mean(turnover.get(name, [0.0])))
        rows.append({
            "method": name,
            "predicted vol": float(np.mean(predicted[name])),
            "realized vol": vol,
            "ratio": vol / float(np.mean(predicted[name])),
            "ann return": gross,
            "sharpe": gross / vol if vol else 0.0,
            "turnover": turn,
            "max weight": float(np.mean([w.max() for _, w in weights[name]])),
            "short": float(np.mean([-np.minimum(w, 0).sum()
                                    for _, w in weights[name]])),
            "skipped": skipped.get(name, 0),
        })
    out = pd.DataFrame(rows)
    out.attrs["rebalances"] = n_rebal
    return out


def show(table: pd.DataFrame, title: str) -> None:
    print(f"\n{title}  ({table.attrs['rebalances']} quarterly rebalances)\n")
    print(f"{'method':<30} {'pred':>6} {'real':>6} {'ratio':>6} {'ret':>7} "
          f"{'sharpe':>7} {'turn':>6} {'maxw':>6} {'short':>6}")
    for _, r in table.sort_values("ratio").iterrows():
        print(f"{r['method']:<30} {r['predicted vol']:>6.1%} "
              f"{r['realized vol']:>6.1%} {r['ratio']:>6.2f} "
              f"{r['ann return']:>7.2%} {r['sharpe']:>7.2f} "
              f"{r['turnover']:>6.1%} {r['max weight']:>6.1%} "
              f"{r['short']:>6.1%}")


def concentration(prices: pd.DataFrame, window: int = WINDOW_DAYS) -> None:
    """How many names each method is really holding.

    The effective number of positions, 1 / sum(w^2), is the honest version
    of "diversified". A book of 50 stocks with 60% in three of them is a
    three-stock book wearing a costume, and min-variance produces exactly
    that whenever the covariance says two names nearly cancel.
    """
    rets = daily_returns(prices).values
    counts: dict[str, list] = {}
    for start in range(window, len(rets) - HOLD_DAYS + 1, HOLD_DAYS):
        cov = np.cov(rets[start - window:start], rowvar=False) * TRADING_DAYS
        for name, solve in methods(cov).items():
            try:
                w = solve()
            except (np.linalg.LinAlgError, RuntimeError, ValueError):
                continue
            counts.setdefault(name, []).append(1.0 / float(np.sum(w ** 2)))
    print(f"\nEffective number of positions, out of {rets.shape[1]}\n")
    for name, vals in sorted(counts.items(), key=lambda kv: -np.mean(kv[1])):
        print(f"{name:<30} {np.mean(vals):>6.1f}")


def cost_drag(prices: pd.DataFrame, table: pd.DataFrame) -> None:
    """Turnover priced at this project's default, so the ranking is net."""
    print(f"\nWhat the trading costs at {DEFAULT_COST_BPS} bps a side\n")
    print(f"{'method':<30} {'turnover':>9} {'drag/yr':>9} {'net ret':>9}")
    for _, r in table.sort_values("turnover").iterrows():
        drag = 2 * r["turnover"] * DEFAULT_COST_BPS / 10_000 * 4  # 4 rebalances
        print(f"{r['method']:<30} {r['turnover']:>9.1%} {drag:>9.2%} "
              f"{r['ann return'] - drag:>9.2%}")


def main() -> None:
    print("Downloading...")
    small = load(SMALL)
    large = load(LARGE)
    print(f"  {len(small.columns)} ETFs and {len(large.columns)} stocks, "
          f"{small.index[0].date()} to {small.index[-1].date()}")

    show(backtest(small), "5 ETFs, sample covariance")
    show(backtest(large), "50 stocks, sample covariance")
    show(backtest(large, shrink=0.3), "50 stocks, shrunk covariance (alpha 0.3)")
    concentration(large)
    cost_drag(large, backtest(large))


if __name__ == "__main__":
    main()
