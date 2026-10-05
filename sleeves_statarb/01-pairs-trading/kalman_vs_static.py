

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from scan import ENTRY, EXIT, ZSCORE_WINDOW, COST_BPS, backtest, generate_positions, \
    max_drawdown, rolling_zscore, sharpe

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

PAIR = ("ACN", "LIN")
STATIC_BETA = 0.931742  # from results.parquet - the formation-window OLS fit


def kalman_hedge_ratio(y: pd.Series, x: pd.Series, delta: float = 1e-4,
                       r_var: float | None = None) -> pd.Series:
   
    df = pd.concat([y, x], axis=1).dropna()
    y_vals = df.iloc[:, 0].to_numpy(dtype=float)
    x_vals = df.iloc[:, 1].to_numpy(dtype=float)
    n = len(df)

    if r_var is None:
        r_var = float(sm.OLS(df.iloc[:, 0], sm.add_constant(df.iloc[:, 1])).fit().resid.var())

    beta = 0.0
    P = 1.0
    q = delta / (1.0 - delta)
    betas = np.empty(n)

    for t in range(n):
        P = P + q
        h = x_vals[t]
        innovation = y_vals[t] - h * beta
        S = P * h * h + r_var
        K = P * h / S
        beta = beta + K * innovation
        P = P * (1.0 - K * h)
        betas[t] = beta

    return pd.Series(betas, index=df.index, name="beta")


def run_static(y: pd.Series, x: pd.Series, beta: float) -> dict:
    spread = y - beta * x
    z = rolling_zscore(spread, ZSCORE_WINDOW)
    pos = generate_positions(z, ENTRY, EXIT)
    result = backtest(y, x, pos, COST_BPS)
    prev = pos.shift(1).fillna(0)
    return {
        "total_return": float(result["equity"].iloc[-1] - 1.0),
        "sharpe": sharpe(result["ret_net"]),
        "max_drawdown": max_drawdown(result["equity"]),
        "n_round_trips": int(((prev != 0) & (pos == 0)).sum()),
    }


def run_kalman(y_full: pd.Series, x_full: pd.Series, trading_index: pd.Index) -> dict:
    
    beta_t = kalman_hedge_ratio(y_full, x_full)
    beta_trading = beta_t.reindex(trading_index)

    y, x = y_full.reindex(trading_index), x_full.reindex(trading_index)
    spread = y - beta_trading * x
    z = rolling_zscore(spread, ZSCORE_WINDOW)
    pos = generate_positions(z, ENTRY, EXIT)
    result = backtest(y, x, pos, COST_BPS)
    prev = pos.shift(1).fillna(0)
    return {
        "total_return": float(result["equity"].iloc[-1] - 1.0),
        "sharpe": sharpe(result["ret_net"]),
        "max_drawdown": max_drawdown(result["equity"]),
        "n_round_trips": int(((prev != 0) & (pos == 0)).sum()),
        "beta_start": float(beta_trading.iloc[0]),
        "beta_end": float(beta_trading.iloc[-1]),
    }


def main() -> None:
    formation = pd.read_parquet(RESULTS / "formation.parquet")
    trading = pd.read_parquet(RESULTS / "trading.parquet")
    a, b = PAIR

    y_static, x_static = trading[a], trading[b]
    static_stats = run_static(y_static, x_static, STATIC_BETA)

    y_full = pd.concat([formation[a], trading[a]])
    x_full = pd.concat([formation[b], trading[b]])
    kalman_stats = run_kalman(y_full, x_full, trading.index)

    print(f"Kalman vs. static hedge ratio - {a}/{b}, the one pair that survived "
          f"Bonferroni, traded out of sample on {trading.index[0].date()} - "
          f"{trading.index[-1].date()}:\n")
    print(f"{'':<14} {'total return':>13} {'sharpe':>8} {'max dd':>9} {'round trips':>12}")
    print(f"{'static beta':<14} {static_stats['total_return']:>13.2%} "
          f"{static_stats['sharpe']:>8.2f} {static_stats['max_drawdown']:>9.2%} "
          f"{static_stats['n_round_trips']:>12}")
    print(f"{'kalman beta':<14} {kalman_stats['total_return']:>13.2%} "
          f"{kalman_stats['sharpe']:>8.2f} {kalman_stats['max_drawdown']:>9.2%} "
          f"{kalman_stats['n_round_trips']:>12}")
    print(f"\nstatic beta used throughout: {STATIC_BETA:.4f}")
    print(f"kalman beta: {kalman_stats['beta_start']:.4f} at the start of trading "
          f"-> {kalman_stats['beta_end']:.4f} by the end")


if __name__ == "__main__":
    main()
