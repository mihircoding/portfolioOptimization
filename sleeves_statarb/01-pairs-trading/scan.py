
from __future__ import annotations

import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller

from universe import TICKERS, same_sector, sector

ROOT = Path(__file__).resolve().parent
CACHE = ROOT / "data"
RESULTS = ROOT / "results"

FORMATION = ("2013-01-01", "2020-12-31")
TRADING = ("2021-01-01", "2025-12-31")

MIN_FORMATION_DAYS = 1000   # ~4 years; below this the ADF test doesnt wokr
ADF_MAXLAG = 12             # uncapped AIC picks ~23 lags and is 4x slower. barely changes p
ZSCORE_WINDOW = 60
ENTRY, EXIT = 2.0, 0.5
COST_BPS = 5.0
TRADING_DAYS = 252


def load_prices(tickers: list[str], start: str, end: str) -> pd.DataFrame:
    CACHE.mkdir(exist_ok=True)
    tag = f"universe_{len(tickers)}_{start}_{end}.csv"
    cache_file = CACHE / tag

    if cache_file.exists():
        return pd.read_csv(cache_file, index_col=0, parse_dates=True)

    import yfinance as yf

    raw = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)
    prices = raw["Close"]
    if isinstance(prices, pd.Series):
        prices = prices.to_frame(tickers[0])
    prices.to_csv(cache_file)
    return prices


def clean(prices: pd.DataFrame, min_days: int) -> pd.DataFrame:
    
    keep = [c for c in prices.columns if prices[c].notna().sum() >= min_days]
    dropped = sorted(set(prices.columns) - set(keep))
    if dropped:
        print(f"  dropped {len(dropped)} tickers with < {min_days} days: {', '.join(dropped)}")
    return prices[keep].dropna(how="any")




def hedge_ratio(y: pd.Series, x: pd.Series) -> float:
    return float(sm.OLS(y, sm.add_constant(x)).fit().params.iloc[1])


def beta_and_pvalue(y: np.ndarray, x: np.ndarray) -> tuple[float, float]:
    # same maths as the notebook, but one regression instead of two
    design = np.column_stack([np.ones(len(x)), x])
    coef, *_ = np.linalg.lstsq(design, y, rcond=None)
    beta = float(coef[1])
    spread = y - design @ coef
    pvalue = float(adfuller(spread, regression="c", autolag="AIC", maxlag=ADF_MAXLAG)[1])
    return beta, pvalue


# signals + backtest

def rolling_zscore(spread: pd.Series, window: int = ZSCORE_WINDOW) -> pd.Series:
    r = spread.rolling(window)
    return (spread - r.mean()) / r.std()


def generate_positions(z: pd.Series, entry: float = ENTRY, exit: float = EXIT) -> pd.Series:
    out = np.zeros(len(z), dtype=int)
    state = 0
    for i, v in enumerate(z.to_numpy(dtype=float)):
        if np.isnan(v):
            state = 0
        elif state == 0:
            if v > entry:
                state = -1
            elif v < -entry:
                state = 1
        elif state == 1:
            if v >= -exit:
                state = 0
        else:
            if v <= exit:
                state = 0
        out[i] = state
    return pd.Series(out, index=z.index, dtype=int)


def backtest(y: pd.Series, x: pd.Series, positions: pd.Series,
             cost_bps: float = COST_BPS) -> pd.DataFrame:
    lagged = positions.shift(1)                       # decide today, trade tomorrow
    ret = (lagged * (y.pct_change() - x.pct_change()) / 2.0).fillna(0.0)

    turnover = positions.diff()
    turnover.iloc[0] = positions.iloc[0]
    ret_net = ret - turnover.abs() * (cost_bps / 10_000.0)

    return pd.DataFrame({"ret": ret, "ret_net": ret_net,
                         "equity": (1.0 + ret_net).cumprod()}, index=y.index)


def sharpe(r: pd.Series) -> float:
    r = r.dropna()
    if len(r) < 2:
        return 0.0
    sd = r.std(ddof=1)
    return 0.0 if sd == 0 or np.isnan(sd) else float(r.mean() / sd * np.sqrt(TRADING_DAYS))


def max_drawdown(equity: pd.Series) -> float:
    eq = equity.dropna()
    return 0.0 if eq.empty else float((eq / eq.cummax() - 1.0).min())


def run_pair(y: pd.Series, x: pd.Series, beta: float,
             window: int = ZSCORE_WINDOW, entry: float = ENTRY,
             exit: float = EXIT, cost_bps: float = COST_BPS) -> tuple[pd.DataFrame, pd.Series, pd.Series, dict]:
    spread = y - beta * x
    z = rolling_zscore(spread, window)
    pos = generate_positions(z, entry, exit)
    result = backtest(y, x, pos, cost_bps)

    prev = pos.shift(1).fillna(0)
    stats = {
        "total_return": float(result["equity"].iloc[-1] - 1.0),
        "sharpe": sharpe(result["ret_net"]),
        "max_drawdown": max_drawdown(result["equity"]),
        "n_round_trips": int(((prev != 0) & (pos == 0)).sum()),
    }
    return result, z, pos, stats


def main() -> None:
    RESULTS.mkdir(exist_ok=True)
    t0 = time.time()

    print(f"Downloading {len(TICKERS)} tickers...")
    formation_raw = load_prices(TICKERS, *FORMATION)
    trading_raw = load_prices(TICKERS, *TRADING)

    print("Cleaning formation window...")
    formation = clean(formation_raw, MIN_FORMATION_DAYS)
    tickers = [t for t in formation.columns if t in trading_raw.columns]
    formation = formation[tickers]
    trading = trading_raw[tickers].dropna(how="any")

    n = len(tickers)
    n_tests = n * (n - 1) // 2
    print(f"  {n} tickers survive | formation {formation.shape} | trading {trading.shape}")
    print(f"  {n_tests} pair tests -> ~{n_tests * 0.05:.0f} expected false positives at 5%")
    print(f"  Bonferroni threshold = 0.05/{n_tests} = {0.05 / n_tests:.2e}")

    print("\nScanning (this is the slow part)...")
    rows = []
    arrays = {t: formation[t].to_numpy(dtype=float) for t in tickers}
    for k, (a, b) in enumerate(combinations(tickers, 2), 1):
        if k % 500 == 0:
            print(f"  {k}/{n_tests}  ({time.time() - t0:.0f}s)")
        beta, pvalue = beta_and_pvalue(arrays[a], arrays[b])
        rows.append({"a": a, "b": b, "beta": beta, "pvalue": pvalue})

    scan = pd.DataFrame(rows)
    scan["same_sector"] = [same_sector(a, b) for a, b in zip(scan["a"], scan["b"])]
    scan["sector_a"] = scan["a"].map(sector)
    scan["sector_b"] = scan["b"].map(sector)
    print(f"  scan finished in {time.time() - t0:.0f}s")

    passed = scan[scan["pvalue"] <= 0.05].copy()
    print(f"\n{len(passed)} pairs pass at 5% ({len(passed) / n_tests:.1%} of tests)")
    print(f"{(scan['pvalue'] <= 0.05 / n_tests).sum()} pairs survive Bonferroni")

    print(f"\nBacktesting all {len(passed)} survivors out of sample...")
    stats_rows = []
    for _, row in passed.iterrows():
        a, b = row["a"], row["b"]
        y, x = trading[a], trading[b]
        _, _, _, stats = run_pair(y, x, row["beta"])
        _, oos_p = beta_and_pvalue(y.to_numpy(dtype=float), x.to_numpy(dtype=float))
        stats_rows.append({**row.to_dict(), **stats, "pvalue_oos": oos_p})

    results = pd.DataFrame(stats_rows).sort_values("pvalue").reset_index(drop=True)

    scan.to_parquet(RESULTS / "scan.parquet", index=False)
    results.to_parquet(RESULTS / "results.parquet", index=False)
    formation.to_parquet(RESULTS / "formation.parquet")
    trading.to_parquet(RESULTS / "trading.parquet")

    meta = {
        "formation": FORMATION, "trading": TRADING,
        "n_tickers": n, "n_tests": n_tests,
        "n_passed": int(len(passed)),
        "n_bonferroni": int((scan["pvalue"] <= 0.05 / n_tests).sum()),
        "expected_false_positives": round(n_tests * 0.05, 1),
        "bonferroni_threshold": 0.05 / n_tests,
        "zscore_window": ZSCORE_WINDOW, "entry": ENTRY, "exit": EXIT,
        "cost_bps": COST_BPS,
        "tickers": tickers,
        "elapsed_seconds": round(time.time() - t0, 1),
    }
    (RESULTS / "meta.json").write_text(json.dumps(meta, indent=2))

    same = results[results["same_sector"]]
    cross = results[~results["same_sector"]]
    print("\n--- headline ---")
    print(f"  same-sector  : {len(same):>4} pairs | mean Sharpe {same['sharpe'].mean():+.3f}")
    print(f"  cross-sector : {len(cross):>4} pairs | mean Sharpe {cross['sharpe'].mean():+.3f}")
    print(f"\n  done in {time.time() - t0:.0f}s -> {RESULTS}")


if __name__ == "__main__":
    main()
