"""Re-estimate every quarter instead of once, and see whether it rescues the book.

Usage:  python walkforward.py          (slow: ~100k cointegration tests)
        python walkforward.py --cached (reuse results/walkforward_scan.parquet)

Everything else in this project estimates once. One formation window picks the
pairs, fixes each hedge ratio, and then five years are traded without ever
looking again. RESULTS.md lists that as a stretch goal and it is the single
most obvious objection to the whole study, because it is not how anyone runs a
pairs book. A relationship that was cointegrated over 2015-2020 has no
obligation to stay that way, and a hedge ratio estimated in 2020 is a statement
about 2020.

So: every quarter, re-scan all 4,950 pairs on the trailing three years, keep
whatever passes, re-estimate its hedge ratio on that same window, and trade the
resulting book for one quarter. Nothing ever sees data from its own quarter.
Five variants, from the frozen original to fully refit, all on the identical
2021-2025 trading period so the numbers sit next to the ones already in
RESULTS.md.

The question this answers is narrow and worth being precise about. It is not
"does pairs trading work". It is "was the frozen-estimate design responsible
for the result", which is the objection a reader should raise, and which can be
settled rather than argued.
"""

from __future__ import annotations

import argparse
import json
import time
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd

from scan import (COST_BPS, ENTRY, EXIT, TRADING_DAYS, ZSCORE_WINDOW,
                  beta_and_pvalue, generate_positions, max_drawdown,
                  rolling_zscore, sharpe)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
SCAN_CACHE = RESULTS / "walkforward_scan.parquet"
OUT = RESULTS / "walkforward.json"
# Same arrangement as portfolio.py: the project site reads its numbers from a
# file this script writes, rather than having them typed into the HTML.
SITE_OUT = ROOT.parent / "docs" / "walkforward.js"

TRAIL_DAYS = 756          # three years of trailing data per re-estimate
PVALUE = 0.05
TIGHT_PVALUE = 0.01
HALF_LIFE_MULTIPLE = 4.0  # z-score window, in half-lives
MIN_WINDOW, MAX_WINDOW = 20, 252


def load_history() -> pd.DataFrame:
    """The formation and trading frames stitched into one continuous series.

    scan.py already downloaded, cleaned and aligned both, so rebuilding them
    from yfinance here would risk a different universe and a different set of
    dropped tickers - and then any difference in the results below could be the
    data rather than the method.
    """
    formation = pd.read_parquet(RESULTS / "formation.parquet")
    trading = pd.read_parquet(RESULTS / "trading.parquet")
    shared = [c for c in formation.columns if c in trading.columns]
    history = pd.concat([formation[shared], trading[shared]]).sort_index()
    return history[~history.index.duplicated(keep="first")].dropna(how="any")


def quarter_starts(index: pd.DatetimeIndex, first: str, trail: int) -> list:
    """First trading day of each quarter that has `trail` days of history."""
    frame = pd.DataFrame(index=index)
    starts = frame.groupby([index.year, index.quarter]).head(1).index
    return [d for d in starts
            if d >= pd.Timestamp(first) and index.get_loc(d) >= trail]


def half_life(spread: np.ndarray) -> float:
    """Ornstein-Uhlenbeck half-life from an AR(1) fit on the spread.

    Regress the change on the level: ds_t = a + b * s_{t-1}. A mean-reverting
    spread has b < 0, and the time to decay halfway back is -ln(2)/ln(1+b).
    Returns nan when b >= 0, which means the fit found no mean reversion at all
    and there is no half-life to report - the caller has to decide what to do
    with that rather than being handed a plausible-looking number.
    """
    level = spread[:-1]
    change = np.diff(spread)
    design = np.column_stack([np.ones(len(level)), level])
    coef, *_ = np.linalg.lstsq(design, change, rcond=None)
    b = coef[1]
    if b >= 0 or 1 + b <= 0:
        return float("nan")
    return float(-np.log(2) / np.log(1 + b))


def scan_all_quarters(history: pd.DataFrame, starts: list) -> pd.DataFrame:
    """Cointegration test and hedge ratio for every pair at every quarter start.

    This is the expensive part: 4,950 pairs times one test per quarter. It is
    cached to parquet because the five variants below all read the same scan,
    and because a reader should be able to re-run the analysis without
    re-running the scan.
    """
    tickers = list(history.columns)
    pairs = list(combinations(tickers, 2))
    values = {t: history[t].to_numpy(dtype=float) for t in tickers}
    positions = {d: history.index.get_loc(d) for d in starts}

    rows = []
    started = time.time()
    for n, date in enumerate(starts, 1):
        end = positions[date]
        window = slice(end - TRAIL_DAYS, end)
        for a, b in pairs:
            beta, pvalue = beta_and_pvalue(values[a][window], values[b][window])
            spread = values[a][window] - beta * values[b][window]
            rows.append({"date": date, "a": a, "b": b, "beta": beta,
                         "pvalue": pvalue, "half_life": half_life(spread)})
        print(f"  {n}/{len(starts)} quarters  ({time.time() - started:.0f}s)")
    return pd.DataFrame(rows)


def pair_quarter_returns(y: np.ndarray, x: np.ndarray, beta: float,
                         warmup_y: np.ndarray, warmup_x: np.ndarray,
                         window: int) -> np.ndarray:
    """Net daily returns for one pair over one quarter.

    The z-score needs `window` observations before the quarter's first day, or
    the first month of every quarter would trade on a half-formed z-score and
    the rebalance dates would show up as a spurious pattern in the results. So
    the spread is built on warmup-plus-quarter and only the quarter is kept.
    The warmup is trailing data, already used for estimation, so it leaks
    nothing forward.
    """
    spread = np.concatenate([warmup_y - beta * warmup_x, y - beta * x])
    series = pd.Series(spread)
    z = rolling_zscore(series, window)
    pos = generate_positions(z, ENTRY, EXIT)

    n_warmup = len(warmup_y)
    prices_y = pd.Series(np.concatenate([warmup_y, y]))
    prices_x = pd.Series(np.concatenate([warmup_x, x]))
    gross = (pos.shift(1) * (prices_y.pct_change() - prices_x.pct_change()) / 2.0)
    turnover = pos.diff().abs()
    net = (gross - turnover * (COST_BPS / 10_000.0)).fillna(0.0)
    return net.to_numpy()[n_warmup:]


def run_variant(history: pd.DataFrame, scan: pd.DataFrame, starts: list,
                label: str, refit_beta: bool, reselect: bool,
                pvalue: float = PVALUE, use_half_life: bool = False,
                frozen: pd.DataFrame | None = None) -> dict:
    """One configuration of the same book, quarter by quarter.

    refit_beta : re-estimate the hedge ratio on the trailing window each quarter
    reselect   : re-run the cointegration screen each quarter, instead of
                 trading the pairs the original 2015-2020 formation window chose
    use_half_life : set each pair's z-score window from its own measured
                 half-life instead of the fixed 60 bars used everywhere else
    """
    values = {t: history[t].to_numpy(dtype=float) for t in history.columns}
    location = {d: history.index.get_loc(d) for d in starts}
    frozen_map = ({(r.a, r.b): r.beta for r in frozen.itertuples()}
                  if frozen is not None else {})

    daily: list[np.ndarray] = []
    dates: list[pd.Timestamp] = []
    counts: list[int] = []
    churn: list[float] = []
    previous: set = set()

    for n, date in enumerate(starts):
        start = location[date]
        stop = location[starts[n + 1]] if n + 1 < len(starts) else len(history)
        quarter = scan[scan["date"] == date]

        chosen = (quarter[quarter["pvalue"] <= pvalue] if reselect
                  else quarter[[(r.a, r.b) in frozen_map
                                for r in quarter.itertuples()]])
        if chosen.empty:
            continue

        book = []
        for row in chosen.itertuples():
            beta = row.beta if refit_beta else frozen_map[(row.a, row.b)]
            if use_half_life and np.isfinite(row.half_life):
                window = int(np.clip(round(HALF_LIFE_MULTIPLE * row.half_life),
                                     MIN_WINDOW, MAX_WINDOW))
            else:
                window = ZSCORE_WINDOW
            warmup = slice(max(start - window - 1, 0), start)
            book.append(pair_quarter_returns(
                values[row.a][start:stop], values[row.b][start:stop], beta,
                values[row.a][warmup], values[row.b][warmup], window))

        held = {(r.a, r.b) for r in chosen.itertuples()}
        churn.append(len(held ^ previous) / max(len(held | previous), 1))
        previous = held

        daily.append(np.mean(np.vstack(book), axis=0))   # equal weight
        dates.extend(history.index[start:stop])
        counts.append(len(book))

    returns = pd.Series(np.concatenate(daily), index=pd.DatetimeIndex(dates))
    equity = (1 + returns).cumprod()
    return {
        "label": label,
        "total_return": float(equity.iloc[-1] - 1),
        "sharpe": sharpe(returns),
        "vol": float(returns.std(ddof=1) * np.sqrt(TRADING_DAYS)),
        "max_drawdown": max_drawdown(equity),
        "mean_pairs": float(np.mean(counts)),
        "min_pairs": int(np.min(counts)),
        "max_pairs": int(np.max(counts)),
        "roster_churn": float(np.mean(churn[1:])) if len(churn) > 1 else 0.0,
        "quarters": len(counts),
        "equity": equity,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cached", action="store_true",
                        help="reuse the cached quarterly scan instead of redoing it")
    parser.add_argument("--first", default="2021-01-01",
                        help="first quarter to trade (default matches scan.py's "
                             "trading window, so results are comparable)")
    args = parser.parse_args()

    history = load_history()
    starts = quarter_starts(history.index, args.first, TRAIL_DAYS)
    print(f"History {history.index[0].date()} -> {history.index[-1].date()}, "
          f"{history.shape[1]} tickers")
    print(f"{len(starts)} quarterly re-estimates, "
          f"{starts[0].date()} -> {starts[-1].date()}, "
          f"{TRAIL_DAYS}-day trailing window\n")

    if args.cached and SCAN_CACHE.exists():
        scan = pd.read_parquet(SCAN_CACHE)
        print(f"Loaded cached scan: {len(scan):,} pair-quarters\n")
    else:
        print(f"Scanning {len(starts) * 4950:,} pair-quarters "
              f"(this takes a while)...")
        scan = scan_all_quarters(history, starts)
        scan.to_parquet(SCAN_CACHE, index=False)
        print()

    frozen = pd.read_parquet(RESULTS / "results.parquet")[["a", "b", "beta"]]

    variants = [
        run_variant(history, scan, starts, "frozen selection, frozen beta",
                    refit_beta=False, reselect=False, frozen=frozen),
        run_variant(history, scan, starts, "frozen selection, quarterly beta",
                    refit_beta=True, reselect=False, frozen=frozen),
        run_variant(history, scan, starts, "quarterly selection and beta",
                    refit_beta=True, reselect=True),
        run_variant(history, scan, starts, "quarterly, p <= 0.01",
                    refit_beta=True, reselect=True, pvalue=TIGHT_PVALUE),
        run_variant(history, scan, starts, "quarterly, z-window from half-life",
                    refit_beta=True, reselect=True, use_half_life=True),
    ]

    print(f"{'variant':<36} {'return':>8} {'sharpe':>7} {'vol':>7} "
          f"{'max dd':>8} {'pairs':>7} {'churn':>7}")
    for v in variants:
        print(f"{v['label']:<36} {v['total_return']:>8.2%} {v['sharpe']:>7.2f} "
              f"{v['vol']:>7.2%} {v['max_drawdown']:>8.2%} "
              f"{v['mean_pairs']:>7.0f} {v['roster_churn']:>7.1%}")

    print("\n'pairs' is the average number held per quarter; 'churn' is the share")
    print("of the roster that changes at each re-estimate.")

    survivors = scan[scan["pvalue"] <= PVALUE]
    per_quarter = survivors.groupby("date").size()
    print(f"\nPairs passing at 5% per quarter: min {per_quarter.min()}, "
          f"median {int(per_quarter.median())}, max {per_quarter.max()} "
          f"(out of 4,950)")

    # How stable is the screen's verdict? A pair that is cointegrated for real
    # should keep passing; one that passed by luck should not. This is the same
    # question the p-value-quintile test in RESULTS.md asks, from the other end.
    counts = survivors.groupby(["a", "b"]).size()
    all_pairs = scan.groupby(["a", "b"]).size().index
    never = len(all_pairs) - len(counts)
    always = int((counts == len(starts)).sum())
    print(f"Of 4,950 pairs: {never:,} never pass, {always} pass in all "
          f"{len(starts)} quarters, "
          f"{len(counts) - always:,} pass in some but not all")
    print(f"Median quarters passed, among pairs that ever pass: "
          f"{int(counts.median())} of {len(starts)}")

    payload = {v["label"]: {k: v[k] for k in v if k != "equity"} for v in variants}
    payload["per_quarter"] = [
        {"date": str(d.date()), "passed": int(per_quarter.get(d, 0))}
        for d in starts]
    # How many of the 20 quarters each pair passed in. Index 0 is the pairs
    # that never pass; the last index is the pairs that pass every time.
    payload["quarters_passed"] = [
        never if k == 0 else int((counts == k).sum())
        for k in range(len(starts) + 1)]
    frame = pd.DataFrame({v["label"]: v["equity"] for v in variants})
    payload["equity"] = {
        "dates": [d.strftime("%Y-%m-%d") for d in frame.index],
        "series": {c: [round(float(x), 5) for x in frame[c]] for c in frame},
    }
    payload["meta"] = {
        "first_quarter": str(starts[0].date()),
        "last_quarter": str(starts[-1].date()),
        "quarters": len(starts),
        "trail_days": TRAIL_DAYS,
        "pairs_per_quarter": {"min": int(per_quarter.min()),
                              "median": int(per_quarter.median()),
                              "max": int(per_quarter.max())},
        "never_pass": never, "always_pass": always,
    }
    OUT.write_text(json.dumps(payload, indent=2))
    print(f"\nWrote {OUT}")

    if SITE_OUT.parent.exists():
        SITE_OUT.write_text(
            "window.WALKFORWARD = " + json.dumps(payload, separators=(",", ":")) + ";\n",
            encoding="utf-8")
        print(f"Wrote {SITE_OUT} ({SITE_OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
