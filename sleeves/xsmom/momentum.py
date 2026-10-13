"""The momentum sleeve: long last year's winners, short last year's losers.

Usage:  python -m sleeves.xsmom.momentum
Verify with:  pytest tests/test_xsmom.py

Signal is the Jegadeesh-Titman 12-1: the return from twelve months ago to one
month ago. The most recent month is skipped on purpose, because at a one-month
horizon individual stocks mean-revert, and a signal that includes it is a
momentum signal with a reversal signal subtracted from it. Every paper since
1993 drops that month and so does this.

Portfolio is the plain version. Rank the universe at each month end, go long
the top quintile and short the bottom quintile, equal weight inside each leg,
one dollar long against one dollar short. Hold for a month, re-rank, repeat.
There is a sector-neutral switch that ranks within GICS sector instead of
across the whole list, which is the first thing a desk would do to it, and the
study reports both so that the difference is a number rather than an opinion.

Costs are charged the way src/costs.py charges them everywhere else in this
repository: the book drifts with prices between rebalances, the trade is from
the drifted weights to the new ones, and every dollar traded pays cost_bps.
Momentum turns over a lot - the top and bottom fifths are not the same names
two months running - so this is not a detail. At 10 bps it is about a third
of the gross return.

What the universe is, and what that does to the answer. These are the 102
names in the S&P 100 as of 2025, with prices back to 2013. Every one of them
is a company that was large and alive at the end of the sample, which is
survivorship bias, and it is worth being exact about which way it cuts here.
A momentum short leg is a bet that last year's losers keep losing. In a
universe where every loser recovered enough to still be in the index a decade
later, that leg is fighting the sample. So the bias runs AGAINST this sleeve
on the short side, and roughly neutral on the long side - the opposite of the
direction it runs for a long-only strategy. The number below is more likely
an understatement than an overstatement, which is the less bad way round, and
it is still a caveat rather than an excuse.
"""

from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

from src.costs import DEFAULT_COST_BPS, traded_notional

DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "statarb", "data")
LOOKBACK = 12       # months of return in the signal
SKIP = 1            # most recent months left out of it
QUANTILE = 0.2      # each leg is this fraction of the ranked universe
MIN_NAMES = 20      # fewer ranked names than this and the month is not traded


def load_universe(data_dir: str = DATA_DIR) -> pd.DataFrame:
    """Daily adjusted closes, every universe_* file on disk stitched together.

    The pairs scan downloaded the universe in two date ranges. They do not
    overlap, so concatenation is the whole join. Columns are kept in the
    order of the first file so that nothing depends on a dictionary.
    """
    paths = sorted(glob.glob(os.path.join(data_dir, "universe_*.csv")))
    if not paths:
        raise FileNotFoundError(f"no universe_*.csv under {data_dir}")
    frames = [pd.read_csv(p, index_col=0, parse_dates=True) for p in paths]
    prices = pd.concat(frames).sort_index()
    prices = prices[~prices.index.duplicated(keep="last")]
    return prices.astype(float)


def monthly_prices(prices: pd.DataFrame) -> pd.DataFrame:
    """Last close of each calendar month."""
    return prices.resample("ME").last()


def momentum_signal(monthly: pd.DataFrame, lookback: int = LOOKBACK,
                    skip: int = SKIP) -> pd.DataFrame:
    """Return from `lookback` months ago to `skip` months ago, per name.

    At row t this is price[t - skip] / price[t - lookback] - 1, so the signal
    dated t uses nothing after t and the month t -> t+1 is the one it is traded
    on. A name without a price at either end has no signal that month and is
    simply not ranked, which is how the 2013 start and the names that list
    later (DOW, KHC, PYPL) are handled without a special case.
    """
    return monthly.shift(skip) / monthly.shift(lookback) - 1.0


def target_weights(signal: pd.Series, quantile: float = QUANTILE,
                   sectors: dict[str, str] | None = None,
                   min_names: int = MIN_NAMES) -> pd.Series:
    """Long the top `quantile` of ranked names, short the bottom, dollar neutral.

    Equal weight inside each leg: +1/n_long and -1/n_short, so the book is one
    dollar long against one dollar short regardless of how many names ranked.
    With `sectors` given the ranking is done inside each sector and the legs
    are the union of the per-sector legs, which removes the sector bet that an
    unconditional sort always carries (a year in which energy rallied is a year
    in which the long leg is all energy).

    Returns an all-zero series when fewer than `min_names` ranked, rather than
    trading a quintile of three stocks.
    """
    ranked = signal.dropna()
    if len(ranked) < min_names:
        return pd.Series(0.0, index=signal.index)

    long_names: list[str] = []
    short_names: list[str] = []
    if sectors is None:
        groups = [ranked]
    else:
        labels = ranked.index.map(lambda t: sectors.get(t, "Unknown"))
        groups = [ranked[labels == s] for s in sorted(set(labels))]
    for group in groups:
        k = int(round(len(group) * quantile))
        if k == 0:
            continue
        order = group.sort_values()
        short_names.extend(order.index[:k])
        long_names.extend(order.index[-k:])

    w = pd.Series(0.0, index=signal.index)
    if long_names:
        w[long_names] = 1.0 / len(long_names)
    if short_names:
        w[short_names] = -1.0 / len(short_names)
    return w


def backtest(prices: pd.DataFrame | None = None,
             cost_bps: float = DEFAULT_COST_BPS,
             sectors: dict[str, str] | None = None,
             quantile: float = QUANTILE,
             min_names: int = MIN_NAMES) -> pd.DataFrame:
    """Monthly returns of the sleeve, gross and net, with turnover.

    Each row is one holding month: the weights were set at the previous month
    end from a signal that stops one month before that, and the row's return
    is what those weights earned over the month. A name that has no return in
    a month it was held (the universe file ends, say) earns zero, which is the
    conservative reading of a missing print.

    Turnover is measured from the DRIFTED book, not the previous target. The
    positions compound at their own returns during the month, so the trade
    back to equal weight is larger than the difference between two target
    vectors - and on a long-short book the winners the sleeve is long have
    grown while the losers it is short have shrunk in absolute size, both of
    which have to be trimmed. That is what the cost model in src/costs.py was
    written to count and it is reused here rather than reimplemented.
    """
    prices = load_universe() if prices is None else prices
    monthly = monthly_prices(prices)
    returns = monthly.pct_change(fill_method=None)
    signal = momentum_signal(monthly)

    rows = []
    held = pd.Series(0.0, index=monthly.columns)      # drifted book
    for i in range(1, len(monthly)):
        date = monthly.index[i]
        target = target_weights(signal.iloc[i - 1], quantile=quantile,
                                sectors=sectors, min_names=min_names)
        traded = traded_notional(held.to_numpy(), target.to_numpy())
        cost = cost_bps / 1e4 * traded
        r = returns.iloc[i].fillna(0.0)
        gross = float((target * r).sum())
        rows.append({
            "date": date,
            "gross": gross,
            "net": gross - cost,
            "turnover": traded / 2.0,
            "n_long": int((target > 0).sum()),
            "n_short": int((target < 0).sum()),
        })
        held = target * (1.0 + r)

    frame = pd.DataFrame(rows).set_index("date")
    # The first LOOKBACK months have no signal and sat in cash; they are not
    # a return of the strategy and reporting them would dilute its volatility.
    traded_months = (frame["n_long"] > 0)
    return frame[traded_months]


def summary(frame: pd.DataFrame, column: str = "net") -> dict:
    """Annualized return, volatility and Sharpe, plus the turnover it took."""
    r = frame[column]
    mean = float(r.mean()) * 12
    vol = float(r.std(ddof=1)) * np.sqrt(12)
    return {
        "months": int(len(r)),
        "ann_return": mean,
        "ann_vol": vol,
        "sharpe": mean / vol if vol > 0 else 0.0,
        "worst_month": float(r.min()),
        "hit_rate": float((r > 0).mean()),
        "monthly_turnover": float(frame["turnover"].mean()),
    }


def market_beta(frame: pd.DataFrame, prices: pd.DataFrame | None = None,
                column: str = "net") -> float:
    """Slope of the sleeve on the equal-weight universe return.

    There is no index series in this repository's data, and downloading one
    for a single regression is the kind of dependency that breaks a year
    later. The equal-weight average of the 102 names is the market the sleeve
    is actually trading in, and a dollar-neutral book should load on it at
    close to zero. If it does not, the "market neutral" in the write-up is a
    construction rather than a fact.
    """
    prices = load_universe() if prices is None else prices
    market = monthly_prices(prices).pct_change(fill_method=None).mean(axis=1)
    both = pd.concat([frame[column], market.rename("mkt")], axis=1).dropna()
    cov = np.cov(both.iloc[:, 0], both["mkt"], ddof=1)
    return float(cov[0, 1] / cov[1, 1])


def main() -> None:
    from sleeves.statarb.universe import SP100

    prices = load_universe()
    plain = backtest(prices)
    neutral = backtest(prices, sectors=SP100)
    print(f"Cross-sectional momentum, {plain.index[0]:%Y-%m} to "
          f"{plain.index[-1]:%Y-%m}, {len(plain)} months, "
          f"{DEFAULT_COST_BPS:.0f} bps a side\n")
    print(f"{'':<16} {'ann ret':>8} {'ann vol':>8} {'sharpe':>7} "
          f"{'worst':>8} {'hit':>5} {'turnover':>9} {'beta':>6}")
    for label, frame in (("plain, gross", plain), ("plain, net", plain),
                         ("sector-neutral", neutral)):
        col = "gross" if label.endswith("gross") else "net"
        s = summary(frame, col)
        print(f"{label:<16} {s['ann_return']:>8.2%} {s['ann_vol']:>8.2%} "
              f"{s['sharpe']:>7.2f} {s['worst_month']:>8.2%} "
              f"{s['hit_rate']:>5.0%} {s['monthly_turnover']:>9.1%} "
              f"{market_beta(frame, prices, col):>6.2f}")
    os.makedirs("results", exist_ok=True)
    for name, frame in (("xsmom_monthly", plain),
                        ("xsmom_monthly_sector_neutral", neutral)):
        write_results(frame, f"results/{name}.json")


def write_results(frame: pd.DataFrame, path: str) -> None:
    """The monthly table as JSON, which is what the repository commits.

    CSVs are gitignored here, and the allocator study reads this sleeve's
    output rather than recomputing it, so the file has to be one that is
    checked in - the same reason sleeves/statarb/results/portfolio.json is.
    """
    out = {"columns": list(frame.columns),
           "rows": [[f"{d:%Y-%m-%d}", *[float(v) for v in row]]
                    for d, row in zip(frame.index, frame.to_numpy())]}
    with open(path, "w") as fh:
        json.dump(out, fh, indent=1)


def read_results(path: str) -> pd.DataFrame:
    """Inverse of write_results."""
    with open(path) as fh:
        raw = json.load(fh)
    frame = pd.DataFrame([r[1:] for r in raw["rows"]], columns=raw["columns"],
                         index=pd.to_datetime([r[0] for r in raw["rows"]]))
    frame.index.name = "date"
    return frame


if __name__ == "__main__":
    main()
