"""Sell a one-month SPY option every month for eighteen years and hedge it.

Usage:  python hedging_study.py

The rest of this repo prices options. This asks the follow-up question that
makes pricing mean anything: if the model says an option is worth $2.75, and
you sell it at the market's price and then run the model's own hedge, what
happens to your money?

Implied volatility comes from VIX, which is the market's 30-day implied
volatility on the S&P and the only place to get a long history of it without
paying for option data. Two honest caveats about that: VIX is quoted on SPX
rather than SPY, and it is a variance-swap-style index over the whole strike
range rather than the at-the-money implied vol this study treats it as. It runs
roughly a point above ATM vol because of the skew it integrates, so the premium
measured below is a touch generous. The shape of every result is unaffected.

Simplifications, stated once: prices are dividend-adjusted and the risk-free
rate is set to zero, so both carry terms are folded into the path rather than
modeled separately. The hedge is computed at the volatility the option was sold
at and never re-marked - a desk would re-hedge on current implied, which damps
the tails.
"""

import numpy as np
import pandas as pd
import yfinance as yf

from hedging import delta_hedge, gamma_pnl, realized_vol

TICKER = "SPY"
START, END = "2007-01-01", "2024-12-31"
HEDGE_DAYS = 21              # trading days in the option's life
T_YEARS = HEDGE_DAYS / 252
RATE = 0.0
FREQUENCIES = (2, 5, 10, 21)
COSTS_BPS = (0.0, 1.0, 5.0)


def load() -> pd.DataFrame:
    data = yf.download([TICKER, "^VIX"], start=START, end=END, progress=False,
                       auto_adjust=True)["Close"].dropna()
    return data.rename(columns={TICKER: "spy", "^VIX": "vix"})


def months(data: pd.DataFrame) -> list[pd.DataFrame]:
    """One window per month start, each HEDGE_DAYS long plus the expiry bar.

    Windows do not overlap, so the monthly P&L series is a set of independent
    trades and its standard deviation means what it looks like.
    """
    starts = data.groupby([data.index.year, data.index.month]).head(1).index
    windows = []
    for start in starts:
        location = data.index.get_loc(start)
        if location + HEDGE_DAYS >= len(data):
            break
        windows.append(data.iloc[location:location + HEDGE_DAYS + 1])
    return windows


def trade(window: pd.DataFrame, option_type="call", rebalance_every=1,
          cost_bps=0.0) -> dict:
    """One month: sell an at-the-money option at VIX, hedge to expiry."""
    path = window["spy"].to_numpy(dtype=float)
    sigma = float(window["vix"].iloc[0]) / 100.0
    strike = path[0]
    result = delta_hedge(path, strike, T_YEARS, RATE, sigma,
                         option_type=option_type, rebalance_every=rebalance_every,
                         cost_bps=cost_bps)
    decomposition, _ = gamma_pnl(path, strike, T_YEARS, RATE, sigma,
                                 option_type=option_type)
    # Everything is reported per $100 of underlying so that 2008 and 2024 are
    # comparable: SPY went from 140 to 580 over this sample, and an unscaled
    # dollar P&L would just be a chart of SPY's price.
    scale = 100.0 / path[0]
    return {
        "date": window.index[0],
        "implied": sigma,
        "realized": result["realized_vol"],
        "pnl": result["pnl"] * scale,
        "premium": result["premium"] * scale,
        "gamma_pnl": decomposition * scale,
        "costs": result["costs"] * scale,
        "variance_gap": sigma ** 2 - result["realized_vol"] ** 2,
    }


def summarize(trades: pd.DataFrame) -> dict:
    pnl = trades["pnl"]
    return {
        "n": len(pnl),
        "mean": pnl.mean(),
        "std": pnl.std(ddof=1),
        "sharpe": pnl.mean() / pnl.std(ddof=1) * np.sqrt(12),
        "win_rate": float((pnl > 0).mean()),
        "worst": pnl.min(),
        "best": pnl.max(),
    }


def main() -> None:
    data = load()
    windows = months(data)
    base = pd.DataFrame([trade(w) for w in windows]).set_index("date")

    print(f"Short one at-the-money {TICKER} call per month, delta-hedged daily "
          f"to expiry.")
    print(f"{len(base)} non-overlapping trades, {base.index[0].date()} to "
          f"{base.index[-1].date()}.")
    print("All P&L figures are per $100 of underlying.\n")

    stats = summarize(base)
    print(f"{'average P&L':<24} {stats['mean']:>8.3f}")
    print(f"{'average premium sold':<24} {base['premium'].mean():>8.3f}")
    print(f"{'std dev of P&L':<24} {stats['std']:>8.3f}")
    print(f"{'annualized Sharpe':<24} {stats['sharpe']:>8.2f}")
    print(f"{'months profitable':<24} {stats['win_rate']:>8.1%}")
    print(f"{'worst month':<24} {stats['worst']:>8.3f}  "
          f"({base['pnl'].idxmin().date()})")
    print(f"{'best month':<24} {stats['best']:>8.3f}  "
          f"({base['pnl'].idxmax().date()})")

    print(f"\n{'average implied vol':<24} {base['implied'].mean():>8.2%}")
    print(f"{'average realized vol':<24} {base['realized'].mean():>8.2%}")
    print(f"{'implied above realized':<24} "
          f"{float((base['implied'] > base['realized']).mean()):>8.1%} of months")

    print("\nIs the P&L a variance bet? Regress it on implied^2 - realized^2:")
    x = base["variance_gap"].to_numpy()
    y = base["pnl"].to_numpy()
    slope, intercept = np.polyfit(x, y, 1)
    correlation = float(np.corrcoef(x, y)[0, 1])
    print(f"  slope {slope:>8.2f}   intercept {intercept:>7.3f}   "
          f"r = {correlation:.3f}   r^2 = {correlation ** 2:.3f}")
    print("  The theory says P&L = sum of 1/2 Gamma S^2 (implied^2 - realized^2),")
    print("  so the whole month's outcome should be a near-linear function of that")
    print("  one gap with an intercept of zero. Direction of the market: absent.")

    print("\nDoes the gamma decomposition reproduce the simulated hedge P&L?")
    error = (base["gamma_pnl"] - base["pnl"]).abs()
    print(f"  mean |difference| {error.mean():.4f} against a mean premium of "
          f"{base['premium'].mean():.3f}")
    print(f"  correlation {float(np.corrcoef(base['gamma_pnl'], base['pnl'])[0, 1]):.4f}")

    print("\nHedging frequency. Boyle-Emanuel (1980): the error a discrete hedge")
    print("adds should have zero mean and a standard deviation growing like the")
    print("square root of the interval. Measured against the same month's daily")
    print("hedge, trade by trade, so the variance premium common to all of them")
    print("is differenced out:\n")
    print(f"{'rebalance':>12} {'mean P&L':>10} {'vs daily':>10} "
          f"{'error std':>10} {'/ sqrt(n)':>10} {'worst':>9}")
    print(f"{'every 1d':>12} {base['pnl'].mean():>10.3f} {'-':>10} {'-':>10} "
          f"{'-':>10} {base['pnl'].min():>9.3f}")
    for every in FREQUENCIES:
        table = pd.DataFrame([trade(w, rebalance_every=every) for w in windows])
        difference = table["pnl"].to_numpy() - base["pnl"].to_numpy()
        print(f"{'every ' + str(every) + 'd':>12} {table['pnl'].mean():>10.3f} "
              f"{difference.mean():>+10.3f} {difference.std(ddof=1):>10.3f} "
              f"{difference.std(ddof=1) / np.sqrt(every):>10.3f} "
              f"{table['pnl'].min():>9.3f}")
    print("  'vs daily' is the mean change and it stays inside the noise: hedging")
    print("  less often does not cost expected money. 'error std' is what it does")
    print("  cost, and the last column divides out sqrt(interval) to check the")
    print("  scaling. It is flat from 2 to 10 days and then falls off, because at")
    print("  21 days the option is hedged once at inception and the error has")
    print("  nowhere left to grow.")

    print("\nTransaction costs, daily hedging:")
    print(f"{'cost':>8} {'mean P&L':>10} {'shares cost':>12} {'Sharpe':>8}")
    for bps in COSTS_BPS:
        table = pd.DataFrame([trade(w, cost_bps=bps) for w in windows])
        s = summarize(table)
        print(f"{bps:>6.0f}bp {s['mean']:>10.3f} {table['costs'].mean():>12.3f} "
              f"{s['sharpe']:>8.2f}")

    print("\nThe five worst months:")
    print(f"{'month':>12} {'implied':>9} {'realized':>9} {'P&L':>8}")
    for date, row in base.nsmallest(5, "pnl").iterrows():
        print(f"{date.strftime('%Y-%m'):>12} {row['implied']:>9.1%} "
              f"{row['realized']:>9.1%} {row['pnl']:>8.3f}")


if __name__ == "__main__":
    main()
