"""Driver: four allocation methods, in sample and out of sample.

The in-sample table is the easy part and it flatters max-Sharpe, because
max-Sharpe is *defined* as the in-sample winner. The walk-forward test is the
one that means anything: estimate mu and Sigma on a trailing window, hold for a
year, roll forward, and compare what you actually earned.

ETFs rather than single stocks: they are diversified already, so the covariance
structure is stable enough for the differences between methods to be visible
rather than drowned in noise.

Usage:  python run_optimization.py
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yfinance as yf

from src.black_litterman import (black_litterman_weights, equilibrium_returns,
                                 implied_risk_aversion, relative_view)
from src.cvar import cvar_of_weights, min_cvar_weights, var_of_weights
from src.frontier import efficient_frontier
from src.optimizer import (max_sharpe_turnover_penalized, max_sharpe_weights,
                          min_variance_weights, portfolio_performance)
from src.returns import TRADING_DAYS, annualized_cov, annualized_mean, daily_returns
from src.risk_parity import (equal_risk_contribution_weights, inverse_vol_weights,
                             ledoit_wolf_alpha, risk_contributions,
                             shrink_covariance)

# stocks / intl stocks / bonds / gold / real estate — deliberately heterogeneous
UNIVERSE = ["SPY", "EFA", "AGG", "GLD", "VNQ"]
START, END = "2007-01-01", "2024-12-31"
LOOKBACK_YEARS = 3
SHRINKAGE = 0.3

# Black-Litterman needs a "market portfolio" to reverse-optimize. These are the
# five ETFs' net assets, so the anchor is what investors actually hold in these
# funds rather than a number picked to make the result look good. Fetched live
# when yfinance cooperates; these are the fallback, as of 2026-09-10.
FALLBACK_AUM = {"SPY": 811.9e9, "EFA": 79.3e9, "AGG": 138.3e9,
                "GLD": 152.9e9, "VNQ": 70.8e9}

# The long-run Sharpe of a diversified market portfolio, used to set Black-
# Litterman's risk aversion as delta = sharpe / sigma_market. Deliberately an
# assumption rather than an estimate: computing delta from the sample mean
# would smuggle the noisy input back in through the one door the whole method
# exists to close.
MARKET_SHARPE = 0.40

# Magnitude of the momentum view, annualized. Fixed, not fitted - the question
# this project asks is whether a view helps at all, and a magnitude tuned on
# the same data would answer a different and much less interesting question.
VIEW_SPREAD = 0.02


def market_weights(tickers: list[str]) -> np.ndarray:
    """Market-cap weights from the ETFs' net assets, normalized to sum to 1.

    The honest caveat, stated once here rather than buried: these are today's
    fund sizes used as the equilibrium anchor for a walk-forward starting in
    2010. AUM shares move slowly and this is a weighting anchor rather than a
    return forecast, so it is a mild offense, not a fatal one - but it is
    information from after the fact, and RESULTS.md says so where the numbers
    are reported.
    """
    aum = {}
    for t in tickers:
        try:
            info = yf.Ticker(t).info
            aum[t] = float(info.get("totalAssets") or info.get("netAssets")
                           or FALLBACK_AUM[t])
        except Exception:
            aum[t] = FALLBACK_AUM[t]
    total = sum(aum.values())
    return np.array([aum[t] / total for t in tickers])


def momentum_view(train: pd.DataFrame, spread: float = VIEW_SPREAD) -> tuple:
    """"The best trailing performer beats the worst by `spread`."

    A view has to come from somewhere, and for a walk-forward it has to come
    from data available at the rebalance date. Twelve-month price momentum is
    the most-documented cross-sectional signal there is, it needs nothing but
    the training window, and it is entirely mechanical - no judgment calls to
    tune afterwards.

    Returns (P, Q) or (None, None) when the window is too short.
    """
    window = train.iloc[-252:] if len(train) >= 252 else train
    if len(window) < 60:
        return None, None
    total = window.iloc[-1] / window.iloc[0] - 1
    best, worst = int(np.argmax(total.values)), int(np.argmin(total.values))
    if best == worst:
        return None, None
    return relative_view(len(total), best, worst, spread)


def section(title: str) -> None:
    print(f"\n{title}\n" + "-" * len(title))


def build_portfolios(mu: np.ndarray, cov: np.ndarray,
                     train_returns: np.ndarray | None = None,
                     w_market: np.ndarray | None = None,
                     view: tuple | None = None) -> dict:
    """train_returns (raw daily, not annualized) is optional and enables one
    more portfolio: Min CVaR. It's the only method here that doesn't reduce
    the training data to (mu, cov) first - it needs the actual scenarios,
    tail shape included. Optional because a couple of tests and any future
    caller working purely from (mu, cov) shouldn't be forced to carry raw
    returns around just to build the other four portfolios.
    """
    n = len(mu)
    portfolios = {
        "Equal weight": np.full(n, 1 / n),
        "Inverse vol": inverse_vol_weights(cov),
        "Min variance": min_variance_weights(cov),
        "Max Sharpe": max_sharpe_weights(mu, cov),
        "Equal risk contribution": equal_risk_contribution_weights(cov),
    }
    if train_returns is not None:
        portfolios["Min CVaR (95%)"] = min_cvar_weights(train_returns)

    if w_market is not None:
        # delta from an assumed market Sharpe and the ESTIMATED market vol -
        # covariance is the input this repo trusts, sample means are not
        market_vol = float(np.sqrt(w_market @ cov @ w_market))
        delta = implied_risk_aversion(MARKET_SHARPE * market_vol, market_vol ** 2)
        portfolios["Black-Litterman (no views)"] = black_litterman_weights(
            cov, w_market, delta)
        if view is not None and view[0] is not None:
            portfolios["Black-Litterman (momentum)"] = black_litterman_weights(
                cov, w_market, delta, view[0], view[1])
    return portfolios


def weight_string(w: np.ndarray) -> str:
    return " ".join(f"{t}:{x:>4.0%}" for t, x in zip(UNIVERSE, w))


def in_sample(prices: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, dict]:
    mu = annualized_mean(prices).values
    cov = annualized_cov(prices).values
    train_returns = daily_returns(prices).values

    section(f"1. In sample, whole history ({prices.index[0].date()} "
            f"-> {prices.index[-1].date()})")
    print(f"{'portfolio':<26} {'ret':>7} {'vol':>7} {'sharpe':>7}   weights")
    portfolios = build_portfolios(mu, cov, train_returns)
    for name, w in portfolios.items():
        ret, vol, sharpe = portfolio_performance(w, mu, cov)
        print(f"{name:<26} {ret:>7.2%} {vol:>7.2%} {sharpe:>7.2f}   {weight_string(w)}")
    print("\nMax Sharpe wins by construction - it is the in-sample argmax. The only")
    print("question worth asking is whether it repeats out of sample.")

    section("1b. Tail risk: variance doesn't see it, CVaR does")
    print(f"{'portfolio':<26} {'vol':>7} {'VaR 95%':>9} {'CVaR 95%':>9}  (daily)")
    for name, w in portfolios.items():
        _, vol, _ = portfolio_performance(w, mu, cov)
        v = var_of_weights(w, train_returns, alpha=0.95)
        c = cvar_of_weights(w, train_returns, alpha=0.95)
        print(f"{name:<26} {vol:>7.2%} {v:>9.2%} {c:>9.2%}")
    print("\nVaR 95% is the daily loss exceeded only 1 day in 20; CVaR 95% is the")
    print("average loss on those worst days specifically. Min CVaR is built to")
    print("minimize the last column directly - RESULTS.md checks whether the")
    print("portfolios variance already favors (min variance, ERC) happen to be")
    print("good at this too, or whether tail risk is a genuinely separate axis.")

    section("2. Risk contributions: weights lie, risk doesn't")
    print(f"{'portfolio':<26} " + " ".join(f"{t:>6}" for t in UNIVERSE))
    for name in ("Equal weight", "Min variance", "Max Sharpe",
                 "Equal risk contribution"):
        rc = risk_contributions(portfolios[name], cov)
        print(f"{name:<26} " + " ".join(f"{x:>6.0%}" for x in rc))
    ew_rc = risk_contributions(portfolios["Equal weight"], cov)
    worst = UNIVERSE[int(np.argmax(ew_rc))]
    print(f"\nEqual weight puts 20% of capital in every asset and "
          f"{ew_rc.max():.0%} of its RISK\nin {worst} alone. "
          "That is the gap ERC exists to close.")

    section(f"3. Covariance shrinkage (alpha = {SHRINKAGE})")
    shrunk = shrink_covariance(cov, SHRINKAGE)
    w_raw = min_variance_weights(cov)
    w_shrunk = min_variance_weights(shrunk)
    print(f"{'min-var on raw cov':<26} {weight_string(w_raw)}")
    print(f"{'min-var on shrunk cov':<26} {weight_string(w_shrunk)}")
    print(f"{'L1 weight change':<26} {np.abs(w_raw - w_shrunk).sum():.4f}")
    eig_raw = np.linalg.eigvalsh(cov)
    eig_shrunk = np.linalg.eigvalsh(shrunk)
    print(f"\ncondition number: raw {eig_raw[-1] / eig_raw[0]:>8.1f}  ->  "
          f"shrunk {eig_shrunk[-1] / eig_shrunk[0]:.1f}")
    print("Shrinkage lifts the smallest eigenvalue. Those near-zero directions are")
    print("the worst-estimated ones, and they are exactly where an optimizer piles")
    print("in - a spuriously low variance looks like free risk reduction.")

    lw_alpha = ledoit_wolf_alpha(daily_returns(prices).values)
    print(f"\nSHRINKAGE={SHRINKAGE} above was hand-picked. Ledoit-Wolf's own "
          f"analytic optimum\non this data (whole-history daily returns) is "
          f"alpha = {lw_alpha:.3f}.")
    w_lw = min_variance_weights(shrink_covariance(cov, lw_alpha))
    print(f"{'min-var on LW-shrunk cov':<26} {weight_string(w_lw)}")
    print(f"{'L1 vs hand-picked shrunk':<26} {np.abs(w_shrunk - w_lw).sum():.4f}")

    return mu, cov, portfolios


def walk_forward(prices: pd.DataFrame, lookback: int = LOOKBACK_YEARS,
                 verbose: bool = True,
                 w_market: np.ndarray | None = None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Refit annually on a trailing window, hold for the next year."""
    if verbose:
        section(f"4. Walk-forward, out of sample ({lookback}y trailing estimate, "
                "annual rebalance)")

    rets = daily_returns(prices)
    years = sorted(rets.index.year.unique())
    test_years = [y for y in years if y - lookback >= years[0]]

    records = []
    prev_weights: dict[str, np.ndarray] = {}
    turnover: dict[str, list] = {}

    for year in test_years:
        train = prices[(prices.index.year >= year - lookback)
                       & (prices.index.year < year)]
        test = rets[rets.index.year == year]
        if len(train) < 250 or test.empty:
            continue

        mu = annualized_mean(train).values
        cov = annualized_cov(train).values
        train_returns = daily_returns(train).values
        try:
            portfolios = build_portfolios(mu, cov, train_returns, w_market,
                                          momentum_view(train))
        except RuntimeError as e:
            if verbose:
                print(f"  {year}: optimizer failed ({e}); skipped")
            continue

        for name, w in portfolios.items():
            # buy-and-hold within the year, so drift is realistic
            realized = (test.values @ w)
            records.append({"year": year, "portfolio": name,
                            "ret": float(np.prod(1 + realized) - 1),
                            "vol": float(realized.std(ddof=1) * np.sqrt(TRADING_DAYS))})
            if name in prev_weights:
                turnover.setdefault(name, []).append(
                    float(np.abs(w - prev_weights[name]).sum() / 2))
            prev_weights[name] = w

    panel = pd.DataFrame(records)

    summary = []
    for name in panel["portfolio"].unique():
        g = panel[panel["portfolio"] == name]
        cagr = float(np.prod(1 + g["ret"]) ** (1 / len(g)) - 1)
        vol = float(g["ret"].std(ddof=1))
        summary.append({"portfolio": name, "cagr": cagr, "vol": vol,
                        "sharpe": cagr / vol if vol > 0 else 0.0,
                        "worst": float(g["ret"].min()),
                        "turnover": float(np.mean(turnover.get(name, [0.0])))})
    summary = pd.DataFrame(summary)

    if verbose:
        print(f"{len(test_years)} annual rebalances, "
              f"{panel['year'].min()}-{panel['year'].max()}\n")
        print(f"{'portfolio':<26} {'ann ret':>8} {'ann vol':>8} {'sharpe':>7} "
              f"{'worst yr':>9} {'turnover':>9}")
        for _, r in summary.iterrows():
            print(f"{r['portfolio']:<26} {r['cagr']:>8.2%} {r['vol']:>8.2%} "
                  f"{r['sharpe']:>7.2f} {r['worst']:>9.2%} {r['turnover']:>9.1%}")
        print("\nSharpe here uses annual return dispersion, not daily - it measures how")
        print("reliably each method delivered year to year. Turnover is the average")
        print("one-way weight change per rebalance: what you pay to hold the view.")

    return panel, summary


def turnover_penalized_walk_forward(prices: pd.DataFrame,
                                    lookback: int = LOOKBACK_YEARS,
                                    penalties=(0.0, 5.0, 15.0, 40.0, 100.0)) -> pd.DataFrame:
    """Same walk-forward as max-Sharpe, but re-run at several turnover penalties.

    Unlike the stateless methods in build_portfolios(), this one needs last
    period's weights, so it keeps its own rolling state instead of reusing
    walk_forward(). penalty=0 should reproduce the plain max-Sharpe row from
    section 4 (same optimizer, same data, same rebalance dates).
    """
    section("5. Trading the turnover away - max-Sharpe at several penalties")
    print("Same walk-forward as section 4's Max Sharpe row, but the optimizer now")
    print("pays a quadratic cost for moving away from last year's weights.\n")

    rets = daily_returns(prices)
    years = sorted(rets.index.year.unique())
    test_years = [y for y in years if y - lookback >= years[0]]

    rows = []
    for penalty in penalties:
        records = []
        turnovers = []
        w_prev = None
        for year in test_years:
            train = prices[(prices.index.year >= year - lookback)
                           & (prices.index.year < year)]
            test = rets[rets.index.year == year]
            if len(train) < 250 or test.empty:
                continue

            mu = annualized_mean(train).values
            cov = annualized_cov(train).values
            if w_prev is None:
                # No prior holding to penalize against on the very first
                # rebalance - fall back to plain max-Sharpe, same as
                # walk_forward() implicitly does by only recording turnover
                # from the second rebalance on.
                w = max_sharpe_weights(mu, cov)
            else:
                try:
                    w = max_sharpe_turnover_penalized(mu, cov, w_prev, penalty)
                except RuntimeError:
                    w = w_prev
                turnovers.append(float(np.abs(w - w_prev).sum() / 2))

            realized = test.values @ w
            records.append({"year": year, "ret": float(np.prod(1 + realized) - 1)})
            w_prev = w

        panel = pd.DataFrame(records)
        cagr = float(np.prod(1 + panel["ret"]) ** (1 / len(panel)) - 1)
        vol = float(panel["ret"].std(ddof=1))
        rows.append({"penalty": penalty, "cagr": cagr, "vol": vol,
                    "sharpe": cagr / vol if vol > 0 else 0.0,
                    "turnover": float(np.mean(turnovers))})

    summary = pd.DataFrame(rows)
    print(f"{'penalty':>9} {'ann ret':>8} {'ann vol':>8} {'sharpe':>7} {'turnover':>9}")
    for _, r in summary.iterrows():
        print(f"{r['penalty']:>9.0f} {r['cagr']:>8.2%} {r['vol']:>8.2%} "
              f"{r['sharpe']:>7.2f} {r['turnover']:>9.1%}")
    print("\nCompare the penalty=0 row above to Max Sharpe in section 4 - same")
    print("optimizer, same data, same dates, so it should land in the same place.")
    print("As the penalty rises, turnover drops toward the risk-based methods'")
    print("levels. Whether Sharpe improves, holds, or degrades on the way there")
    print("is the actual answer to 'does penalizing turnover help' - not assumed.")
    return summary


def equilibrium_section(prices: pd.DataFrame, cov: np.ndarray,
                        w_market: np.ndarray) -> None:
    """What the market's own positioning implies about expected returns.

    Everything in section 1 that involves mu uses the sample mean, which is
    the input RESULTS.md spends most of its length being suspicious of.
    Black-Litterman offers a different one: assume the market portfolio is
    optimal for somebody, and solve backwards for the returns that would make
    it so. No sample mean is involved at any point.
    """
    section("7. Black-Litterman: what does the market already believe?")

    market_vol = float(np.sqrt(w_market @ cov @ w_market))
    delta = implied_risk_aversion(MARKET_SHARPE * market_vol, market_vol ** 2)
    pi = equilibrium_returns(cov, w_market, delta)
    sample_mu = annualized_mean(prices).values

    print(f"Market portfolio by fund net assets: {weight_string(w_market)}")
    print(f"Market vol {market_vol:.2%}, assumed Sharpe {MARKET_SHARPE:.2f} "
          f"-> risk aversion delta = {delta:.2f}\n")
    print(f"{'asset':<8} {'equilibrium':>12} {'sample mean':>12} {'difference':>12}")
    for i, t in enumerate(UNIVERSE):
        print(f"{t:<8} {pi[i]:>12.2%} {sample_mu[i]:>12.2%} "
              f"{sample_mu[i] - pi[i]:>12.2%}")
    print("\nThe two columns disagree by multiples, and the sample column is the")
    print("one with 18 years of noise in it. Equilibrium says gold and bonds")
    print("should return little because they carry little of the market's risk;")
    print("the sample says whatever the last 18 years happened to deliver.")
    print("Max-Sharpe optimizes against the second column. That is the whole")
    print("complaint this project has been making, restated as two columns.")


# Anchors to test Black-Litterman's dependence on the equilibrium it starts
# from. "AUM" is today's fund sizes; the other two are deliberate alternatives
# so the result can be checked against a less equity-heavy starting point.
ALT_ANCHORS = {
    "Fund AUM (65% SPY)": None,                                  # filled at runtime
    "Textbook global market": np.array([0.40, 0.15, 0.30, 0.05, 0.10]),
    "Equal weight anchor": np.full(5, 0.2),
}


def anchor_sensitivity(prices: pd.DataFrame, w_market: np.ndarray) -> None:
    """How much of Black-Litterman's result is the anchor?

    This matters more here than the usual robustness check, because the AUM
    anchor is the one genuinely forward-looking input in the whole project:
    SPY is 65% of these five funds' assets TODAY, and a large part of why is
    that US equities outperformed over exactly the window being tested. An
    equilibrium anchor built from the winners is not a fair starting point,
    and a Black-Litterman result that only survives that anchor is not a
    result at all.

    So: run it again from two anchors that know nothing about the outcome.
    """
    section("8. Is Black-Litterman's edge just a well-chosen anchor?")

    anchors = dict(ALT_ANCHORS)
    anchors["Fund AUM (65% SPY)"] = w_market

    print(f"{'anchor':<24} {'weights':<44} {'no views':>9} {'momentum':>9}")
    for label, anchor in anchors.items():
        _, summary = walk_forward(prices, verbose=False, w_market=anchor)
        rows = summary.set_index("portfolio")["sharpe"]
        no_view = rows.get("Black-Litterman (no views)", float("nan"))
        with_view = rows.get("Black-Litterman (momentum)", float("nan"))
        print(f"{label:<24} {weight_string(anchor):<44} "
              f"{no_view:>9.2f} {with_view:>9.2f}")

    _, base = walk_forward(prices, verbose=False)
    ew = float(base.loc[base["portfolio"] == "Equal weight", "sharpe"].iloc[0])
    print(f"\nEqual weight, same walk-forward, for comparison: {ew:.2f}")
    print("If the two outcome-blind anchors also clear that bar, the method is")
    print("doing work. If only the AUM anchor does, the result was the anchor.")


def lookback_sensitivity(prices: pd.DataFrame) -> None:
    """The same test at four estimation windows.

    One walk-forward result could be an accident of the window length. If the
    ranking survives 2, 3, 5 and 7 years it is telling you something about the
    methods rather than about the choice.
    """
    section("6. Does the ranking survive a different estimation window?")
    print(f"{'portfolio':<26} " + " ".join(f"{lb}y".rjust(7) for lb in (2, 3, 5, 7)))

    w_mkt = market_weights(UNIVERSE)
    results = {lb: walk_forward(prices, lookback=lb, verbose=False, w_market=w_mkt)[1]
               for lb in (2, 3, 5, 7)}
    names = results[3]["portfolio"].tolist()
    for name in names:
        row = " ".join(
            f"{float(results[lb].loc[results[lb]['portfolio'] == name, 'sharpe'].iloc[0]):>7.2f}"
            for lb in (2, 3, 5, 7)
        )
        print(f"{name:<26} {row}")
    print("\n(out-of-sample Sharpe of annual returns, by trailing estimation window)")


def main() -> None:
    prices = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True,
                         progress=False)["Close"].dropna()
    prices = prices[UNIVERSE]  # yfinance sorts columns; keep our order
    print(f"Universe: {', '.join(UNIVERSE)}")
    print(f"Data: {prices.index[0].date()} -> {prices.index[-1].date()}, "
          f"{len(prices):,} trading days")

    w_mkt = market_weights(UNIVERSE)

    mu, cov, portfolios = in_sample(prices)
    panel, _ = walk_forward(prices, w_market=w_mkt)
    turnover_penalized_walk_forward(prices)
    equilibrium_section(prices, cov, w_mkt)
    anchor_sensitivity(prices, w_mkt)
    lookback_sensitivity(prices)

    ef = efficient_frontier(mu, cov, n_points=60)
    fig, axes = plt.subplots(2, 1, figsize=(10, 11))

    axes[0].plot(ef["volatility"], ef["target_return"], "-", lw=1.5,
                 label="efficient frontier")
    for name, w in portfolios.items():
        ret, vol, _ = portfolio_performance(w, mu, cov)
        axes[0].scatter(vol, ret, s=45, zorder=3, label=name)
    for i, ticker in enumerate(UNIVERSE):
        axes[0].scatter(np.sqrt(cov[i, i]), mu[i], marker="x", c="gray", zorder=2)
        axes[0].annotate(ticker, (np.sqrt(cov[i, i]), mu[i]), fontsize=8,
                         xytext=(4, -2), textcoords="offset points", color="gray")
    axes[0].set_xlabel("Volatility (annualized)")
    axes[0].set_ylabel("Expected return (annualized)")
    axes[0].set_title("Efficient frontier, full sample — individual assets in grey")
    axes[0].legend(fontsize=8)

    wide = panel.pivot(index="year", columns="portfolio", values="ret")
    cumulative = (1 + wide).cumprod()
    for col in cumulative.columns:
        axes[1].plot(cumulative.index, cumulative[col], marker="o", ms=3, label=col)
    axes[1].axhline(1.0, c="gray", lw=0.6)
    axes[1].set_xlabel("year"); axes[1].set_ylabel("growth of $1")
    axes[1].set_title("Out of sample — 3y trailing estimates, annual rebalance")
    axes[1].legend(fontsize=8)

    plt.tight_layout()
    plt.savefig("frontier.png", dpi=120)
    print("\nSaved plot to frontier.png")


if __name__ == "__main__":
    main()
