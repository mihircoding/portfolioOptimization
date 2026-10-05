"""Allocating across the two sleeves, which is the job the optimizer was built for.

Usage:  python multi_strategy.py [--quick]
Verify with:  pytest tests/test_multi_strategy.py

Every result in notes/allocator.md tests the optimizer on ASSETS - five ETFs,
then fifty large caps. That is not what a risk allocator is for. Its purpose is
combining return streams that are not very correlated with each other, and
correlated assets are the one input that guarantees it nothing: when everything
in the book is 85% correlated with the market, no weighting can diversify away
a risk that is not there to diversify.

So the standing result of that file - that equal weighting beats mean-variance
out of sample, at every estimation window tested - has always had an obvious
rejoinder. Of course it does. You never gave it anything worth optimizing.

This repository now contains two things worth optimizing, and they came from
the other two projects that merged into it:

  statarb    930 cointegrated equity pairs, long one leg and short the other,
             market-neutral by construction. notes/statarb.md.
  volcarry   sell a one-month at-the-money SPY call every month and delta-hedge
             it daily, which is the variance risk premium. notes/volcarry.md.

One is long volatility-of-spreads and the other is short volatility outright.
There is no reason for them to be correlated and they are not. If the optimizer
cannot earn its keep here, the rejoinder is dead and the standing result is the
real one.

The answer turns out to be yes, for the first time in this repository, and the
reason is not the one the diversification argument predicts. It is that one of
the two sleeves does not make money, mean-variance can see that and the
risk-based methods cannot - because risk parity, inverse volatility and HRP
never look at returns at all, and the sleeve that loses money is also the one
with the lower volatility, so they allocate *toward* it. Section 3 has the
numbers and section 4 has the test that says in advance which regime you are in.

Frequency and sample. The volatility sleeve is genuinely monthly: one
non-overlapping trade per month, so its P&L series has 12 independent
observations a year and no autocorrelation to argue about. The stat-arb sleeve
is daily and gets compounded to monthly to match. The overlap is what it is -
four years, around 48 months - which is a thin sample for estimating a
covariance matrix, and that is stated rather than worked around. It is also
exactly the regime notes/allocator.md section 9 is about, so a thin sample is
the honest test rather than a caveat on one.
"""

from __future__ import annotations

import argparse
import json
import os

import numpy as np
import pandas as pd

from src.hrp import hrp_weights
from src.optimizer import max_sharpe_weights, min_variance_weights
from src.risk_parity import (equal_risk_contribution_weights,
                             inverse_vol_weights, ledoit_wolf_alpha,
                             shrink_covariance)

MONTHS_PER_YEAR = 12
LOOKBACK_MONTHS = 24     # trailing window the walk-forward estimates on
# Half-spreads on the option, in vol points, for section 7. 0.10 is the
# sleeve's own default and roughly SPY's front-month market; the rest run out
# to 8.00, which is eighty times that and not a market anyone has seen, in
# order to find where the conclusion actually breaks rather than to propose it.
SPREAD_VOL_SWEEP = (0.0, 0.10, 0.25, 0.50, 1.00, 2.00, 4.00, 8.00)
SLEEVES = ("statarb", "volcarry")
OUT = "results/multi_strategy.json"


# ---------- the two sleeves' return streams ----------

def statarb_monthly(path: str = "sleeves/statarb/results/portfolio.json") -> pd.Series:
    """Monthly returns of the equal-weight 930-pair book.

    Read from the sleeve's own committed output rather than recomputed, so this
    file cannot quietly disagree with notes/statarb.md about what the sleeve
    did. The equity curve there is the equal-weight book, which is that
    project's headline portfolio and the only one it reports a full series for.
    """
    with open(path) as fh:
        equity = json.load(fh)["equity"]
    series = pd.Series({pd.Timestamp(d): float(v) for d, v in equity}).sort_index()
    monthly = series.resample("ME").last()
    return monthly.pct_change().dropna()


def volcarry_monthly(start: str = "2007-01-01", end: str = "2024-12-31",
                     cache: str | None = None,
                     spread_vol_pts: float | None = None) -> pd.Series:
    """Monthly P&L of the delta-hedged short call, as a return.

    hedging_study reports P&L per $100 of underlying, which is already a return
    on a notional - dividing by 100 is the whole conversion. That notional is a
    choice, and it is the one the sleeve's own write-up uses, so the Sharpe here
    matches the Sharpe there and the two files cannot drift.

    spread_vol_pts : half the option's bid-ask in vol points, charged on the
        monthly sale. Defaults to the sleeve's own OPTION_SPREAD_VOL_PTS so
        this file and hedging_study cannot disagree about what the sleeve
        costs. Section 6 sweeps it, including 0, which is the mid-market
        series the first version of this study used.

        This matters because the two sleeves were NOT cost-matched. statarb
        charges bid-ask on every share it trades; volcarry charged hedging
        costs in basis points but sold the option itself at the mid. Comparing
        a sleeve that pays its spreads against one that does not overstates
        the gap between them by an amount nobody had measured.

    Cached because it is the only part of this study that needs the network,
    and a cached file means the walk-forward is reproducible after Yahoo
    changes its mind about something. The cache name carries the spread, so
    changing it does not silently read back a series priced differently.
    """
    from sleeves.volcarry.hedging_study import (OPTION_SPREAD_VOL_PTS, load,
                                                months, trade)

    if spread_vol_pts is None:
        spread_vol_pts = OPTION_SPREAD_VOL_PTS
    if cache is None:
        tag = f"{spread_vol_pts:g}".replace(".", "p")
        cache = f"results/volcarry_monthly_sp{tag}.csv"

    if os.path.exists(cache):
        frame = pd.read_csv(cache, index_col=0, parse_dates=True)
        return frame["ret"]

    data = load()
    data = data.loc[start:end]
    rows = [trade(window, option_spread_vol_pts=spread_vol_pts)
            for window in months(data)]
    frame = pd.DataFrame(rows).set_index("date")
    series = (frame["pnl"] / 100.0).rename("ret")
    series.index = series.index.to_period("M").to_timestamp("M")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    series.to_frame().to_csv(cache)
    return series


def sleeve_panel(**kwargs) -> pd.DataFrame:
    """Both sleeves on one monthly index, over the months both existed.

    The intersection is taken on the month, not forward-filled. A sleeve that
    was not running has no return, and carrying the last one forward would
    invent a flat month and flatter the volatility estimate.
    """
    statarb = statarb_monthly()
    volcarry = volcarry_monthly(**kwargs)
    statarb.index = statarb.index.to_period("M").to_timestamp("M")
    volcarry.index = volcarry.index.to_period("M").to_timestamp("M")
    panel = pd.DataFrame({"statarb": statarb, "volcarry": volcarry}).dropna()
    return panel


# ---------- the allocators ----------

def allocators(shrink: bool = True) -> dict:
    """Every weighting scheme in this repository, on a common signature.

    Each takes (mu, cov) and returns weights, whether or not it uses mu. That
    the risk-based four ignore it is the point of section 3, so they take it and
    throw it away rather than having a different signature that hides the fact.
    """
    return {
        "equal weight": lambda mu, cov: np.ones(len(mu)) / len(mu),
        "inverse vol": lambda mu, cov: inverse_vol_weights(cov),
        "risk parity": lambda mu, cov: equal_risk_contribution_weights(cov),
        "HRP": lambda mu, cov: hrp_weights(cov),
        "min variance": lambda mu, cov: min_variance_weights(cov),
        "max Sharpe": lambda mu, cov: max_sharpe_weights(mu, cov),
    }


def estimate(train: pd.DataFrame, shrink: bool = True) -> tuple:
    """Annualized mean and covariance from a trailing window.

    Ledoit-Wolf shrinkage is on by default because notes/allocator.md section 9
    is about what an unshrunk covariance does to a small sample, and 24 months
    of a 2x2 is a small sample. With two assets there are three parameters, so
    the shrinkage barely bites - reported anyway, so nobody has to guess.
    """
    values = train.to_numpy(dtype=float)
    mu = values.mean(axis=0) * MONTHS_PER_YEAR
    cov = np.cov(values, rowvar=False, ddof=1) * MONTHS_PER_YEAR
    if shrink:
        cov = shrink_covariance(cov, ledoit_wolf_alpha(values))
    return mu, cov


def walk_forward(panel: pd.DataFrame, lookback: int = LOOKBACK_MONTHS,
                 shrink: bool = True) -> dict:
    """Estimate on the trailing window, hold for one month, repeat.

    Nothing here sees a return before it has to allocate against it, which is
    the only configuration in which a comparison between allocators means
    anything. The first `lookback` months are spent estimating and are not
    traded by anybody, so every scheme is measured on an identical sample.
    """
    schemes = allocators()
    realized = {name: [] for name in schemes}
    weights = {name: [] for name in schemes}
    dates = []

    for i in range(lookback, len(panel)):
        train = panel.iloc[i - lookback:i]
        nxt = panel.iloc[i].to_numpy(dtype=float)
        mu, cov = estimate(train, shrink=shrink)
        dates.append(panel.index[i])
        for name, fn in schemes.items():
            w = np.asarray(fn(mu, cov), dtype=float)
            realized[name].append(float(w @ nxt))
            weights[name].append(w)

    out = {}
    for name in schemes:
        series = pd.Series(realized[name], index=dates)
        w = np.array(weights[name])
        out[name] = {
            **performance(series),
            "mean_weight": dict(zip(panel.columns, w.mean(axis=0).round(4))),
            "weight_on_statarb": float(w[:, list(panel.columns).index("statarb")].mean()),
        }
    return {"months_traded": len(dates), "first": str(dates[0].date()),
            "last": str(dates[-1].date()), "schemes": out}


def performance(monthly: pd.Series) -> dict:
    """Annualized return, volatility, Sharpe and worst month."""
    mean = float(monthly.mean()) * MONTHS_PER_YEAR
    vol = float(monthly.std(ddof=1)) * np.sqrt(MONTHS_PER_YEAR)
    return {
        "ann_return": mean,
        "ann_vol": vol,
        "sharpe": mean / vol if vol > 0 else 0.0,
        "worst_month": float(monthly.min()),
        "hit_rate": float((monthly > 0).mean()),
    }


# ---------- the test that says which regime you are in ----------

def return_gap_tstat(panel: pd.DataFrame) -> dict:
    """How confidently the two sleeves' mean returns can be told apart.

    This is the number that decides whether an optimizer has anything to work
    with. Mean-variance beats equal weighting exactly when the difference in
    expected return is large relative to the error on estimating it; when the
    t-statistic is under 1 the optimizer is allocating on noise, which is the
    situation in notes/allocator.md and the reason equal weighting wins there.

    Welch's t-test rather than a paired one, because the two sleeves are not
    two measurements of the same thing - they are independent strategies, which
    is the entire premise of combining them, and the correlation reported
    beside it is the check on that premise.
    """
    a, b = panel["statarb"], panel["volcarry"]
    diff = float(b.mean() - a.mean())
    se = float(np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b)))
    return {
        "monthly_gap": diff,
        "annualized_gap": diff * MONTHS_PER_YEAR,
        "std_error": se,
        "tstat": diff / se if se > 0 else 0.0,
        "correlation": float(a.corr(b)),
        "n_months": int(len(panel)),
    }


def diversification_ratio(panel: pd.DataFrame) -> float:
    """Weighted average sleeve volatility divided by the equal-weight book's.

    1.0 means combining them bought nothing. With non-negative correlations the
    square root of the number of sleeves - 1.41 for two - is the ceiling, and it
    is reached only at exactly zero correlation and equal volatility. Genuinely
    negative correlation can exceed it without limit, which is worth stating
    because two hedged strategies can be anti-correlated and a ratio of 4 would
    otherwise look like a bug.
    """
    w = np.ones(panel.shape[1]) / panel.shape[1]
    vols = panel.std(ddof=1).to_numpy(dtype=float)
    cov = np.cov(panel.to_numpy(dtype=float), rowvar=False, ddof=1)
    return float((w @ vols) / np.sqrt(w @ cov @ w))


def window_bias(cache: str | None = None,
                overlap: pd.DataFrame | None = None,
                spread_vol_pts: float | None = None) -> dict:
    """What the overlap window leaves out of the volatility sleeve's history.

    This is the check that decides whether section 2 is a result or an artifact.
    The stat-arb sleeve only exists from 2021, so the two sleeves overlap over
    2021-2024 - and 2021-2024 contains no 2008 and no March 2020. Selling
    volatility is a strategy whose entire risk is in months like those. Its own
    write-up records losing 2.675 per $100 in March 2020, against a mean month
    of +0.405.

    So the sleeve is being judged on a window chosen by the OTHER sleeve's
    start date, which is a selection effect even though nobody selected it. The
    comparison below is the size of it. If the full-history Sharpe is far below
    the overlap-window Sharpe, then section 2's answer is partly a statement
    about 2021-2024 rather than about allocation, and the write-up has to say so.

    Costs come from the same place the panel's do. This read the mid-market
    cache by default until the option spread existed, which made section 5
    quote a different sleeve from section 1 - a small inconsistency, and the
    kind that is invisible once it is in a table.
    """
    full = volcarry_monthly(cache=cache, spread_vol_pts=spread_vol_pts)
    full.index = full.index.to_period("M").to_timestamp("M")
    window = full if overlap is None else full.loc[overlap.index]
    excluded = full.drop(window.index, errors="ignore")
    worst = excluded.nsmallest(3) if len(excluded) else pd.Series(dtype=float)
    return {
        "full": {"n_months": int(len(full)), **performance(full)},
        "overlap": {"n_months": int(len(window)), **performance(window)},
        "worst_excluded": {str(d.date()): float(v) for d, v in worst.items()},
        "sharpe_inflation": (performance(window)["sharpe"]
                             - performance(full)["sharpe"]),
    }


def lookback_sensitivity(panel: pd.DataFrame,
                         lookbacks=(12, 18, 24, 30)) -> list[dict]:
    """Does max Sharpe keep winning as the estimation window changes?

    A single window is a single draw. notes/allocator.md makes the same move
    for the same reason: its headline result is quoted as holding at every
    window tested, which is a much stronger claim than holding at one.
    """
    rows = []
    for lb in lookbacks:
        if len(panel) - lb < 6:
            continue
        result = walk_forward(panel, lookback=lb)
        rows.append({
            "lookback": lb, "months_traded": result["months_traded"],
            **{name: r["sharpe"] for name, r in result["schemes"].items()},
        })
    return rows


# ---------- reporting ----------

def print_sleeves(panel: pd.DataFrame, gap: dict, dr: float) -> None:
    print(f"\n1. THE TWO SLEEVES   {panel.index[0]:%Y-%m} to "
          f"{panel.index[-1]:%Y-%m}, {len(panel)} months")
    print(f"\n{'sleeve':<12} {'ann return':>11} {'ann vol':>9} {'sharpe':>8} "
          f"{'worst month':>12} {'hit rate':>9}")
    for name in panel.columns:
        p = performance(panel[name])
        print(f"{name:<12} {p['ann_return']:>10.2%} {p['ann_vol']:>9.2%} "
              f"{p['sharpe']:>8.2f} {p['worst_month']:>12.2%} "
              f"{p['hit_rate']:>9.0%}")
    print(f"\n   correlation between them: {gap['correlation']:+.3f}")
    print(f"   diversification ratio:    {dr:.2f}   (1.00 is no benefit, "
          f"1.41 is the ceiling for two sleeves)")


def print_walk_forward(result: dict) -> None:
    print(f"\n2. ALLOCATED OUT OF SAMPLE   {result['months_traded']} months, "
          f"{result['first']} to {result['last']}, "
          f"{LOOKBACK_MONTHS}-month trailing estimate")
    print(f"\n{'scheme':<14} {'ann return':>11} {'ann vol':>9} {'sharpe':>8} "
          f"{'worst month':>12} {'weight on statarb':>19}")
    rows = sorted(result["schemes"].items(), key=lambda kv: -kv[1]["sharpe"])
    for name, r in rows:
        print(f"{name:<14} {r['ann_return']:>10.2%} {r['ann_vol']:>9.2%} "
              f"{r['sharpe']:>8.2f} {r['worst_month']:>12.2%} "
              f"{r['weight_on_statarb']:>18.0%}")


def print_blind_spot(result: dict, panel: pd.DataFrame) -> None:
    schemes = result["schemes"]
    risk_based = ["inverse vol", "risk parity", "HRP", "min variance"]
    print(f"\n3. WHY THE RISK-BASED METHODS LOSE HERE")
    worst = max(risk_based, key=lambda n: schemes[n]["weight_on_statarb"])
    print(f"   None of {', '.join(risk_based)} reads an expected return. They "
          f"weight by\n   volatility, and the sleeve that earns nothing is "
          f"also the quieter one:")
    for name in panel.columns:
        print(f"      {name:<10} {panel[name].std(ddof=1) * np.sqrt(12):>7.2%} "
              f"annualized vol, {performance(panel[name])['ann_return']:>+7.2%} "
              f"annualized return")
    print(f"\n   So they allocate toward it. {worst} puts "
          f"{schemes[worst]['weight_on_statarb']:.0%} of the book in the sleeve "
          f"with\n   the lower return, and max Sharpe puts "
          f"{schemes['max Sharpe']['weight_on_statarb']:.0%}.")


def print_regime_test(gap: dict) -> None:
    print(f"\n4. THE TEST THAT SAYS WHICH REGIME YOU ARE IN")
    print(f"   return gap      {gap['annualized_gap']:>+8.2%} a year "
          f"({gap['monthly_gap']:+.4%} a month)")
    print(f"   standard error  {gap['std_error']:>8.4%} a month, on "
          f"{gap['n_months']} months")
    print(f"   t-statistic     {gap['tstat']:>8.2f}")
    verdict = ("large enough to allocate on" if abs(gap["tstat"]) > 2
               else "suggestive but not decisive" if abs(gap["tstat"]) > 1
               else "indistinguishable from noise")
    print(f"\n   The difference in mean return is {verdict}. That is the "
          f"condition under\n   which mean-variance can beat equal weighting, "
          f"and it is computable before\n   choosing an allocator rather than "
          f"after.")


def print_window_bias(bias: dict) -> None:
    full, win = bias["full"], bias["overlap"]
    print(f"\n5. WHAT THE OVERLAP WINDOW LEAVES OUT")
    print(f"\n{'volcarry over':<22} {'months':>7} {'ann return':>11} "
          f"{'ann vol':>9} {'sharpe':>8} {'worst month':>12}")
    for label, r in (("its full history", full), ("the overlap only", win)):
        print(f"{label:<22} {r['n_months']:>7} {r['ann_return']:>10.2%} "
              f"{r['ann_vol']:>9.2%} {r['sharpe']:>8.2f} "
              f"{r['worst_month']:>12.2%}")
    if bias["worst_excluded"]:
        months = ", ".join(f"{d[:7]} {v:+.2%}"
                           for d, v in bias["worst_excluded"].items())
        print(f"\n   Worst months excluded by the overlap: {months}")
    print(f"   Sharpe inflation from the window: "
          f"{bias['sharpe_inflation']:+.2f}")


def print_lookback(rows: list[dict]) -> None:
    if not rows:
        return
    names = [k for k in rows[0] if k not in ("lookback", "months_traded")]
    print(f"\n6. AT EVERY ESTIMATION WINDOW   (out-of-sample Sharpe)")
    header = "".join(f"{n[:12]:>14}" for n in names)
    print(f"\n{'lookback':>9} {'months':>7}{header}")
    for r in rows:
        cells = "".join(f"{r[n]:>14.2f}" for n in names)
        print(f"{r['lookback']:>8}m {r['months_traded']:>7}{cells}")


def spread_sensitivity(spreads=SPREAD_VOL_SWEEP,
                       lookback: int = LOOKBACK_MONTHS) -> list[dict]:
    """How wrong the option spread has to be before the conclusion moves.

    The two sleeves were never cost-matched. statarb pays bid-ask on every
    share it trades; volcarry charged its delta-hedging in basis points but
    sold the option itself at the mid. So the return gap between them was
    measured with one sleeve paying its spreads and the other not, and that
    is the most obvious objection to section 3.

    Charging it is one parameter, and the useful form of the question is not
    "what is the right spread" - nobody has a long history of SPY option
    quotes to settle it - but "how large would it have to be to change the
    answer". That turns an unmeasured hole into a bounded one, which is the
    most an honest study can do with a number it cannot observe.

    Note what does NOT move: the correlation and the diversification ratio.
    A per-month half-spread is very nearly a constant drag, so it shifts the
    mean and leaves the covariance alone. That is exactly why it cannot
    reorder the risk-based allocators against each other - it only moves the
    input that mean-variance is the one to read.
    """
    rows = []
    for spread in spreads:
        panel = sleeve_panel(spread_vol_pts=spread)
        schemes = walk_forward(panel, lookback=lookback)["schemes"]
        gap = return_gap_tstat(panel)
        rows.append({
            "spread_vol_pts": spread,
            "volcarry": performance(panel["volcarry"]),
            "annualized_gap": gap["annualized_gap"],
            "gap_tstat": gap["tstat"],
            "correlation": gap["correlation"],
            "diversification_ratio": diversification_ratio(panel),
            "max_sharpe": schemes["max Sharpe"]["sharpe"],
            "equal_weight": schemes["equal weight"]["sharpe"],
            "w_statarb_max_sharpe": schemes["max Sharpe"]["weight_on_statarb"],
            "w_statarb_inverse_vol": schemes["inverse vol"]["weight_on_statarb"],
        })
    return rows


def print_spread_sensitivity(rows: list[dict]) -> None:
    print(f"\n7. THE COST THE SLEEVES WERE NOT MATCHED ON")
    print(f"   volcarry sold the option at the mid. Charging the half-spread, "
          f"in vol points:")
    print(f"\n{'half-spread':>11} {'volcarry ret':>13} {'sharpe':>7} "
          f"{'gap t':>7} {'maxSharpe':>10} {'1/N':>7} "
          f"{'w(statarb) MV':>14} {'w(statarb) IV':>14}")
    for r in rows:
        print(f"{r['spread_vol_pts']:>11.2f} {r['volcarry']['ann_return']:>12.2%} "
              f"{r['volcarry']['sharpe']:>7.2f} {r['gap_tstat']:>7.2f} "
              f"{r['max_sharpe']:>10.2f} {r['equal_weight']:>7.2f} "
              f"{r['w_statarb_max_sharpe']:>13.0%} "
              f"{r['w_statarb_inverse_vol']:>13.0%}")

    live = [r for r in rows if r["gap_tstat"] >= 2.0]
    if live:
        edge = max(r["spread_vol_pts"] for r in live)
        print(f"\n   The return gap stays significant out to a half-spread of "
              f"{edge:.2f} vol points,")
        print(f"   which is about {edge / 0.10:.0f}x SPY's front-month market. "
              f"At the realistic 0.10 the")
        print(f"   sleeve gives up {rows[0]['volcarry']['ann_return'] - rows[1]['volcarry']['ann_return']:.2%} "
              f"of return and {rows[0]['volcarry']['sharpe'] - rows[1]['volcarry']['sharpe']:.2f} of Sharpe.")
    print(f"\n   The last column is the finding from section 4 again, harder: "
          f"inverse volatility")
    print(f"   holds the same weight on statarb at EVERY spread, including the "
          f"ones where the")
    print(f"   other sleeve stops making money. It is not reacting slowly - it "
          f"is not reacting.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true",
                       help="12-month estimation window instead of 24")
    args = parser.parse_args()
    lookback = 12 if args.quick else LOOKBACK_MONTHS

    panel = sleeve_panel()
    gap = return_gap_tstat(panel)
    dr = diversification_ratio(panel)
    print_sleeves(panel, gap, dr)

    result = walk_forward(panel, lookback=lookback)
    print_walk_forward(result)
    print_blind_spot(result, panel)
    print_regime_test(gap)
    bias = window_bias(overlap=panel)
    print_window_bias(bias)
    sens = lookback_sensitivity(panel)
    print_lookback(sens)
    spreads = spread_sensitivity(lookback=lookback)
    print_spread_sensitivity(spreads)

    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump({"panel_start": str(panel.index[0].date()),
                   "panel_end": str(panel.index[-1].date()),
                   "n_months": len(panel),
                   "sleeves": {c: performance(panel[c]) for c in panel.columns},
                   "gap": gap, "diversification_ratio": dr,
                   "lookback_months": lookback,
                   "walk_forward": result,
                   "window_bias": bias,
                   "lookback_sensitivity": sens,
                   "spread_sensitivity": spreads}, fh, indent=1, default=float)

    # Also on its own, because section 7 is the one table someone is likely to
    # want without parsing the whole run.
    with open("results/spread_sensitivity.json", "w") as fh:
        json.dump(spreads, fh, indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
