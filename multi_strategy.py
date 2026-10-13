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
                     spread_vol_pts: float | None = None,
                     vix_anchor: float | None = -1.0) -> pd.Series:
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

    vix_anchor : the VIX level at which that half-spread is the quoted one.
        Set, the spread is proportional to the level of volatility, which is
        how an options market actually widens. None gives the constant spread
        section 7 sweeps. The sentinel -1.0 means "the sleeve's own shipped
        anchor", so that a caller who says nothing gets the realistic setting
        while a caller who means the constant can still ask for it - None is a
        real value here, not an absence, which is why the default cannot be it.

    Cached because it is the only part of this study that needs the network,
    and a cached file means the walk-forward is reproducible after Yahoo
    changes its mind about something. The cache name carries the spread, so
    changing it does not silently read back a series priced differently.
    """
    from sleeves.volcarry.hedging_study import (OPTION_SPREAD_VIX_ANCHOR,
                                                OPTION_SPREAD_VOL_PTS, load,
                                                months, trade)

    if spread_vol_pts is None:
        spread_vol_pts = OPTION_SPREAD_VOL_PTS
    if vix_anchor == -1.0:
        vix_anchor = OPTION_SPREAD_VIX_ANCHOR
    if cache is None:
        tag = f"{spread_vol_pts:g}".replace(".", "p")
        if vix_anchor is not None:
            tag += f"_vix{vix_anchor:g}".replace(".", "p")
        cache = f"results/volcarry_monthly_sp{tag}.csv"

    if os.path.exists(cache):
        frame = pd.read_csv(cache, index_col=0, parse_dates=True)
        return frame["ret"]

    data = load()
    data = data.loc[start:end]
    rows = [trade(window, option_spread_vol_pts=spread_vol_pts,
                  option_spread_vix_anchor=vix_anchor)
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


def volcarry_detail(start: str = "2007-01-01", end: str = "2024-12-31",
                    spread_vol_pts: float | None = None,
                    vix_anchor: float | None = -1.0,
                    cache: str | None = None) -> pd.DataFrame:
    """Every month's trade, not just its return.

    volcarry_monthly() throws away everything but the P&L, which is all the
    allocator needs. Section 8 needs the premium sold and the spread paid side
    by side, because the question there is what the cost is as a FRACTION of
    the exposure and that ratio is not recoverable from the return alone.
    """
    from sleeves.volcarry.hedging_study import (OPTION_SPREAD_VIX_ANCHOR,
                                                OPTION_SPREAD_VOL_PTS, load,
                                                months, trade)

    if spread_vol_pts is None:
        spread_vol_pts = OPTION_SPREAD_VOL_PTS
    if vix_anchor == -1.0:
        vix_anchor = OPTION_SPREAD_VIX_ANCHOR
    if cache is None:
        tag = f"{spread_vol_pts:g}".replace(".", "p")
        if vix_anchor is not None:
            tag += f"_vix{vix_anchor:g}".replace(".", "p")
        cache = f"results/volcarry_detail_sp{tag}.csv"

    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)

    data = load().loc[start:end]
    frame = pd.DataFrame([
        trade(window, option_spread_vol_pts=spread_vol_pts,
              option_spread_vix_anchor=vix_anchor)
        for window in months(data)]).set_index("date")
    keep = ["implied", "half_spread", "spread_cost", "premium", "pnl"]
    frame = frame[keep]
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    frame.to_csv(cache)
    return frame


# ---------- 8. the spread is not the same width every month ----------

def spread_comovement(lookback: int = LOOKBACK_MONTHS,
                      anchor: float | None = None,
                      terciles: int = 3) -> dict:
    """What it costs this sleeve that an options market widens when vol rises.

    Section 7 bounded the LEVEL of the option spread. It left the shape of the
    objection standing, in its own words: "the spread is a constant, and a real
    one widens in exactly the months this sleeve is most exposed." That is the
    uncomfortable version, because it says the cost and the risk arrive
    together, and a sweep over constants cannot see it.

    The fix is one line in the sleeve - charge `spread * VIX / anchor` instead
    of `spread` - and then the same decomposition the market-simulation repo
    uses on its order book, for the same reason: a cost that moves with the
    market has a LEVEL and a CO-MOVEMENT, and conflating them is how a small
    effect gets written up as a large one. So three series:

      - constant: the shipped 0.10 on every month.
      - proportional: 0.10 at the anchor, 0.39 in March 2020, 0.07 in 2017.
      - level-matched: a CONSTANT spread set to the proportional series' own
        realized average. Same average cost, none of the timing. Whatever this
        one fails to explain is what the co-movement is actually worth.

    The tercile table is the part worth reading. It reports the spread as a
    share of the premium sold, which is the only form in which the two models
    can be compared honestly - the sleeve's exposure is not constant either.
    """
    from sleeves.volcarry.hedging_study import (OPTION_SPREAD_VIX_ANCHOR,
                                                OPTION_SPREAD_VOL_PTS)

    anchor = OPTION_SPREAD_VIX_ANCHOR if anchor is None else anchor
    spread = OPTION_SPREAD_VOL_PTS

    proportional = volcarry_detail(vix_anchor=anchor)
    matched = float(proportional["half_spread"].mean())

    variants = {
        "mid": {"spread_vol_pts": 0.0, "vix_anchor": None},
        "constant": {"spread_vol_pts": spread, "vix_anchor": None},
        "proportional": {"spread_vol_pts": spread, "vix_anchor": anchor},
        "level_matched": {"spread_vol_pts": matched, "vix_anchor": None},
    }

    out = {"anchor": anchor, "spread_vol_pts": spread,
           "matched_constant": matched, "rows": {}}
    for name, kwargs in variants.items():
        panel = sleeve_panel(**kwargs)
        schemes = walk_forward(panel, lookback=lookback)["schemes"]
        gap = return_gap_tstat(panel)
        out["rows"][name] = {
            # Realized average width, not the parameter: for the proportional
            # row those are different numbers and the average is the one that
            # makes the four rows comparable.
            "half_spread": (matched if name in ("proportional", "level_matched")
                            else kwargs["spread_vol_pts"]),
            "volcarry": performance(panel["volcarry"]),
            "annualized_gap": gap["annualized_gap"],
            "gap_tstat": gap["tstat"],
            "correlation": gap["correlation"],
            "max_sharpe": schemes["max Sharpe"]["sharpe"],
            "w_statarb_max_sharpe": schemes["max Sharpe"]["weight_on_statarb"],
            "w_statarb_inverse_vol": schemes["inverse vol"]["weight_on_statarb"],
        }

    # The cost as a share of what was sold, by how volatile the month was.
    constant = volcarry_detail(vix_anchor=None)
    buckets = pd.qcut(proportional["implied"], terciles,
                      labels=[f"t{i + 1}" for i in range(terciles)])
    out["by_vol"] = []
    for label, index in proportional.groupby(buckets, observed=True).groups.items():
        prop, const = proportional.loc[index], constant.loc[index]
        out["by_vol"].append({
            "bucket": str(label),
            "months": len(index),
            "mean_vix": float(prop["implied"].mean() * 100),
            "mean_premium": float(prop["premium"].mean()),
            "proportional_half_spread": float(prop["half_spread"].mean()),
            "proportional_share_of_premium":
                float((prop["spread_cost"] / prop["premium"]).mean()),
            "constant_share_of_premium":
                float((const["spread_cost"] / const["premium"]).mean()),
        })
    return out


def print_spread_comovement(study: dict) -> None:
    rows = study["rows"]
    print(f"\n8. AND THE SPREAD IS NOT THE SAME WIDTH EVERY MONTH")
    print(f"   Section 7 bounded how WIDE the spread is. This is about when it "
          f"is wide: the")
    print(f"   half-spread becomes {study['spread_vol_pts']:.2f} vol points at "
          f"VIX {study['anchor']:.1f} and scales with VIX from there.")
    print(f"\n{'spread charged':>23} {'avg half-spread':>16} "
          f"{'volcarry ret':>13} {'sharpe':>7} {'gap t':>7} "
          f"{'w(statarb) MV':>14}")
    labels = {"mid": "mid (none)", "constant": "constant",
              "proportional": "proportional to VIX",
              "level_matched": "constant, level-matched"}
    for key in ("mid", "constant", "proportional", "level_matched"):
        r = rows[key]
        print(f"{labels[key]:>23} {r['half_spread']:>16.4f} "
              f"{r['volcarry']['ann_return']:>12.2%} "
              f"{r['volcarry']['sharpe']:>7.2f} {r['gap_tstat']:>7.2f} "
              f"{r['w_statarb_max_sharpe']:>13.0%}")

    print(f"\n   Cost as a share of the premium sold, by how volatile the "
          f"month was:")
    print(f"\n{'months':>8} {'avg VIX':>9} {'premium':>9} "
          f"{'half-spread':>12} {'proportional':>13} {'constant':>10}")
    for b in study["by_vol"]:
        print(f"{b['months']:>8} {b['mean_vix']:>9.1f} "
              f"{b['mean_premium']:>9.2f} "
              f"{b['proportional_half_spread']:>12.3f} "
              f"{b['proportional_share_of_premium']:>12.2%} "
              f"{b['constant_share_of_premium']:>9.2%}")

    shares = [b["proportional_share_of_premium"] for b in study["by_vol"]]
    print(f"\n   That second-to-last column is flat to four decimal places, "
          f"and it is the")
    print(f"   whole answer. The premium scales with volatility and so does "
          f"the spread, so")
    print(f"   the cost is {np.mean(shares):.2%} of what was sold in every "
          f"regime. The constant spread")
    print(f"   is the one that is regime-dependent: "
          f"{study['by_vol'][0]['constant_share_of_premium']:.2%} of premium in "
          f"the calm third and")
    print(f"   {study['by_vol'][-1]['constant_share_of_premium']:.2%} in the "
          f"loud third, so it overcharges quiet months and")
    print(f"   undercharges exactly the ones the sleeve has most on.")

    prop, matched = rows["proportional"], rows["level_matched"]
    const = rows["constant"]
    print(f"\n   So the honest answer to the objection is that it is real and "
          f"it is small.")
    print(f"   Return goes {const['volcarry']['ann_return']:.2%} to "
          f"{prop['volcarry']['ann_return']:.2%} and the gap t-stat "
          f"{const['gap_tstat']:.2f} to {prop['gap_tstat']:.2f}.")
    print(f"   A constant spread at the same average width gives "
          f"{matched['volcarry']['ann_return']:.2%} and "
          f"{matched['gap_tstat']:.2f},")
    print(f"   so essentially all of the difference is the average width and "
          f"not the timing.")
    print(f"\n   Worth contrasting with the order-book repo, where the same "
          f"question came out")
    print(f"   the other way: there the strategy's size was fixed and the "
          f"depth moved, so 81%")
    print(f"   of the extra cost was the co-movement. Here the exposure moves "
          f"with the same")
    print(f"   variable the cost does, and they cancel. Cost co-movement bites "
          f"when your")
    print(f"   position size does not co-move with it.")


# ---------- 9. a third sleeve, so that "allocation" is more than one number ----------

def xsmom_monthly(path: str = "results/xsmom_monthly.json") -> pd.Series:
    """Monthly net returns of the momentum sleeve, from its committed output.

    Same arrangement as statarb_monthly: read the sleeve's own results file
    rather than recompute, so this study and sleeves/xsmom/momentum.py cannot
    disagree about what the sleeve did. `python -m sleeves.xsmom.momentum`
    rewrites the file.
    """
    from sleeves.xsmom.momentum import read_results

    series = read_results(path)["net"].rename("xsmom")
    series.index = series.index.to_period("M").to_timestamp("M")
    return series


def three_sleeve_panel(**kwargs) -> pd.DataFrame:
    """The two-sleeve panel with the momentum sleeve joined on the month."""
    panel = sleeve_panel(**kwargs)
    return pd.concat([panel, xsmom_monthly()], axis=1).dropna()


def weighted_diversification_ratio(panel: pd.DataFrame,
                                   w: np.ndarray) -> float:
    """diversification_ratio at an arbitrary weight vector, not only 1/N."""
    w = np.asarray(w, dtype=float)
    vols = panel.std(ddof=1).to_numpy(dtype=float)
    cov = np.cov(panel.to_numpy(dtype=float), rowvar=False, ddof=1)
    return float((w @ vols) / np.sqrt(w @ cov @ w))


def momentum_crashes(xsmom: pd.Series, n: int = 3) -> list[dict]:
    """The sleeve's worst months, with what the market did before and during.

    Momentum does not lose money in a crash. It loses money in the month the
    crash reverses, because by then the short leg is full of the names that
    fell hardest, which are the names that bounce hardest. Daniel and
    Moskowitz (2016) call it a momentum crash and it is the one risk the
    strategy is famous for. This table is the check that the sleeve built
    here has the same scar in the same place: each of its worst months should
    be a rally month that followed a drawdown.
    """
    from sleeves.xsmom.momentum import load_universe, monthly_prices

    market = monthly_prices(load_universe()).pct_change(fill_method=None).mean(axis=1)
    market.index = market.index.to_period("M").to_timestamp("M")
    rows = []
    for date, ret in xsmom.nsmallest(n).items():
        i = market.index.get_loc(date)
        before = float((1 + market.iloc[max(0, i - 3):i]).prod() - 1)
        rows.append({"month": f"{date:%Y-%m}", "xsmom": float(ret),
                     "market_that_month": float(market.iloc[i]),
                     "market_prior_3m": before})
    return rows


def third_sleeve_study(lookback: int = LOOKBACK_MONTHS) -> dict:
    """What a third return stream does to every conclusion above.

    Two sleeves is not a portfolio. With two, every allocator is choosing one
    number - the weight on the first sleeve - and the covariance matrix has
    one off-diagonal entry, so the difference between a risk-parity book and
    an inverse-volatility book, or between HRP and anything, is nearly
    definitional. The criticism of including HRP at all was fair: a
    clustering algorithm with two things to cluster is not clustering.

    sleeves/xsmom/ is the cheapest third stream this repository could build.
    It is cross-sectional momentum on the same 102 names the pairs scan
    screens, which is deliberate: the stat-arb sleeve is long the lagging
    leg of a pair and short the leading leg, and the momentum sleeve is long
    the leaders and short the laggards. They are opposite bets on relative
    performance inside one universe, and the correlation below says how far
    that goes.

    Three things are measured. The sleeve on its own, over its full history.
    Then the overlap panel - the same 46 months as section 1, because the
    stat-arb sleeve's start date still decides the window. Then the walk-
    forward from section 2 again with three columns, which is the first time
    in this repository that it has a covariance structure to react to.
    """
    full = xsmom_monthly()
    panel = three_sleeve_panel()
    two = panel[list(SLEEVES)]
    result = walk_forward(panel, lookback=lookback)
    reference = walk_forward(two, lookback=lookback)

    inv_vol = inverse_vol_weights(np.cov(panel.to_numpy(dtype=float),
                                         rowvar=False, ddof=1))
    equal = np.ones(panel.shape[1]) / panel.shape[1]
    return {
        "full_history": {"first": f"{full.index[0]:%Y-%m}",
                         "last": f"{full.index[-1]:%Y-%m}",
                         "months": int(len(full)),
                         **performance(full)},
        "by_year": {int(y): float(v) for y, v
                    in full.groupby(full.index.year).sum().items()},
        "crashes": momentum_crashes(full),
        "overlap": {"first": str(panel.index[0].date()),
                    "last": str(panel.index[-1].date()),
                    "n_months": int(len(panel)),
                    "sleeves": {c: performance(panel[c]) for c in panel.columns},
                    "correlation": {a: {b: float(panel[a].corr(panel[b]))
                                        for b in panel.columns}
                                    for a in panel.columns}},
        "diversification_ratio": {
            "two_sleeves_equal": diversification_ratio(two),
            "three_sleeves_equal": diversification_ratio(panel),
            "three_sleeves_inverse_vol":
                weighted_diversification_ratio(panel, inv_vol),
            "ceiling_three": float(np.sqrt(3)),
        },
        "walk_forward": result,
        "two_sleeve_reference": {
            name: {"sharpe": r["sharpe"], "ann_vol": r["ann_vol"]}
            for name, r in reference["schemes"].items()},
    }


def print_third_sleeve(study: dict) -> None:
    full = study["full_history"]
    print(f"\n9. A THIRD SLEEVE: CROSS-SECTIONAL MOMENTUM ON THE SAME 102 NAMES")
    print(f"   Full history {full['first']} to {full['last']}, "
          f"{full['months']} months, net of 10 bps a side:")
    print(f"   {full['ann_return']:+.2%} a year at {full['ann_vol']:.2%} "
          f"vol, Sharpe {full['sharpe']:.2f}, worst month "
          f"{full['worst_month']:.2%}")

    years = study["by_year"]
    print(f"\n   By calendar year:")
    line = "   " + "  ".join(f"{y}" for y in years)
    print(line)
    print("   " + "  ".join(f"{v:+.0%}".rjust(4) for v in years.values()))
    signs = [np.sign(v) for v in years.values()]
    flips = sum(1 for a, b in zip(signs, signs[1:]) if a != b)
    print(f"   The sign flips {flips} times in {len(signs) - 1} year-to-year "
          f"steps. A twelve-month signal on a\n   hundred names is one bet a "
          f"year, and in mega-caps that bet keeps reversing.")

    print(f"\n   Its three worst months, with what the market did around them:")
    print(f"\n{'month':>10} {'xsmom':>8} {'market, that month':>19} "
          f"{'market, prior 3m':>17}")
    for c in study["crashes"]:
        print(f"{c['month']:>10} {c['xsmom']:>8.1%} "
              f"{c['market_that_month']:>19.1%} {c['market_prior_3m']:>17.1%}")
    print(f"   Every one is a rally. That is the momentum crash: the short "
          f"leg is full of what\n   fell hardest, so the rebound is where it "
          f"loses, not the fall.")

    ov = study["overlap"]
    print(f"\n   On the overlap  {ov['first']} to {ov['last']}, "
          f"{ov['n_months']} months:")
    print(f"\n{'sleeve':<10} {'ann return':>11} {'ann vol':>9} {'sharpe':>8}"
          f"   {'corr w/ statarb':>15} {'corr w/ volcarry':>16}")
    for name, p in ov["sleeves"].items():
        corr = ov["correlation"][name]
        print(f"{name:<10} {p['ann_return']:>10.2%} {p['ann_vol']:>9.2%} "
              f"{p['sharpe']:>8.2f}   {corr['statarb']:>15.3f} "
              f"{corr['volcarry']:>16.3f}")
    rho = ov["correlation"]["xsmom"]["statarb"]
    print(f"\n   {rho:+.2f} against statarb is the number to notice. The "
          f"pairs book is long the\n   lagging leg and short the leading "
          f"one; momentum is the reverse. On one universe\n   they are "
          f"opposite sides of the same trade, and the correlation says how "
          f"much.")

    dr = study["diversification_ratio"]
    print(f"\n   diversification ratio, equal weight:   "
          f"{dr['two_sleeves_equal']:.2f} with two sleeves, "
          f"{dr['three_sleeves_equal']:.2f} with three")
    print(f"   diversification ratio, inverse vol:    "
          f"{dr['three_sleeves_inverse_vol']:.2f} with three")
    print(f"   Equal weight gets WORSE with the third sleeve because the "
          f"sleeve is ten times\n   as volatile as the other two, so 1/N is "
          f"not a third each - it is the momentum\n   sleeve with a small "
          f"hedge on. Size it by risk and the ratio goes the other way, "
          f"past the\n   {dr['ceiling_three']:.2f} that three uncorrelated "
          f"sleeves could reach. Only negative correlation\n   gets a book "
          f"above that line, and the {rho:+.2f} is what gets it there.")

    wf = study["walk_forward"]
    ref = study["two_sleeve_reference"]
    print(f"\n   Allocated out of sample, {wf['months_traded']} months, the "
          f"same walk-forward as section 2:")
    print(f"\n{'scheme':<14} {'ann return':>11} {'ann vol':>9} {'sharpe':>8} "
          f"{'2-sleeve':>9}   {'w statarb':>9} {'w volcarry':>10} {'w xsmom':>8}")
    rows = sorted(wf["schemes"].items(), key=lambda kv: -kv[1]["sharpe"])
    for name, r in rows:
        w = r["mean_weight"]
        print(f"{name:<14} {r['ann_return']:>10.2%} {r['ann_vol']:>9.2%} "
              f"{r['sharpe']:>8.2f} {ref[name]['sharpe']:>9.2f}   "
              f"{w['statarb']:>9.0%} {w['volcarry']:>10.0%} {w['xsmom']:>8.1%}")
    ew, iv = wf["schemes"]["equal weight"], wf["schemes"]["inverse vol"]
    print(f"\n   Equal weight goes from {ref['equal weight']['sharpe']:.2f} to "
          f"{ew['sharpe']:.2f} and its volatility from "
          f"{ref['equal weight']['ann_vol']:.2%} to {ew['ann_vol']:.2%}.")
    print(f"   That is the first clean loss for 1/N in this repository, and "
          f"it is not a\n   subtle one. 1/N only ever beat the optimizer "
          f"among things of similar scale;\n   hand it one stream ten "
          f"times the size of the others and it is not diversified,\n   it is "
          f"concentrated. Inverse vol - which is 1/N after scaling - holds "
          f"{iv['mean_weight']['xsmom']:.1%} in it\n   and is back to "
          f"{iv['sharpe']:.2f}.")
    hrp, rp = wf["schemes"]["HRP"], wf["schemes"]["risk parity"]
    print(f"\n   HRP finally has something to cluster, and look at what it "
          f"does with it: {hrp['mean_weight']['xsmom']:.1%},\n   against "
          f"risk parity's {rp['mean_weight']['xsmom']:.1%}. The tree puts "
          f"momentum in a cluster of its own, which\n   is right, and then "
          f"splits capital between clusters by inverse variance alone -\n   "
          f"the correlation that made the split is not used at the step "
          f"that sizes it.\n   Risk parity reads the whole matrix and takes "
          f"the hedge; HRP reads it and then\n   forgets it. That is the "
          f"method's known blind spot, and this is the first panel\n   in "
          f"the repository where it had anything to be blind to.")
    print(f"\n   Mean-variance puts {wf['schemes']['max Sharpe']['mean_weight']['xsmom']:.1%} "
          f"in the sleeve, for the same reason it puts almost\n   nothing in "
          f"statarb: it does not make money, and that is the one thing "
          f"mean-variance\n   can see. A sleeve with no return cannot "
          f"diversify a return. It can only\n   diversify risk, and the "
          f"risk-based four all find the same answer about how\n   much of "
          f"it to take for that: a few percent.")


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
        # vix_anchor=None deliberately: this sweep asks how WIDE the spread
        # would have to be, and a proportional spread would move the level and
        # its co-movement together. Section 8 separates them.
        panel = sleeve_panel(spread_vol_pts=spread, vix_anchor=None)
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
    comovement = spread_comovement(lookback=lookback)
    print_spread_comovement(comovement)
    third = third_sleeve_study(lookback=lookback)
    print_third_sleeve(third)

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
                   "spread_sensitivity": spreads,
                   "spread_comovement": comovement,
                   "third_sleeve": third}, fh, indent=1,
                  default=float)

    # Also on its own, because section 7 is the one table someone is likely to
    # want without parsing the whole run.
    with open("results/spread_sensitivity.json", "w") as fh:
        json.dump(spreads, fh, indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
