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


# ---------- 10. what the -0.46 is worth in return units ----------

def market_monthly() -> pd.Series:
    """Equal-weight monthly return of the 102-name universe, as the market.

    The three sleeves all trade inside this list, so the average of the list
    is the market they can be long or short. SPY would be the conventional
    choice and is not used, on purpose: the pairs and momentum books cannot
    hold anything outside these names, and a beta to an index that includes
    400 other stocks would mix "is the book market-neutral" with "do these
    102 names track the index". The equal-weight average of what the books
    can actually hold is the cleaner question.
    """
    from sleeves.xsmom.momentum import load_universe, monthly_prices

    market = monthly_prices(load_universe()).pct_change(fill_method=None).mean(axis=1)
    market.index = market.index.to_period("M").to_timestamp("M")
    return market.rename("market")


def regress(y: pd.Series, factors: pd.DataFrame) -> dict:
    """OLS of one sleeve on one or more factors, with the usual table.

    Plain least squares with classical standard errors. The sleeves are
    monthly and non-overlapping, so there is no induced autocorrelation to
    correct for, and with 46 observations a Newey-West bandwidth would be a
    bigger judgement call than the thing it corrects. The t-statistics are
    read with that sample size in mind.

    Returns alpha both monthly and annualized, every beta with its t, the
    R-squared, and the Sharpe of the sleeve with its factor exposure hedged
    out: y minus beta times each factor, which is what the sleeve would have
    earned if a desk had neutralized it. That last number is the point of
    the exercise. A correlation says two things move together; the hedged
    Sharpe says whether removing the comovement would have helped.
    """
    frame = pd.concat([y, factors], axis=1).dropna()
    yv = frame.iloc[:, 0].to_numpy(dtype=float)
    X = np.column_stack([np.ones(len(frame)), frame.iloc[:, 1:].to_numpy(dtype=float)])
    n, k = X.shape
    coef, *_ = np.linalg.lstsq(X, yv, rcond=None)
    resid = yv - X @ coef
    dof = n - k
    s2 = float(resid @ resid) / dof if dof > 0 else float("nan")
    cov = s2 * np.linalg.inv(X.T @ X)
    se = np.sqrt(np.diag(cov))
    tstats = coef / se
    ss_tot = float(((yv - yv.mean()) ** 2).sum())
    r2 = 1.0 - float(resid @ resid) / ss_tot if ss_tot > 0 else 0.0

    hedged = yv - X[:, 1:] @ coef[1:]
    hedged_series = pd.Series(hedged, index=frame.index)
    names = list(factors.columns)
    return {
        "n_months": int(n),
        "alpha_monthly": float(coef[0]),
        "alpha_ann": float(coef[0]) * MONTHS_PER_YEAR,
        "alpha_t": float(tstats[0]),
        "beta": {f: float(b) for f, b in zip(names, coef[1:])},
        "beta_t": {f: float(t) for f, t in zip(names, tstats[1:])},
        "r2": float(r2),
        "raw": performance(frame.iloc[:, 0]),
        "hedged": performance(hedged_series),
    }


def factor_exposure_study(panel: pd.DataFrame | None = None,
                          market: pd.Series | None = None) -> dict:
    """Each sleeve regressed on the market, and on the momentum sleeve.

    Section 9 found a -0.46 correlation between the stat-arb and momentum
    sleeves and called nearly half of the stat-arb variance a momentum bet.
    That was a correlation read as a variance share, and the two are not the
    same number: the share of variance one factor explains is the correlation
    SQUARED, so -0.46 is a fifth, not a half. This section puts the regression
    on the page so the claim is a number with a standard error instead of a
    sentence.

    Three things come out of it.

    The beta, in return units. A correlation of -0.46 between a 1.5%-vol
    sleeve and an 18%-vol sleeve is a beta of -0.04: a dollar of stat-arb is
    four cents short of momentum. Small in dollars, large as a share of a very
    quiet book.

    What hedging it would have done. The hedged Sharpe is the sleeve minus
    its beta times the factor. If the comovement is pure noise, hedging it
    removes variance and leaves return, and the Sharpe goes up. If the sleeve
    has been earning something FROM the tilt, hedging takes that away too.
    Over this window momentum lost money, so a short-momentum tilt was a
    small source of return, and taking it out makes the stat-arb sleeve look
    worse, not better.

    Market beta, for all three. Every sleeve here is market-neutral on paper:
    long and short legs of equal size, or an option delta-hedged daily. The
    regression on the equal-weight universe is the check that the paper is
    right, and the only one of the three where the answer is interesting is
    the one that is short volatility, because a short call is short the
    market on the way down whatever the hedge says at the open.
    """
    if panel is None:
        panel = three_sleeve_panel()
    if market is None:
        market = market_monthly()
    market = market.reindex(panel.index)

    out: dict = {"n_months": int(len(panel)),
                 "window": {"first": str(panel.index[0].date()),
                            "last": str(panel.index[-1].date())}}
    out["on_market"] = {c: regress(panel[c], market.to_frame())
                        for c in panel.columns}
    out["statarb_on_xsmom"] = regress(panel["statarb"],
                                      panel[["xsmom"]])
    out["volcarry_on_xsmom"] = regress(panel["volcarry"],
                                       panel[["xsmom"]])
    two = pd.concat([market, panel["xsmom"]], axis=1)
    out["on_market_and_xsmom"] = {c: regress(panel[c], two)
                                  for c in ("statarb", "volcarry")}
    rho = float(panel["statarb"].corr(panel["xsmom"]))
    out["correlation_vs_variance_share"] = {
        "correlation": rho,
        "variance_share": rho * rho,
        "section_9_said": "nearly half",
    }
    return out


def print_factor_exposures(study: dict) -> None:
    cv = study["correlation_vs_variance_share"]
    print(f"\n10. THE -0.46 IN RETURN UNITS")
    print(f"   Section 9 read a correlation of {cv['correlation']:+.2f} as "
          f"\"nearly half\" of the stat-arb variance.")
    print(f"   The variance share is the correlation squared: "
          f"{cv['variance_share']:.0%}. A fifth, not a half.")

    r = study["statarb_on_xsmom"]
    print(f"\n   statarb on xsmom, {r['n_months']} months:")
    print(f"     beta {r['beta']['xsmom']:+.3f}  (t = {r['beta_t']['xsmom']:+.1f})"
          f"   alpha {r['alpha_ann']:+.2%} a year  (t = {r['alpha_t']:+.1f})"
          f"   R2 {r['r2']:.2f}")
    print(f"     Sharpe raw {r['raw']['sharpe']:+.2f} at {r['raw']['ann_vol']:.2%} vol"
          f"  ->  momentum-hedged {r['hedged']['sharpe']:+.2f} at "
          f"{r['hedged']['ann_vol']:.2%} vol")
    print(f"     Hedging the tilt cuts the vol and cuts the return by more. "
          f"The short-momentum\n     position was earning money this window "
          f"because momentum was losing it.")

    print(f"\n   Market beta of each sleeve, on the equal-weight universe:")
    print(f"\n{'sleeve':>10} {'beta':>8} {'t':>6} {'alpha/yr':>10} {'t':>6} "
          f"{'R2':>6} {'Sharpe':>8} {'hedged':>8}")
    for name, r in study["on_market"].items():
        print(f"{name:>10} {r['beta']['market']:>+8.3f} "
              f"{r['beta_t']['market']:>+6.1f} {r['alpha_ann']:>+10.2%} "
              f"{r['alpha_t']:>+6.1f} {r['r2']:>6.2f} "
              f"{r['raw']['sharpe']:>+8.2f} {r['hedged']['sharpe']:>+8.2f}")

    print(f"\n   Both factors at once:")
    print(f"\n{'sleeve':>10} {'b_mkt':>8} {'t':>6} {'b_mom':>8} {'t':>6} "
          f"{'alpha/yr':>10} {'t':>6} {'R2':>6}")
    for name, r in study["on_market_and_xsmom"].items():
        print(f"{name:>10} {r['beta']['market']:>+8.3f} "
              f"{r['beta_t']['market']:>+6.1f} {r['beta']['xsmom']:>+8.3f} "
              f"{r['beta_t']['xsmom']:>+6.1f} {r['alpha_ann']:>+10.2%} "
              f"{r['alpha_t']:>+6.1f} {r['r2']:>6.2f}")


# ---------- 11. beta-hedge the momentum sleeve before it goes in ----------

BETA_WINDOW_MONTHS = 36   # trailing window the hedge ratio is estimated on


def trailing_beta(y: pd.Series, factor: pd.Series,
                  window: int = BETA_WINDOW_MONTHS) -> pd.Series:
    """The OLS beta of y on the factor, using only months before each date.

    Section 10 measured the momentum sleeve's market beta at -0.47 on the
    full 46-month overlap. That is the number a desk would like to hedge with
    and cannot, because it was not known until the window was over. The
    hedge ratio a desk actually has at the start of a month is whatever the
    trailing window said the month before, so that is what this returns: the
    beta at date t is fitted on the `window` months ending at t-1 and is NaN
    until there are that many. Nothing in it has seen month t.
    """
    frame = pd.concat([y, factor], axis=1).dropna()
    yv = frame.iloc[:, 0]
    fv = frame.iloc[:, 1]
    cov = yv.rolling(window).cov(fv)
    var = fv.rolling(window).var()
    beta = (cov / var).shift(1)
    return beta.reindex(y.index).rename("beta")


def beta_hedged(y: pd.Series, factor: pd.Series,
                window: int = BETA_WINDOW_MONTHS) -> pd.DataFrame:
    """y with its trailing beta to the factor taken out, month by month.

    hedged_t = y_t - beta_{t-1} * factor_t. The months before the first
    full window are dropped rather than carried through unhedged, because a
    series that is hedged for most of its life and not for the start is
    neither thing. The beta is kept alongside so the write-up can show the
    path the hedge ratio actually took.
    """
    beta = trailing_beta(y, factor, window)
    frame = pd.concat([y.rename("raw"), factor.rename("factor"), beta], axis=1)
    frame = frame.dropna()
    frame["hedged"] = frame["raw"] - frame["beta"] * frame["factor"]
    return frame


def hedged_momentum_study(lookback: int = LOOKBACK_MONTHS,
                          window: int = BETA_WINDOW_MONTHS) -> dict:
    """The three-sleeve table again, with the momentum sleeve beta-hedged.

    Section 10 ended with a measurable question: the momentum book as built
    carries a market beta of -0.47, a desk would hedge it before allocating
    to it, and hedging changes the panel the allocators see. This runs that.

    The hedge has to be out of sample or the exercise is circular. The -0.47
    is a full-window number, so it is replaced with the trailing 36-month
    beta re-estimated every month on the sleeve's own prior history, which
    goes back to 2014 and so is available before the overlap starts. The
    full-window beta is reported next to the trailing path to show how far
    apart the hedge a desk could run and the hedge the regression suggests
    actually are.

    Three things are then compared raw against hedged: the sleeve on its own
    history, the overlap correlation matrix, and the walk-forward table.
    Hedging an exposure that lost money over the window should raise the
    sleeve's Sharpe, and whether that moves the allocators depends on whether
    they look at returns at all.
    """
    full = xsmom_monthly()
    market = market_monthly()
    hedge = beta_hedged(full, market, window)
    hedged_full = hedge["hedged"].rename("xsmom")

    raw_panel = three_sleeve_panel()
    hedged_panel = pd.concat([raw_panel[list(SLEEVES)], hedged_full],
                             axis=1).dropna()
    # same months on both sides, so the two walk-forwards are comparable
    raw_panel = raw_panel.loc[hedged_panel.index]
    in_window = hedge.loc[hedged_panel.index]

    full_window = regress(raw_panel["xsmom"], market.reindex(raw_panel.index)
                          .to_frame())
    raw_wf = walk_forward(raw_panel, lookback=lookback)
    hedged_wf = walk_forward(hedged_panel, lookback=lookback)

    crash_months = [row["month"] for row in momentum_crashes(full)]
    beta_at_crashes = {m: float(hedge.loc[f"{m}", "beta"].iloc[0])
                       for m in crash_months if m in
                       hedge.index.strftime("%Y-%m")}

    return {
        "beta_window_months": int(window),
        "full_window_beta": full_window["beta"]["market"],
        "full_window_beta_t": full_window["beta_t"]["market"],
        "trailing_beta": {
            "first": str(in_window.index[0].date()),
            "last": str(in_window.index[-1].date()),
            "mean": float(in_window["beta"].mean()),
            "min": float(in_window["beta"].min()),
            "max": float(in_window["beta"].max()),
            "share_of_months_negative": float((in_window["beta"] < 0).mean()),
            "at_overlap_start": float(in_window["beta"].iloc[0]),
            "at_overlap_end": float(in_window["beta"].iloc[-1]),
            "at_crash_months": beta_at_crashes,
        },
        "sleeve_full_history": {
            "months": int(len(hedge)),
            "first": f"{hedge.index[0]:%Y-%m}",
            "last": f"{hedge.index[-1]:%Y-%m}",
            "raw": performance(hedge["raw"]),
            "hedged": performance(hedge["hedged"]),
        },
        "overlap": {
            "n_months": int(len(hedged_panel)),
            "raw": performance(raw_panel["xsmom"]),
            "hedged": performance(hedged_panel["xsmom"]),
            "market_beta_after_hedge": regress(
                hedged_panel["xsmom"],
                market.reindex(hedged_panel.index).to_frame())["beta"]["market"],
            "correlation_raw": {a: {b: float(raw_panel[a].corr(raw_panel[b]))
                                    for b in raw_panel.columns}
                                for a in raw_panel.columns},
            "correlation_hedged": {a: {b: float(hedged_panel[a].corr(hedged_panel[b]))
                                       for b in hedged_panel.columns}
                                   for a in hedged_panel.columns},
            "statarb_hedged_corr": float(raw_panel["statarb"].corr(hedged_panel["xsmom"])),
        },
        "walk_forward_raw": raw_wf,
        "walk_forward_hedged": hedged_wf,
    }


def print_hedged_momentum(study: dict) -> None:
    tb = study["trailing_beta"]
    print(f"\n11. BETA-HEDGE THE MOMENTUM SLEEVE BEFORE IT GOES IN")
    print(f"   Full-window beta to the market, the one section 10 reported: "
          f"{study['full_window_beta']:+.2f} (t {study['full_window_beta_t']:+.1f}).")
    print(f"   Trailing {study['beta_window_months']}-month beta a desk "
          f"could actually have run, over the same {study['overlap']['n_months']} months:")
    print(f"     mean {tb['mean']:+.2f}, range {tb['min']:+.2f} to "
          f"{tb['max']:+.2f}, negative in {tb['share_of_months_negative']:.0%} "
          f"of months,")
    print(f"     {tb['at_overlap_start']:+.2f} at the start of the window and "
          f"{tb['at_overlap_end']:+.2f} at the end.")
    if tb["at_crash_months"]:
        parts = ", ".join(f"{m}: {b:+.2f}" for m, b in tb["at_crash_months"].items())
        print(f"     Hedge ratio going into the sleeve's worst months: {parts}")

    fh = study["sleeve_full_history"]
    print(f"\n   The sleeve on its own history, {fh['first']} to {fh['last']} "
          f"({fh['months']} months):")
    print(f"{'':>14} {'ret/yr':>8} {'vol':>8} {'Sharpe':>8} {'worst mo':>10}")
    for name in ("raw", "hedged"):
        r = fh[name]
        print(f"{name:>14} {r['ann_return']:>+8.2%} {r['ann_vol']:>8.2%} "
              f"{r['sharpe']:>+8.2f} {r['worst_month']:>+10.2%}")

    ov = study["overlap"]
    print(f"\n   On the overlap ({ov['n_months']} months):")
    for name in ("raw", "hedged"):
        r = ov[name]
        print(f"{name:>14} {r['ann_return']:>+8.2%} {r['ann_vol']:>8.2%} "
              f"{r['sharpe']:>+8.2f} {r['worst_month']:>+10.2%}")
    print(f"   Market beta of the hedged sleeve, measured after the fact: "
          f"{ov['market_beta_after_hedge']:+.2f}.")
    print(f"   Correlation with statarb: {ov['correlation_raw']['statarb']['xsmom']:+.2f} "
          f"raw, {ov['correlation_hedged']['statarb']['xsmom']:+.2f} hedged; "
          f"with volcarry: {ov['correlation_raw']['volcarry']['xsmom']:+.2f} raw, "
          f"{ov['correlation_hedged']['volcarry']['xsmom']:+.2f} hedged.")

    print(f"\n   Walk-forward, {lookback_label(study)} months traded, "
          f"raw momentum sleeve against hedged:")
    print(f"\n{'scheme':>14} {'Sharpe raw':>11} {'hedged':>8} {'vol raw':>9} "
          f"{'hedged':>8} {'w(xsmom) raw':>13} {'hedged':>8}")
    for name in study["walk_forward_raw"]["schemes"]:
        a = study["walk_forward_raw"]["schemes"][name]
        b = study["walk_forward_hedged"]["schemes"][name]
        print(f"{name:>14} {a['sharpe']:>+11.2f} {b['sharpe']:>+8.2f} "
              f"{a['ann_vol']:>9.2%} {b['ann_vol']:>8.2%} "
              f"{a['mean_weight']['xsmom']:>13.2f} {b['mean_weight']['xsmom']:>8.2f}")


def lookback_label(study: dict) -> int:
    return int(study["walk_forward_raw"]["months_traded"])


# ---------- reporting ----------

# ---------- 12. lever every sleeve to the same risk ----------

VOL_TARGET = 0.10        # annualized vol each sleeve is scaled to before allocation
VOL_WINDOW_MONTHS = 36   # trailing window for the two sleeves that only exist monthly
VOL_WINDOW_DAYS = 63     # trailing window for the pairs book, which has daily P&L
LEVERAGE_CAP = 5.0       # the most any sleeve is allowed in the capped variant


def trailing_vol(returns: pd.Series, window: int,
                 periods_per_year: int = MONTHS_PER_YEAR) -> pd.Series:
    """Annualized rolling standard deviation, known one observation late.

    The value at date t is the std of the `window` returns ending at t-1,
    shifted for the same reason trailing_beta is: the vol a desk sizes month
    t with is whatever the history said at the close of t-1. NaN until the
    window is full.
    """
    vol = returns.rolling(window).std(ddof=1) * np.sqrt(periods_per_year)
    return vol.shift(1).rename("vol")


def statarb_trailing_vol(path: str = "sleeves/statarb/results/portfolio.json",
                         window_days: int = VOL_WINDOW_DAYS) -> pd.Series:
    """The pairs book's vol from its daily P&L, stamped on the month it sizes.

    The other two sleeves only exist monthly and have history back to 2007
    and 2014, so a 36-month window is available before the overlap starts.
    The pairs book starts in January 2021: a 36-month window on its monthly
    returns would close in 2024 and leave nothing to allocate. It does have
    1,250 daily returns, though, and 63 of those pin a vol down better than
    36 monthly ones do. The estimate at the last trading day of month t-1 is
    the one month t is sized with.
    """
    with open(path) as fh:
        equity = json.load(fh)["equity"]
    series = pd.Series({pd.Timestamp(d): float(v) for d, v in equity}).sort_index()
    daily = series.pct_change().dropna()
    vol = daily.rolling(window_days).std(ddof=1) * np.sqrt(252)
    monthly = vol.resample("ME").last()
    monthly.index = monthly.index.to_period("M").to_timestamp("M")
    return monthly.shift(1).rename("vol")


def vol_targeted(y: pd.Series, vol: pd.Series, target: float = VOL_TARGET,
                 max_leverage: float | None = None) -> pd.DataFrame:
    """y scaled month by month to a constant forecast volatility.

    scaled_t = y_t * target / vol_{t-1}. The ratio is the leverage the desk
    runs that month, kept alongside so the write-up can show where it was
    going into the months that mattered. Months before the first full window
    are dropped, as in beta_hedged. A leverage cap is optional and off by
    default, so the first pass shows what the rule asks for before anyone
    decides what a prime broker would allow.
    """
    frame = pd.concat([y.rename("raw"), vol.rename("vol")], axis=1).dropna()
    leverage = target / frame["vol"]
    if max_leverage is not None:
        leverage = leverage.clip(upper=max_leverage)
    frame["leverage"] = leverage
    frame["scaled"] = frame["raw"] * leverage
    return frame


def vol_target_study(lookback: int = LOOKBACK_MONTHS, target: float = VOL_TARGET,
                     window: int = VOL_WINDOW_MONTHS,
                     beta_window: int = BETA_WINDOW_MONTHS,
                     cap: float = LEVERAGE_CAP,
                     volcarry_vol: pd.Series | None = None) -> dict:
    """The three-sleeve table again, with every sleeve levered to one vol.

    The last unticked item in the limitations list. The sleeves run at 1.5%,
    1.7% and 17% vol, and section 8 showed what that does to the one scheme
    that does not read a covariance: equal weight was a momentum book with two
    small positions attached. Every real multi-strategy desk sizes sleeves to
    a risk budget before an allocator sees them, so this does that, with the
    same discipline as the hedge: a trailing estimate, shifted a month, and
    nothing in the scale factor has seen the return it multiplies.

    The momentum sleeve goes in already beta-hedged, because section 10 is
    where the panel was left. Both walk-forwards run on exactly the months
    all three scaled sleeves exist, so the raw column here is not quite the
    hedged column of section 10 - three months shorter at the start, which
    is the daily window on the pairs book filling.

    The uncapped rule asks for whatever leverage the vol says, which on a
    1.5%-vol pairs book is 6x on an average month and more after a quiet one.
    A prime broker has a view on that, so the walk-forward is also run with
    every sleeve's leverage capped at `cap`, and the two are reported side by
    side rather than picking the one that looks better.

    volcarry_vol : a vol forecast for the vol sleeve to size on instead of
        its trailing estimate, on the same month-end index and already
        shifted so that the value at t was known before t. Section 12 passes
        the implied one; the other two sleeves are sized as above either way,
        so the only thing that differs between the two runs is that forecast.
    """
    market = market_monthly()
    xsmom = beta_hedged(xsmom_monthly(), market, beta_window)["hedged"].rename("xsmom")
    statarb = statarb_monthly()
    statarb.index = statarb.index.to_period("M").to_timestamp("M")
    volcarry = volcarry_monthly()
    volcarry.index = volcarry.index.to_period("M").to_timestamp("M")

    frames = {
        "statarb": vol_targeted(statarb, statarb_trailing_vol(), target),
        "volcarry": vol_targeted(volcarry, trailing_vol(volcarry, window)
                                 if volcarry_vol is None else volcarry_vol, target),
        "xsmom": vol_targeted(xsmom, trailing_vol(xsmom, window), target),
    }
    scaled_panel = pd.DataFrame({n: f["scaled"] for n, f in frames.items()}).dropna()
    raw_panel = pd.DataFrame({n: f["raw"] for n, f in frames.items()}).loc[scaled_panel.index]
    leverage = pd.DataFrame({n: f["leverage"] for n, f in frames.items()}).loc[scaled_panel.index]
    capped_panel = (raw_panel * leverage.clip(upper=cap)).loc[scaled_panel.index]

    sleeves = {}
    for name, frame in frames.items():
        lev = leverage[name]
        worst_raw = raw_panel[name].idxmin()
        worst_scaled = scaled_panel[name].idxmin()
        sleeves[name] = {
            "overlap_raw": performance(raw_panel[name]),
            "overlap_scaled": performance(scaled_panel[name]),
            "own_history_raw": performance(frame["raw"]),
            "own_history_scaled": performance(frame["scaled"]),
            "own_history_months": int(len(frame)),
            "own_history_first": f"{frame.index[0]:%Y-%m}",
            "leverage": {"mean": float(lev.mean()), "min": float(lev.min()),
                         "max": float(lev.max()), "last": float(lev.iloc[-1])},
            "worst_raw_month": f"{worst_raw:%Y-%m}",
            "leverage_into_worst_raw_month": float(lev.loc[worst_raw]),
            "worst_scaled_month": f"{worst_scaled:%Y-%m}",
            "leverage_into_worst_scaled_month": float(lev.loc[worst_scaled]),
            # how far the realized vol of the scaled sleeve lands from the target
            "realized_over_target": float(performance(scaled_panel[name])["ann_vol"] / target),
            "share_of_months_capped": float((lev > cap).mean()),
            "overlap_capped": performance(capped_panel[name]),
        }

    return {
        "target": float(target),
        "leverage_cap": float(cap),
        "window_months": int(window),
        "window_days_statarb": int(VOL_WINDOW_DAYS),
        "n_months": int(len(scaled_panel)),
        "first": str(scaled_panel.index[0].date()),
        "last": str(scaled_panel.index[-1].date()),
        "sleeves": sleeves,
        "correlation_raw": {a: {b: float(raw_panel[a].corr(raw_panel[b]))
                                for b in raw_panel.columns} for a in raw_panel.columns},
        "correlation_scaled": {a: {b: float(scaled_panel[a].corr(scaled_panel[b]))
                                   for b in scaled_panel.columns} for a in scaled_panel.columns},
        "walk_forward_raw": walk_forward(raw_panel, lookback=lookback),
        "walk_forward_scaled": walk_forward(scaled_panel, lookback=lookback),
        "walk_forward_capped": walk_forward(capped_panel, lookback=lookback),
    }


def print_vol_target(study: dict) -> None:
    print(f"\n12. LEVER EVERY SLEEVE TO {study['target']:.0%} VOL BEFORE ALLOCATING")
    print(f"   Trailing {study['window_months']}-month vol for volcarry and xsmom, "
          f"trailing {study['window_days_statarb']}-day vol from daily P&L for statarb, "
          f"all shifted one month.")
    print(f"   {study['n_months']} months, {study['first']} to {study['last']}.\n")
    print(f"{'sleeve':>10} {'vol raw':>8} {'scaled':>8} {'Sharpe raw':>11} {'scaled':>8} "
          f"{'worst raw':>10} {'scaled':>8} {'lev mean':>9} {'range':>13}")
    for name, s in study["sleeves"].items():
        a, b, lev = s["overlap_raw"], s["overlap_scaled"], s["leverage"]
        print(f"{name:>10} {a['ann_vol']:>8.2%} {b['ann_vol']:>8.2%} "
              f"{a['sharpe']:>+11.2f} {b['sharpe']:>+8.2f} "
              f"{a['worst_month']:>+10.2%} {b['worst_month']:>+8.2%} "
              f"{lev['mean']:>9.1f}x {lev['min']:>5.1f}-{lev['max']:<5.1f}x")
    print()
    for name, s in study["sleeves"].items():
        print(f"   {name}: leverage going into its worst raw month ({s['worst_raw_month']}) "
              f"was {s['leverage_into_worst_raw_month']:.1f}x; into its worst scaled "
              f"month ({s['worst_scaled_month']}) {s['leverage_into_worst_scaled_month']:.1f}x.")
    print(f"\n   On each sleeve's own history, raw -> scaled Sharpe:")
    for name, s in study["sleeves"].items():
        print(f"     {name:>9} from {s['own_history_first']} ({s['own_history_months']} months): "
              f"{s['own_history_raw']['sharpe']:+.2f} -> {s['own_history_scaled']['sharpe']:+.2f}")
    cr, cs = study["correlation_raw"], study["correlation_scaled"]
    print(f"\n   Correlations raw -> scaled: statarb/xsmom {cr['statarb']['xsmom']:+.2f} -> "
          f"{cs['statarb']['xsmom']:+.2f}, statarb/volcarry {cr['statarb']['volcarry']:+.2f} -> "
          f"{cs['statarb']['volcarry']:+.2f}, volcarry/xsmom {cr['volcarry']['xsmom']:+.2f} -> "
          f"{cs['volcarry']['xsmom']:+.2f}.")

    raw, scaled = study["walk_forward_raw"], study["walk_forward_scaled"]
    capped = study["walk_forward_capped"]
    print(f"\n   Walk-forward, {raw['months_traded']} months traded, raw sleeves against "
          f"levered, and levered with a {study['leverage_cap']:.0f}x cap:")
    print(f"\n{'scheme':>14} {'Sharpe raw':>11} {'scaled':>8} {'capped':>8} {'vol raw':>9} "
          f"{'scaled':>8} {'w(statarb)':>11} {'w(volcarry)':>12} {'w(xsmom)':>9}")
    for name in raw["schemes"]:
        a, b, c = raw["schemes"][name], scaled["schemes"][name], capped["schemes"][name]
        w = b["mean_weight"]
        print(f"{name:>14} {a['sharpe']:>+11.2f} {b['sharpe']:>+8.2f} {c['sharpe']:>+8.2f} "
              f"{a['ann_vol']:>9.2%} {b['ann_vol']:>8.2%} "
              f"{w['statarb']:>11.2f} {w['volcarry']:>12.2f} {w['xsmom']:>9.2f}")
    capped_share = {n: s["share_of_months_capped"] for n, s in study["sleeves"].items()}
    print("   Months at the cap: " + ", ".join(f"{n} {v:.0%}" for n, v in capped_share.items()))


# ---------- 13. size the vol sleeve on the vol it sells ----------

IMPLIED_FIT_MIN_MONTHS = 36   # months of history before the implied rule forecasts anything


def volcarry_implied(**kwargs) -> pd.Series:
    """The implied vol the sleeve sold each month, stamped on the month it sizes.

    volcarry_detail keeps the implied vol of the call sold on the first
    session of each month. That number is known the morning the trade goes
    on, so stamping it on the month it belongs to is not a shift that needs
    arguing about: the vol sold in month t is the one thing about month t a
    desk does know on day one.
    """
    detail = volcarry_detail(**kwargs)
    implied = detail["implied"].copy()
    implied.index = implied.index.to_period("M").to_timestamp("M")
    return implied.rename("implied")


def implied_vol_forecast(returns: pd.Series, implied: pd.Series,
                         min_months: int = IMPLIED_FIT_MIN_MONTHS) -> pd.Series:
    """The sleeve's vol for month t, forecast from the implied vol it sells in t.

    The sleeve's monthly return is a short call's hedged P&L, and that P&L is
    vega times the gap between the vol sold and the vol that happened, so
    its scale moves with the level of vol. The forecast uses that:

        var_t = beta_{t-1} * implied_t^2

    with beta the ratio of the sleeve's summed squared returns to the summed
    squared implieds over every month before t - an expanding regression
    through the origin, so beta at t has not seen month t and the only thing
    about month t in the forecast is the implied, which was known on day
    one. Annualized the same way trailing_vol is, so the two forecasts are
    the same quantity and vol_targeted can take either. NaN until there are
    `min_months` of history to fit on, which on this sleeve means the
    forecast starts in 2010, where the 36-month trailing rule's does.

    The fit is expanding rather than rolling on purpose. A ratio of two sums
    of squares is dominated by the months with the most of them, and a
    36-month window that happens to contain 2008 or 2020 fits a different
    sleeve from one that does not. Everything since 2007 is a better estimate
    of how this P&L scales with vol than the last three years are, and the
    rule has no business forgetting the one crash it has seen.
    """
    frame = pd.concat([returns.rename("r"), implied.rename("iv")], axis=1).dropna()
    beta = ((frame["r"] ** 2).cumsum() / (frame["iv"] ** 2).cumsum()).shift(1)
    beta.iloc[:min_months] = np.nan
    forecast = np.sqrt(MONTHS_PER_YEAR * beta) * frame["iv"]
    return forecast.rename("vol")


def leverage_alignment(frame: pd.DataFrame) -> float:
    """Correlation between the leverage a rule chose and the size of the move
    it chose it for. A sizing rule exists to be small in the big months, so
    this should be negative, and the more negative the better the rule
    anticipated rather than remembered."""
    return float(frame["leverage"].corr(frame["raw"].abs()))


def implied_sizing_study(levered: dict, lookback: int = LOOKBACK_MONTHS,
                         target: float = VOL_TARGET, window: int = VOL_WINDOW_MONTHS,
                         cap: float = LEVERAGE_CAP,
                         min_months: int = IMPLIED_FIT_MIN_MONTHS) -> dict:
    """Section 11's finding was that a trailing estimate is the wrong sizing
    signal for a short-vol sleeve. This sizes it on implied instead, which is
    the premium being sold and is known before the month starts, and reruns
    the sleeve on its own and the allocators on top of it.

    `levered` is the section 11 study, so the trailing-rule column of every
    table here is that section's number and not a re-estimate of it.
    """
    volcarry = volcarry_monthly()
    volcarry.index = volcarry.index.to_period("M").to_timestamp("M")
    implied = volcarry_implied()

    rules = {
        "trailing": vol_targeted(volcarry, trailing_vol(volcarry, window), target),
        "implied": vol_targeted(volcarry, implied_vol_forecast(volcarry, implied, min_months), target),
    }
    # Same months for both, so the own-history columns are comparable: the
    # trailing rule's first forecast and the implied rule's land on the same
    # month on this sleeve, but a different min_months would move one.
    common = rules["trailing"].index.intersection(rules["implied"].index)
    rules = {name: frame.loc[common] for name, frame in rules.items()}
    overlap_start = pd.Timestamp(levered["first"])

    worst = volcarry.loc[common].nsmallest(3).index
    sleeve = {}
    for name, frame in rules.items():
        on_overlap = frame.loc[overlap_start:]
        capped = frame["raw"] * frame["leverage"].clip(upper=cap)
        sleeve[name] = {
            "own_history": performance(frame["scaled"]),
            "own_history_capped": performance(capped),
            # the one year that separates the two rules, left out, so the
            # write-up can say how much of the gap is that year
            "own_history_ex_2020": performance(frame["scaled"][frame.index.year != 2020]),
            "overlap": performance(on_overlap["scaled"]),
            "overlap_capped": performance(capped.loc[overlap_start:]),
            "leverage": {"mean": float(frame["leverage"].mean()),
                         "min": float(frame["leverage"].min()),
                         "max": float(frame["leverage"].max())},
            "realized_over_target": float(performance(frame["scaled"])["ann_vol"] / target),
            "alignment": leverage_alignment(frame),
            "leverage_into_worst_months": {f"{d:%Y-%m}": float(frame["leverage"].loc[d])
                                           for d in worst},
            "share_of_months_capped": float((frame["leverage"] > cap).mean()),
        }
    raw = performance(volcarry.loc[common])
    forecasts = pd.DataFrame({n: f["vol"] for n, f in rules.items()})

    with_implied = vol_target_study(lookback=lookback, target=target, window=window,
                                    cap=cap, volcarry_vol=rules["implied"]["vol"])
    return {
        "target": float(target),
        "leverage_cap": float(cap),
        "min_months": int(min_months),
        "n_months": int(len(common)),
        "first": f"{common[0]:%Y-%m}",
        "last": f"{common[-1]:%Y-%m}",
        "worst_months": [f"{d:%Y-%m}" for d in worst],
        "raw_own_history": raw,
        "raw_own_history_ex_2020": performance(volcarry.loc[common][common.year != 2020]),
        "raw_overlap": performance(volcarry.loc[overlap_start:]),
        "forecast_correlation": float(forecasts["trailing"].corr(forecasts["implied"])),
        "sleeve": sleeve,
        "walk_forward_trailing": levered["walk_forward_scaled"],
        "walk_forward_trailing_capped": levered["walk_forward_capped"],
        "walk_forward_implied": with_implied["walk_forward_scaled"],
        "walk_forward_implied_capped": with_implied["walk_forward_capped"],
        "volcarry_on_overlap": with_implied["sleeves"]["volcarry"],
    }


def print_implied_sizing(study: dict) -> None:
    print(f"\n13. SIZE THE VOL SLEEVE ON THE VOL IT SELLS")
    print(f"   Implied vol at the sale, times an expanding ratio fitted on the months "
          f"before, against the trailing {VOL_WINDOW_MONTHS}-month rule of section 12.")
    print(f"   {study['n_months']} months, {study['first']} to {study['last']}; "
          f"the two forecasts correlate {study['forecast_correlation']:+.2f}.\n")
    raw = study["raw_own_history"]
    print(f"{'rule':>10} {'Sharpe':>7} {'capped':>7} {'ret':>8} {'vol':>7} "
          f"{'lev mean':>9} {'range':>12} {'corr(lev,|r|)':>14} "
          + " ".join(f"{m:>8}" for m in study["worst_months"]))
    print(f"{'none':>10} {raw['sharpe']:>+7.2f} {'':>7} {raw['ann_return']:>8.2%} "
          f"{raw['ann_vol']:>7.2%}")
    for name, s in study["sleeve"].items():
        own, lev = s["own_history"], s["leverage"]
        print(f"{name:>10} {own['sharpe']:>+7.2f} {s['own_history_capped']['sharpe']:>+7.2f} "
              f"{own['ann_return']:>8.2%} {own['ann_vol']:>7.2%} "
              f"{lev['mean']:>8.1f}x {lev['min']:>5.1f}-{lev['max']:<5.1f}x "
              f"{s['alignment']:>+14.2f} "
              + " ".join(f"{v:>7.1f}x" for v in s["leverage_into_worst_months"].values()))
    print(f"   Without 2020: raw {study['raw_own_history_ex_2020']['sharpe']:+.2f}, "
          + ", ".join(f"{n} {s['own_history_ex_2020']['sharpe']:+.2f}"
                      for n, s in study["sleeve"].items()) + ".")
    ov = study["raw_overlap"]
    print(f"   On the overlap: raw {ov['sharpe']:+.2f}, "
          + ", ".join(f"{n} {s['overlap']['sharpe']:+.2f} (capped {s['overlap_capped']['sharpe']:+.2f})"
                      for n, s in study["sleeve"].items()) + ".")

    tr, im = study["walk_forward_trailing"], study["walk_forward_implied"]
    trc, imc = study["walk_forward_trailing_capped"], study["walk_forward_implied_capped"]
    print(f"\n   Walk-forward, {im['months_traded']} months traded, the vol sleeve sized on "
          f"trailing against implied, uncapped and capped at {study['leverage_cap']:.0f}x:")
    print(f"\n{'scheme':>14} {'trailing':>9} {'implied':>8} {'tr capped':>10} {'im capped':>10} "
          f"{'w(volcarry) tr':>15} {'im':>6}")
    for name in im["schemes"]:
        print(f"{name:>14} {tr['schemes'][name]['sharpe']:>+9.2f} {im['schemes'][name]['sharpe']:>+8.2f} "
              f"{trc['schemes'][name]['sharpe']:>+10.2f} {imc['schemes'][name]['sharpe']:>+10.2f} "
              f"{tr['schemes'][name]['mean_weight']['volcarry']:>15.2f} "
              f"{im['schemes'][name]['mean_weight']['volcarry']:>6.2f}")


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
    exposures = factor_exposure_study()
    print_factor_exposures(exposures)
    hedged = hedged_momentum_study(lookback=lookback)
    print_hedged_momentum(hedged)
    levered = vol_target_study(lookback=lookback)
    print_vol_target(levered)
    implied = implied_sizing_study(levered, lookback=lookback)
    print_implied_sizing(implied)

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
                   "third_sleeve": third,
                   "factor_exposures": exposures,
                   "hedged_momentum": hedged,
                   "vol_target": levered,
                   "implied_sizing": implied}, fh, indent=1,
                  default=float)

    # Also on its own, because section 7 is the one table someone is likely to
    # want without parsing the whole run.
    with open("results/spread_sensitivity.json", "w") as fh:
        json.dump(spreads, fh, indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
