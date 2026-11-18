"""Thirteen months nothing in multi_strategy.py has seen.

Usage:  python holdout.py              (everything, sections 19-24)
        python holdout.py --estimator  (section 23 only, written beside the rest)
        python holdout.py --age        (section 24 only, written beside the rest)
Verify with:  pytest tests/test_holdout.py

Every section of multi_strategy.py ends in November 2024, because that is
where the volatility sleeve's cached P&L ends: the December 2024 option
needed 21 trading days the download did not have. The other two sleeves run
through December 2025. So the allocator has been judged on 47 months while 13
more sat in two of the three files, and section 18 closed on the observation
that more months are the only thing that would let a test tell signals apart.

This file is deliberately NOT a section of multi_strategy.py. The main study
reads its vol sleeve from a cache that stops in 2024 and this file does not
touch that cache, so nothing in the main study can have seen 2025 - the
months here are a holdout in the only sense that matters, which is that the
code that chose the methods never had them. The sleeve is re-run here with
the same trade, the same spread, the same anchor, on a download that goes a
month past the end of 2025, and the three-sleeve walk-forward of sections 11
to 18 (vol-targeted, capped at 5x, 24- and 12-month windows) is run on the
extended panel. Every number is then split: the months the main study saw,
the thirteen it did not, and the two together.

Written before it ran:

1. Thirteen months will not tell any allocator from 1/N. The bootstrap's
   90% interval on every Sharpe gap will straddle zero, on 13 months as it
   did on 19 to 31.
2. The full-sample ranking will not change: 1/N stays ahead of the window's
   max-Sharpe book at both lookbacks, because the new months are a fifth of
   the sample and the gap was half a point.
3. 2025 holds the first stress month for the vol sleeve since March 2020 -
   April, when the VIX printed above 50 - and the optimizer, overweight the
   vol sleeve after a calm 2024, takes it worse than 1/N does. That is the
   2022 shape from section 18, and if it repeats it is the one thing here a
   third sample would have shown that two did not.
"""

from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from multi_strategy import (BETA_WINDOW_MONTHS, LEVERAGE_CAP, LOOKBACK_MONTHS, MONTHS_PER_YEAR,
                            VOL_TARGET, VOL_WINDOW_MONTHS, allocators, beta_hedged,
                            estimate, market_monthly, paired_sharpe_bootstrap,
                            ewma_age_vol, ewma_lambda, ewma_vol, performance, statarb_monthly,
                            statarb_trailing_vol, trailing_rms_vol, trailing_vol,
                            vol_targeted, volcarry_monthly, xsmom_monthly)

START = "2007-01-01"
HOLDOUT_END = "2026-01-31"        # one month past 2025 so December's option has its 21 days
HOLDOUT_FROM = "2024-12-01"       # the first month the main study never saw
CACHE = "results/volcarry_monthly_holdout.csv"
OUT = "results/holdout.json"
LOOKBACKS = (LOOKBACK_MONTHS, 12)


# ---------- the vol sleeve, re-run past where the main study stops ----------

def volcarry_extended(cache: str = CACHE, start: str = START, end: str = HOLDOUT_END) -> pd.Series:
    """The sleeve's monthly return, same trade, same costs, longer tape.

    Cached under its own name so that the main study's cache is never
    rewritten by this file: that file ending in 2024 is what makes the
    months here a holdout.
    """
    from sleeves.volcarry.hedging_study import (OPTION_SPREAD_VIX_ANCHOR,
                                                OPTION_SPREAD_VOL_PTS, TICKER,
                                                months, trade)
    import yfinance as yf

    if os.path.exists(cache):
        return pd.read_csv(cache, index_col=0, parse_dates=True)["ret"]
    data = yf.download([TICKER, "^VIX"], start=start, end=end, progress=False,
                       auto_adjust=True)["Close"].dropna()
    data = data.rename(columns={TICKER: "spy", "^VIX": "vix"})
    rows = [trade(window, option_spread_vol_pts=OPTION_SPREAD_VOL_PTS,
                  option_spread_vix_anchor=OPTION_SPREAD_VIX_ANCHOR)
            for window in months(data)]
    frame = pd.DataFrame(rows).set_index("date")
    series = (frame["pnl"] / 100.0).rename("ret")
    series.index = series.index.to_period("M").to_timestamp("M")
    os.makedirs(os.path.dirname(cache), exist_ok=True)
    series.to_frame().to_csv(cache)
    return series


def agreement(extended: pd.Series, shipped: pd.Series) -> dict:
    """Does the longer download reproduce the months the main study used?

    Yahoo re-adjusts the whole history after each dividend, so the two will
    not match to the digit; the question is whether any month changes by
    enough to matter, and whether any changes sign.
    """
    both = pd.concat([extended.rename("new"), shipped.rename("old")], axis=1).dropna()
    diff = (both["new"] - both["old"]).abs()
    return {"months_compared": int(len(both)),
            "max_abs_diff": float(diff.max()),
            "max_abs_diff_month": f"{diff.idxmax():%Y-%m}",
            "mean_abs_diff": float(diff.mean()),
            "sign_flips": int((np.sign(both["new"]) != np.sign(both["old"])).sum()),
            "correlation": float(both["new"].corr(both["old"]))}


# ---------- the panel of sections 11-18, on the longer tape ----------

# how the vol sleeve measures its own vol: the sample std that shipped,
# the root mean square that keeps the mean, and the EWMA that keeps it
# and forgets gradually
ESTIMATORS = {"std": trailing_vol, "rms": trailing_rms_vol, "ewma": ewma_vol,
              "ewma_age": ewma_age_vol}


def frames(volcarry: pd.Series, target: float = VOL_TARGET, window: int = VOL_WINDOW_MONTHS,
           beta_window: int = BETA_WINDOW_MONTHS, volcarry_window: int | None = None,
           estimator: str = "std", xsmom_estimator: str = "std") -> dict:
    """multi_strategy.sleeve_frames with the vol sleeve handed in, so the
    only thing different from section 11 is how long that sleeve runs.

    `volcarry_window` sizes the vol sleeve on its own window and leaves the
    momentum sleeve on `window`; None is the shipped setting, both on 36.
    `estimator` is how the vol sleeve measures that window - see ESTIMATORS;
    `xsmom_estimator` is the momentum sleeve's, on the sample std unless
    section 24 asks otherwise.
    """
    market = market_monthly()
    xsmom = beta_hedged(xsmom_monthly(), market, beta_window)["hedged"].rename("xsmom")
    statarb = statarb_monthly()
    statarb.index = statarb.index.to_period("M").to_timestamp("M")
    volcarry = volcarry.copy()
    volcarry.index = volcarry.index.to_period("M").to_timestamp("M")
    return {"statarb": vol_targeted(statarb, statarb_trailing_vol(), target),
            "volcarry": vol_targeted(volcarry, ESTIMATORS[estimator](volcarry, volcarry_window or window), target),
            "xsmom": vol_targeted(xsmom, ESTIMATORS[xsmom_estimator](xsmom, window), target)}


def capped_panel(frames: dict, cap: float = LEVERAGE_CAP) -> pd.DataFrame:
    """The capped panel of section 11: raw returns times leverage clipped at `cap`."""
    scaled = pd.DataFrame({n: f["scaled"] for n, f in frames.items()}).dropna()
    raw = pd.DataFrame({n: f["raw"] for n, f in frames.items()}).loc[scaled.index]
    leverage = pd.DataFrame({n: f["leverage"] for n, f in frames.items()}).loc[scaled.index]
    return raw * leverage.clip(upper=cap)


def walk_forward_paths(panel: pd.DataFrame, lookback: int) -> tuple:
    """multi_strategy.walk_forward, keeping the monthly path of every scheme
    and the weights it held, instead of only its summary, so the months can
    be split afterwards and a month's weights looked up."""
    schemes = allocators()
    rows, held, dates = [], [], []
    for i in range(lookback, len(panel)):
        mu, cov = estimate(panel.iloc[i - lookback:i])
        nxt = panel.iloc[i].to_numpy(dtype=float)
        weights = {name: np.asarray(fn(mu, cov), dtype=float) for name, fn in schemes.items()}
        rows.append({name: float(w @ nxt) for name, w in weights.items()})
        held.append({name: dict(zip(panel.columns, w.round(4))) for name, w in weights.items()})
        dates.append(panel.index[i])
    return pd.DataFrame(rows, index=dates), pd.Series(held, index=dates)


def split_scores(paths: pd.DataFrame, holdout_from: str = HOLDOUT_FROM) -> dict:
    """Every scheme against equal weight on three samples: seen, unseen, both."""
    samples = {"seen": paths.loc[:pd.Timestamp(holdout_from) - pd.Timedelta(days=1)],
               "holdout": paths.loc[holdout_from:],
               "all": paths}
    out = {}
    for label, part in samples.items():
        ew = part["equal weight"].to_numpy()
        out[label] = {"months": int(len(part)),
                      "first": f"{part.index[0]:%Y-%m}", "last": f"{part.index[-1]:%Y-%m}",
                      "schemes": {name: {**performance(part[name]),
                                         "vs_equal_weight": paired_sharpe_bootstrap(part[name].to_numpy(), ew)}
                                  for name in part.columns}}
    return out


def holdout_study(lookbacks=LOOKBACKS, holdout_from: str = HOLDOUT_FROM) -> dict:
    extended = volcarry_extended()
    shipped = volcarry_monthly()
    shipped.index = shipped.index.to_period("M").to_timestamp("M")
    panel = capped_panel(frames(extended))
    new_months = extended.loc[holdout_from:]
    seen_sleeve = extended.loc[:pd.Timestamp(holdout_from) - pd.Timedelta(days=1)]
    worst = new_months.idxmin()
    study = {
        "holdout_from": holdout_from,
        "sleeve_agreement": agreement(extended, shipped),
        "sleeve": {
            "shipped_last": f"{shipped.index[-1]:%Y-%m}",
            "extended_last": f"{extended.index[-1]:%Y-%m}",
            "new_months": int(len(new_months)),
            "holdout": performance(new_months),
            "seen": performance(seen_sleeve),
            "worst_new_month": f"{worst:%Y-%m}",
            "worst_new_month_return": float(new_months.min()),
            "months_worse_in_history": int((seen_sleeve < new_months.min()).sum()),
            "path": [[f"{d:%Y-%m}", float(v)] for d, v in new_months.items()],
        },
        "panel": {"first": f"{panel.index[0]:%Y-%m}", "last": f"{panel.index[-1]:%Y-%m}",
                  "months": int(len(panel)),
                  "holdout_sleeves": {c: performance(panel.loc[holdout_from:, c]) for c in panel.columns}},
        "lookbacks": {},
    }
    for lookback in lookbacks:
        paths, held = walk_forward_paths(panel, lookback)
        worst_row = paths.loc[holdout_from:].idxmin()
        study["lookbacks"][str(lookback)] = {
            **split_scores(paths, holdout_from),
            "worst_holdout_month": {name: f"{d:%Y-%m}" for name, d in worst_row.items()},
            "holdout_path": {name: [[f"{d:%Y-%m}", float(v)] for d, v in paths.loc[holdout_from:, name].items()]
                             for name in paths.columns},
            # what each book held going into the sleeve's worst new month, and
            # the month after it - the de-risking the risk-based rules do
            "weights_into_worst": held.loc[worst],
            "weights_after_worst": held.loc[worst:].iloc[1] if len(held.loc[worst:]) > 1 else None,
            "volcarry_weight_path": {name: [[f"{d:%Y-%m}", w[name]["volcarry"]]
                                            for d, w in held.loc[holdout_from:].items()]
                                     for name in paths.columns},
        }
    return study


# ---------- the window the cut was made on ----------

# The panel is 56 months, so 42 is the longest window that still leaves the
# 13 holdout months traded; the main study's 24 and 12 sit in the middle.
WINDOWS = (6, 12, 18, 24, 36, 42)
RISK_BASED = ("inverse vol", "risk parity", "HRP", "min variance")


def window_sweep(panel: pd.DataFrame, windows=WINDOWS, holdout_from: str = HOLDOUT_FROM,
                 sleeve: str = "volcarry") -> dict:
    """Section 19's cut-after-the-loss, as a function of the window it was
    estimated on.

    The four risk-based rules took the vol sleeve from 37% of the book to
    25% after April 2025 and lagged 1/N in every one of the eight up months
    that followed. That cut is one month entering a variance estimate, so
    its size is a window-length question: April is one month in six or one
    in forty-two. Here the walk-forward is re-run at every window from 6 to
    42 months, and for each scheme the holdout is read three ways - what it
    held going into the sleeve's worst month, how much it cut the month
    after, and how many of the months after that it finished behind equal
    weight.

    Written before it ran:

    1. The cut shrinks with the window, roughly as one over it, for all four
       risk-based rules: at 6 months April is a sixth of the variance and
       the weight should roughly halve; at 42 it is a forty-second and the
       weight after should be within a few points of the weight before.
    2. The holdout Sharpe gap to 1/N shrinks toward zero as the window
       lengthens, and stays negative at every window. Any window that
       includes April cuts something and 1/N cuts nothing, and the sleeve
       then won eight in a row, so no window holds on well enough to catch
       up - the gap closes, it does not cross.
    3. Max Sharpe's weight going into April is the one number here with no
       clean prediction, because it comes off the mean, not the variance:
       the 12-month window was 90% in the sleeve and the 24-month 75%, and
       the 6-month window - Oct 2024 to Mar 2025, the sleeve's best run -
       should be the heaviest of all. Its April loss goes with that weight.
    """
    worst = panel.loc[holdout_from:, sleeve].idxmin()
    after = panel.loc[holdout_from:].index[panel.loc[holdout_from:].index > worst]
    prior = panel.loc[:worst, sleeve].iloc[:-1]
    out = {"holdout_from": holdout_from, "sleeve": sleeve, "worst_month": f"{worst:%Y-%m}",
           "worst_month_return": float(panel.loc[worst, sleeve]),
           # the month's square against the mean square of every month before
           # it: how many ordinary months of variance the one month is worth,
           # which is what decides how much of it a window can dilute
           "worst_month_in_ordinary_months": float(panel.loc[worst, sleeve] ** 2 / (prior ** 2).mean()),
           "months_after_worst": int(len(after)),
           "sleeve_after_worst": {"up_months": int((panel.loc[after, sleeve] > 0).sum()),
                                  "return": float((1 + panel.loc[after, sleeve]).prod() - 1)},
           "windows": {}}
    for window in windows:
        paths, held = walk_forward_paths(panel, window)
        # the sleeve's vol estimate the month after against the month before,
        # on this window: the whole of the risk-based cut is in this ratio
        with_it = panel.loc[:worst, sleeve].iloc[-window:].std(ddof=1)
        without = prior.iloc[-window:].std(ddof=1)
        part = paths.loc[holdout_from:]
        ew = part["equal weight"]
        schemes = {}
        for name in part.columns:
            w_into = held.loc[worst][name][sleeve]
            w_after = held.loc[after[0]][name][sleeve] if len(after) else None
            tail = held.loc[after]
            behind = part.loc[after, name] < ew.loc[after]
            schemes[name] = {
                **performance(part[name]),
                "vs_equal_weight": paired_sharpe_bootstrap(part[name].to_numpy(), ew.to_numpy()),
                "sharpe_gap": float(performance(part[name])["sharpe"] - performance(ew)["sharpe"]),
                "worst_month_return": float(part.loc[worst, name]),
                "weight_into_worst": float(w_into),
                "weight_after_worst": None if w_after is None else float(w_after),
                "cut": None if w_after is None else float(w_into - w_after),
                "cut_share": None if w_after is None or w_into == 0 else float((w_into - w_after) / w_into),
                "lowest_weight_after": float(min(h[name][sleeve] for h in tail)) if len(tail) else None,
                "weight_at_end": float(held.iloc[-1][name][sleeve]),
                "months_behind_after_worst": int(behind.sum()),
                "return_after_worst": float((1 + part.loc[after, name]).prod() - 1),
                "shortfall_after_worst": float(((1 + part.loc[after, name]).prod()
                                                - (1 + ew.loc[after]).prod())),
            }
        out["windows"][str(window)] = {"months_traded": int(len(paths)),
                                       "first_traded": f"{paths.index[0]:%Y-%m}",
                                       "holdout_months": int(len(part)),
                                       "sleeve_vol_multiplier": float(with_it / without),
                                       "schemes": schemes}
    return out


def window_verdicts(sweep: dict, risk_based=RISK_BASED) -> dict:
    """The three predictions, scored on the sweep."""
    windows = sorted(int(w) for w in sweep["windows"])
    by = {int(w): v["schemes"] for w, v in sweep["windows"].items()}
    # a worst month that is the last month has no month after, and no cut
    cuts = {name: [np.nan if by[w][name]["cut"] is None else by[w][name]["cut"] for w in windows]
            for name in risk_based}
    gaps = {name: [by[w][name]["sharpe_gap"] for w in windows] for name in by[windows[0]]}
    # one over the window, scaled to the shortest window's cut, is the shape
    # prediction 1 names; the ratio of the longest cut to it says how close
    scale = {name: [c[0] * windows[0] / w for w in windows] for name, c in cuts.items()}
    return {
        "windows": windows,
        "cut_shrinks_with_window": {name: all(b < a for a, b in zip(c, c[1:])) for name, c in cuts.items()},
        "cut_longest_over_shortest": {name: c[-1] / c[0] if c[0] else None for name, c in cuts.items()},
        "one_over_window_would_give": {name: s[-1] / cuts[name][0] if cuts[name][0] else None
                                       for name, s in scale.items()},
        "gap_closes_with_window": {name: g[-1] > g[0] for name, g in gaps.items()},
        "gap_negative_at_every_window": {name: all(x < 0 for x in g) for name, g in gaps.items()},
        "best_window_for": {name: windows[int(np.argmax(g))] for name, g in gaps.items()},
        "max_sharpe_heaviest_at": windows[int(np.argmax([by[w]["max Sharpe"]["weight_into_worst"]
                                                          for w in windows]))],
        "months_behind": {name: [by[w][name]["months_behind_after_worst"] for w in windows]
                          for name in by[windows[0]]},
    }


def print_window_sweep(sweep: dict, verdicts: dict) -> None:
    windows = verdicts["windows"]
    by = {int(w): v for w, v in sweep["windows"].items()}
    n_after = sweep["months_after_worst"]
    print(f"\nTHE WINDOW THE CUT WAS MADE ON   worst holdout month for the {sweep['sleeve']} sleeve "
          f"{sweep['worst_month']} at {sweep['worst_month_return']:+.1%}, worth "
          f"{sweep['worst_month_in_ordinary_months']:.0f} ordinary months of variance; "
          f"the {n_after} months after it: "
          f"{sweep['sleeve_after_worst']['up_months']} up, {sweep['sleeve_after_worst']['return']:+.1%}")
    print("\n   the sleeve's vol estimate the month after, over the month before: " + ", ".join(
        f"{w}mo x{by[w]['sleeve_vol_multiplier']:.2f}" for w in windows))
    names = list(by[windows[0]]["schemes"])
    print(f"\n   weight on the sleeve going into {sweep['worst_month']} -> the month after (cut in points):")
    print(f"\n{'window':>7} " + " ".join(f"{n[:12]:>22}" for n in names))
    for w in windows:
        print(f"{w:>5}mo " + " ".join(
            f"{by[w]['schemes'][n]['weight_into_worst']:>6.0%} -> {by[w]['schemes'][n]['weight_after_worst']:>4.0%} "
            f"({by[w]['schemes'][n]['cut'] * 100:>+4.0f})" for n in names))
    print(f"\n   {sweep['worst_month']} return, then months behind equal weight of the {n_after} after it, "
          "and the holdout Sharpe gap to equal weight [bootstrap 90%]:")
    print(f"\n{'window':>7} " + " ".join(f"{n[:12]:>28}" for n in names))
    for w in windows:
        cells = []
        for n in names:
            s = by[w]["schemes"][n]
            b = s["vs_equal_weight"]
            cells.append(f"{s['worst_month_return']:>+6.1%} {s['months_behind_after_worst']:>2}/{n_after} "
                         f"{s['sharpe_gap']:>+5.2f} [{b['low']:>+5.2f},{b['high']:>+5.2f}]")
        print(f"{w:>5}mo " + " ".join(f"{c:>28}" for c in cells))
    print("\n   the predictions:")
    for name in RISK_BASED:
        print(f"   {name:>13}: cut shrinks with the window {str(verdicts['cut_shrinks_with_window'][name]):<5} "
              f"(42mo cut is {verdicts['cut_longest_over_shortest'][name]:.2f} of 6mo's; 1/window says "
              f"{verdicts['one_over_window_would_give'][name]:.2f}); gap closes {str(verdicts['gap_closes_with_window'][name]):<5} "
              f"and stays negative {str(verdicts['gap_negative_at_every_window'][name]):<5} best window {verdicts['best_window_for'][name]}mo")
    print(f"   {'max Sharpe':>13}: heaviest in the sleeve going in at {verdicts['max_sharpe_heaviest_at']}mo; "
          f"best window {verdicts['best_window_for']['max Sharpe']}mo; gap closes "
          f"{verdicts['gap_closes_with_window']['max Sharpe']}, negative throughout "
          f"{verdicts['gap_negative_at_every_window']['max Sharpe']}")


# ---------- the sleeve's own window ----------

# Before the allocator sees the vol sleeve, the sleeve sizes itself: section
# 12 scales every month by target / trailing vol, on a 36-month window,
# capped at 5x. So April 2025 enters two variance estimates, not one, and
# section 20 only swept the allocator's. The sleeve has history from 2007,
# so 60 months is available before the panel starts in 2021; 6 is the
# shortest section 20 ran.
SLEEVE_WINDOWS = (6, 12, 24, 36, 60)


def sleeve_window_sweep(volcarry: pd.Series, windows=SLEEVE_WINDOWS,
                        lookback: int = LOOKBACK_MONTHS, holdout_from: str = HOLDOUT_FROM,
                        sleeve: str = "volcarry", cap: float = LEVERAGE_CAP,
                        target: float = VOL_TARGET, estimator: str = "std",
                        xsmom_estimator: str = "std") -> dict:
    """Section 20's question asked of the other dial: the window the vol
    sleeve sizes itself on, with the allocator's held at `lookback`.

    The sleeve's leverage in month t is target over the trailing vol at
    t-1, so the month after April the sleeve cuts itself by one minus the
    inverse of the vol multiplier section 20 measured - before any
    allocator moves a weight. The book's exposure to the raw strategy is
    the allocator's weight times the sleeve's leverage, and that product
    is what this sweep reads, going into April and the month after, for
    every scheme and every sleeve window.

    Written before it ran:

    1. The sleeve's own cut is set by the same ratio as section 20's: with
       April worth k ordinary months of variance, the vol estimate goes
       x sqrt((w - 1 + k) / w) and the leverage the inverse of that, so
       the sleeve cuts itself by about 56% at 6 months, 43% at 12, 30% at
       24, 23% at 36 and 16% at 60 - and nothing the allocator does can
       put that back.
    2. Equal weight is not immune. Sections 19 and 20 said 1/N "cuts
       nothing"; it cuts no WEIGHT, and the sleeve under it cut itself by
       a quarter at the shipped window. Every book's exposure to the raw
       strategy fell after April by at least the sleeve's own cut, and the
       risk-based books' fell by that compounded with theirs - roughly
       half at 36/24, the shipped pair.
    3. The window sets what the sleeve took into April: a short window is
       the calm of late 2024 and levers up into it, a long one still
       remembers 2022 and does not. So the sleeve's April loss on the
       scaled series is largest at 6 months and smallest at 60, and its
       own holdout Sharpe rises with the window - the long window took
       less into April and gave less back after it.
    4. The allocator's cut, as a share of its weight, is deepest at the
       shortest sleeve window, because the April the allocator's 24-month
       variance sees is the scaled one, and the scaled April is largest
       where the leverage into it was.

    `cap` and `target` are the book's, applied to every sleeve: section 22
    moves them to see what the window does when the cap is not standing
    on it. `estimator` is the vol sleeve's own, section 23's dial;
    `xsmom_estimator` is the momentum sleeve's, section 24's.
    """
    out = {"holdout_from": holdout_from, "sleeve": sleeve, "lookback": lookback,
           "cap": cap, "target": target, "estimator": estimator,
           "xsmom_estimator": xsmom_estimator, "windows": {}}
    for window in windows:
        fr = frames(volcarry, target=target, volcarry_window=window, estimator=estimator,
                    xsmom_estimator=xsmom_estimator)
        panel = capped_panel(fr, cap)
        own = fr[sleeve].loc[panel.index]
        leverage = own["leverage"].clip(upper=cap)
        raw = own["raw"]
        worst = raw.loc[holdout_from:].idxmin()
        hold = panel.loc[holdout_from:].index
        after = hold[hold > worst]
        prior = raw.loc[:worst].iloc[:-1]
        # the raw sleeve's worst holdout month against an ordinary one, on
        # the months the sleeve itself has - the k in prediction 1
        k = float(raw.loc[worst] ** 2 / (prior.iloc[-window:] ** 2).mean())
        # the same month against the sample variance, which is what the
        # shipped estimator measures it against - section 22's correction
        k_demeaned = float((raw.loc[worst] - prior.iloc[-window:].mean()) ** 2
                           / prior.iloc[-window:].var(ddof=1))
        # the ratio counts the month that comes in and not the one that goes
        # out: a window is a queue, and the month it drops when April enters
        # is the month `window` months before - for a whole number of years,
        # an earlier April. The exact rms multiplier is 1 + (k - k_out) / w;
        # the EWMA drops nothing and weights April against its own estimate
        # (on the sleeve's whole history, which reaches past the panel's
        # start: k above is on the panel's months, as sections 21 and 22 read it)
        history = fr[sleeve]["raw"].loc[:worst].iloc[:-1]
        out_month = history.iloc[-window]
        var_in = own.loc[worst, "vol"] ** 2 / MONTHS_PER_YEAR
        k_own = float(raw.loc[worst] ** 2 / var_in)
        k_out = float(out_month ** 2 / var_in)
        # an EWMA's multiplier is its own recursion: sqrt(lam + (1 - lam) k)
        # on its own k, which is sqrt((w - 1 + k) / w) when the newest month
        # is weighted 1/w and sqrt((w - 1 + 2k) / (w + 1)) when the mean age
        # is matched instead
        exact = {"std": None, "rms": float(np.sqrt(1 + (k_own - k_out) / window)),
                 "ewma": float(np.sqrt((window - 1 + k_own) / window)),
                 "ewma_age": float(np.sqrt(ewma_lambda(window, "age")
                                           + (1 - ewma_lambda(window, "age")) * k_own))}[estimator]
        paths, held = walk_forward_paths(panel, lookback)
        part = paths.loc[holdout_from:]
        ew = part["equal weight"]
        lev_into = float(leverage.loc[worst])
        lev_after = float(leverage.loc[after[0]]) if len(after) else None
        scaled = panel[sleeve]
        # the momentum sleeve, so section 24 can see what sizing it on the
        # same estimator did to it (nothing, if its mean is nothing)
        xs = fr["xsmom"].loc[panel.index]
        xs_lev = xs["leverage"].clip(upper=cap)
        row = {"worst_month": f"{worst:%Y-%m}", "months_traded": int(len(paths)),
               "xsmom": {"leverage_into_worst": float(xs_lev.loc[worst]),
                         "mean_leverage_holdout": float(xs_lev.loc[holdout_from:].mean()),
                         "mean_leverage": float(xs_lev.mean()),
                         "mean_share_of_mean_square": float(
                             xs["raw"].mean() ** 2 / (xs["raw"] ** 2).mean()),
                         **{f"holdout_{k}": v for k, v in performance(panel.loc[holdout_from:, "xsmom"]).items()}},
               "worst_in_ordinary_months": k,
               "worst_in_ordinary_months_demeaned": k_demeaned,
               "worst_in_ordinary_months_own_estimate": k_own,
               "month_dropped": f"{history.index[-window]:%Y-%m}",
               "month_dropped_return": float(out_month),
               "month_dropped_in_ordinary_months": k_out,
               "predicted_multiplier_demeaned": float(np.sqrt((window - 1 + k_demeaned) / window)),
               "predicted_multiplier_exact": exact,
               "vol_multiplier": float(own.loc[after[0], "vol"] / own.loc[worst, "vol"]) if len(after) else None,
               "predicted_multiplier": float(np.sqrt((window - 1 + k) / window)),
               "leverage_into_worst": lev_into, "leverage_after_worst": lev_after,
               "uncapped_leverage_into_worst": float(own.loc[worst, "leverage"]),
               "uncapped_leverage_after_worst": float(own.loc[after[0], "leverage"]) if len(after) else None,
               "vol_into_worst": float(own.loc[worst, "vol"]),
               "vol_after_worst": float(own.loc[after[0], "vol"]) if len(after) else None,
               "cap_binds_into_worst": bool(own.loc[worst, "leverage"] > cap),
               "cap_binds_after_worst": bool(own.loc[after[0], "leverage"] > cap) if len(after) else None,
               # how much of the time the target asks for more than the cap
               # allows: the share of months the "vol targeting" is the cap
               "cap_binds_share": float((own["leverage"] > cap).mean()),
               "cap_binds_share_holdout": float((own.loc[holdout_from:, "leverage"] > cap).mean()),
               "own_cut": None if lev_after is None else float(1 - lev_after / lev_into),
               "predicted_own_cut": float(1 - 1 / np.sqrt((window - 1 + k) / window)),
               "predicted_own_cut_demeaned": float(1 - 1 / np.sqrt((window - 1 + k_demeaned) / window)),
               # the scaled month against the scaled series' mean square on the
               # allocator's own window: what the allocator's variance sees
               "scaled_worst_in_ordinary_months": float(
                   scaled.loc[worst] ** 2 / (scaled.loc[:worst].iloc[:-1].iloc[-lookback:] ** 2).mean()),
               "sleeve": {**performance(scaled.loc[holdout_from:]),
                          "worst_month_return": float(scaled.loc[worst]),
                          "raw_worst_month_return": float(raw.loc[worst]),
                          "return_after_worst": float((1 + scaled.loc[after]).prod() - 1),
                          "raw_return_after_worst": float((1 + raw.loc[after]).prod() - 1),
                          "mean_leverage_holdout": float(leverage.loc[holdout_from:].mean())},
               "schemes": {}}
        for name in part.columns:
            w_into = float(held.loc[worst][name][sleeve])
            w_after = float(held.loc[after[0]][name][sleeve]) if len(after) else None
            x_into = w_into * lev_into
            x_after = None if w_after is None else w_after * lev_after
            row["schemes"][name] = {
                **performance(part[name]),
                "xsmom_weight_into_worst": float(held.loc[worst][name]["xsmom"]),
                "vs_equal_weight": paired_sharpe_bootstrap(part[name].to_numpy(), ew.to_numpy()),
                "sharpe_gap": float(performance(part[name])["sharpe"] - performance(ew)["sharpe"]),
                "worst_month_return": float(part.loc[worst, name]),
                "weight_into_worst": w_into, "weight_after_worst": w_after,
                "weight_cut_share": None if w_after is None or w_into == 0 else float(1 - w_after / w_into),
                "exposure_into_worst": x_into, "exposure_after_worst": x_after,
                "exposure_cut_share": None if x_after is None or x_into == 0 else float(1 - x_after / x_into),
                "months_behind_after_worst": int((part.loc[after, name] < ew.loc[after]).sum()),
                "return_after_worst": float((1 + part.loc[after, name]).prod() - 1),
            }
        out["windows"][str(window)] = row
    return out


def sleeve_window_verdicts(sweep: dict, risk_based=RISK_BASED) -> dict:
    """The four predictions, scored on the sweep."""
    windows = sorted(int(w) for w in sweep["windows"])
    by = {int(w): v for w, v in sweep["windows"].items()}
    # a worst month that is the last month has no month after, and no cut
    own = [np.nan if by[w]["own_cut"] is None else by[w]["own_cut"] for w in windows]
    predicted = [by[w]["predicted_own_cut"] for w in windows]
    sleeve_april = [by[w]["sleeve"]["worst_month_return"] for w in windows]
    sleeve_sharpe = [by[w]["sleeve"]["sharpe"] for w in windows]
    names = list(by[windows[0]]["schemes"])
    exposure = {n: [np.nan if by[w]["schemes"][n]["exposure_cut_share"] is None
                    else by[w]["schemes"][n]["exposure_cut_share"] for w in windows] for n in names}
    weight = {n: [np.nan if by[w]["schemes"][n]["weight_cut_share"] is None
                  else by[w]["schemes"][n]["weight_cut_share"] for w in windows] for n in names}
    return {
        "windows": windows,
        # the thing none of the predictions allowed for
        "cap_binds_into_worst": [by[w]["cap_binds_into_worst"] for w in windows],
        "cap_binds_after_worst": [by[w]["cap_binds_after_worst"] for w in windows],
        "cap_binds_share": [by[w]["cap_binds_share"] for w in windows],
        "cap_binds_share_holdout": [by[w]["cap_binds_share_holdout"] for w in windows],
        # 1. the sleeve's own cut, against the ratio's prediction
        "own_cut": own, "predicted_own_cut": predicted,
        "own_cut_within_five_points": [abs(a - b) < 0.05 for a, b in zip(own, predicted)],
        "own_cut_shrinks_with_window": all(b < a for a, b in zip(own, own[1:])),
        # 2. every book's exposure fell by at least the sleeve's own cut, and
        #    the risk-based books' by more
        "equal_weight_exposure_cut_is_own_cut": [abs(exposure["equal weight"][i] - own[i]) < 1e-6
                                                 for i in range(len(windows))],
        "risk_based_exposure_cut_above_own": {n: all(e > o for e, o in zip(exposure[n], own))
                                              for n in risk_based},
        "exposure_cut": exposure,
        "exposure_cut_at_shipped": {n: by[36]["schemes"][n]["exposure_cut_share"] for n in names} if 36 in by else None,
        # 3. April largest at the shortest window, the sleeve's Sharpe rising with it
        "sleeve_april": sleeve_april,
        "sleeve_april_worst_at": windows[int(np.argmin(sleeve_april))],
        "sleeve_sharpe": sleeve_sharpe,
        "sleeve_sharpe_rises_with_window": all(b > a for a, b in zip(sleeve_sharpe, sleeve_sharpe[1:])),
        "sleeve_best_window": windows[int(np.argmax(sleeve_sharpe))],
        "leverage_into": [by[w]["leverage_into_worst"] for w in windows],
        # 4. the allocator's own cut deepest at the shortest sleeve window
        "weight_cut": weight,
        "weight_cut_deepest_at": {n: windows[int(np.argmax(weight[n]))] for n in risk_based},
        "sharpe_gap": {n: [by[w]["schemes"][n]["sharpe_gap"] for w in windows] for n in names},
        "months_behind": {n: [by[w]["schemes"][n]["months_behind_after_worst"] for w in windows] for n in names},
    }


def print_sleeve_window_sweep(sweep: dict, verdicts: dict) -> None:
    windows = verdicts["windows"]
    by = {int(w): v for w, v in sweep["windows"].items()}
    first = by[windows[0]]
    print(f"\nTHE SLEEVE'S OWN WINDOW   the {sweep['sleeve']} sleeve sized on 6 to 60 months, the "
          f"allocator on {sweep['lookback']}; worst holdout month {first['worst_month']} at "
          f"{first['sleeve']['raw_worst_month_return']:+.1%} raw")
    print(f"\n   the sleeve by itself: leverage into {first['worst_month']} -> the month after, the cut "
          f"it made on its own (and what the ratio predicts), its scaled {first['worst_month']}, "
          "and its holdout Sharpe:")
    print(f"\n{'window':>7} {'vol in':>7} {'asks':>6} {'lev in':>7} {'vol out':>8} {'asks':>6} {'lev out':>8} "
          f"{'own cut':>8} {'predicted':>10} {'cap binds':>10} {'April':>7} {'after':>7} {'Sharpe':>7}")
    for w in windows:
        r = by[w]
        print(f"{w:>5}mo {r['vol_into_worst']:>7.2%} {r['uncapped_leverage_into_worst']:>5.2f}x "
              f"{r['leverage_into_worst']:>6.2f}x {r['vol_after_worst']:>8.2%} "
              f"{r['uncapped_leverage_after_worst']:>5.2f}x {r['leverage_after_worst']:>7.2f}x "
              f"{r['own_cut']:>8.0%} {r['predicted_own_cut']:>10.0%} "
              f"{r['cap_binds_share']:>4.0%}/{r['cap_binds_share_holdout']:>4.0%} "
              f"{r['sleeve']['worst_month_return']:>+7.1%} {r['sleeve']['return_after_worst']:>+7.1%} "
              f"{r['sleeve']['sharpe']:>+7.2f}")
    print(f"\n   (asks = target / vol, before the {LEVERAGE_CAP:.0f}x cap; cap binds = share of panel "
          "months / holdout months the sleeve asks for more than the cap)")
    names = list(first["schemes"])
    print("\n   each book's exposure to the raw strategy (weight x leverage) going in -> after, "
          "and the cut as a share:")
    print(f"\n{'window':>7} " + " ".join(f"{n[:12]:>22}" for n in names))
    for w in windows:
        print(f"{w:>5}mo " + " ".join(
            f"{by[w]['schemes'][n]['exposure_into_worst']:>5.2f} -> {by[w]['schemes'][n]['exposure_after_worst']:>4.2f} "
            f"({by[w]['schemes'][n]['exposure_cut_share']:>4.0%})" for n in names))
    print("\n   the allocator's own cut as a share of its weight, months behind 1/N of the "
          f"{len(by[windows[0]]['schemes'][names[0]]) and 8} after, and the holdout Sharpe gap [bootstrap 90%]:")
    print(f"\n{'window':>7} " + " ".join(f"{n[:12]:>26}" for n in names))
    for w in windows:
        cells = []
        for n in names:
            s = by[w]["schemes"][n]
            b = s["vs_equal_weight"]
            cut = "   -" if s["weight_cut_share"] is None else f"{s['weight_cut_share']:>4.0%}"
            cells.append(f"{cut} {s['months_behind_after_worst']:>1}/8 {s['sharpe_gap']:>+5.2f} "
                         f"[{b['low']:>+5.2f},{b['high']:>+5.2f}]")
        print(f"{w:>5}mo " + " ".join(f"{c:>26}" for c in cells))
    v = verdicts
    print("\n   the predictions:")
    print(f"   0. the cap binds going into {first['worst_month']} at "
          f"{sum(v['cap_binds_into_worst'])} of {len(windows)} windows, and the month after at "
          f"{sum(v['cap_binds_after_worst'])}")
    print("   1. own cut within five points of the ratio's: "
          f"{sum(v['own_cut_within_five_points'])} of {len(windows)} windows; shrinks with the window "
          f"{v['own_cut_shrinks_with_window']}")
    print("   2. equal weight's exposure cut is the sleeve's own cut at every window: "
          f"{all(v['equal_weight_exposure_cut_is_own_cut'])}; risk-based books cut more than that at every "
          "window: " + ", ".join(f"{n} {ok}" for n, ok in v["risk_based_exposure_cut_above_own"].items()))
    print(f"   3. the sleeve's April is worst at {v['sleeve_april_worst_at']}mo; its Sharpe rises with the "
          f"window {v['sleeve_sharpe_rises_with_window']}, best at {v['sleeve_best_window']}mo")
    print("   4. the allocator's cut is deepest at: " + ", ".join(
        f"{n} {w}mo" for n, w in v["weight_cut_deepest_at"].items()))


# ---------- section 22: unpin the dial ----------

UNPINNED = (("10% / 5x", 0.10, 5.0), ("10% / 10x", 0.10, 10.0), ("10% / uncapped", 0.10, np.inf),
            ("5% / 5x", 0.05, 5.0), ("5% / 10x", 0.05, 10.0))


def unpinned_sweep(volcarry: pd.Series, settings=UNPINNED, windows=SLEEVE_WINDOWS,
                   lookback: int = LOOKBACK_MONTHS, holdout_from: str = HOLDOUT_FROM) -> dict:
    """Section 21 again, at settings where the cap is not the leverage.

    Section 21 swept the vol sleeve's own window and found it set nothing,
    because a 10% target on a sleeve that makes under 2% of vol asks for
    five to eight times and the 5x cap gave five whatever the window. This
    runs the same sweep with the dial unpinned two ways: a higher cap at
    the same target, and the shipped cap at half the target. Both are the
    book's settings and move every sleeve, so the comparison stays a book
    against a book.

    Written before it ran:

    1. Unpinned, the window does what section 20's ratio says. With the
       cap off the leverage going into April, the sleeve's own cut after it
       is one minus the inverse of the vol multiplier, within five points,
       at every window - the 56/43/30/23/16% section 21 predicted and could
       not see. At 10x the cap should still bind at the two shortest
       windows going in (the calm of late 2024 asks for more than ten), so
       the clean reading is the 5% target.
    2. The hump goes. Section 21 found the sleeve's holdout Sharpe best at
       36 months because 36 was where the cap never let go, so a fixed 5x
       took the whole recovery. With the cap off, the sleeve's Sharpe
       should rise with the window, as section 21's prediction 3 said:
       the long window took less into April and gives less back after.
    3. The April is worse at 10x. The sleeve goes in at 7.9x instead of 5x
       at the shipped window, so the scaled April is about -17% against
       -11%, and the allocator's cut is deeper than section 21's at every
       window - the April the 24-month variance sees is the scaled one.
    4. Nothing here rescues the risk-based books. The Sharpe gap to 1/N
       stays negative for all four at the shipped lookback in every
       setting; unpinning the dial changes how the sleeve sizes itself,
       not whether the allocator should have cut it.
    """
    out = {"holdout_from": holdout_from, "lookback": lookback, "settings": {}}
    for label, target, cap in settings:
        out["settings"][label] = sleeve_window_sweep(volcarry, windows, lookback, holdout_from,
                                                     cap=cap, target=target)
    return out


def unpinned_verdicts(sweep: dict, risk_based=RISK_BASED) -> dict:
    """The four predictions, scored across the settings."""
    labels = list(sweep["settings"])
    per = {label: sleeve_window_verdicts(sweep["settings"][label], risk_based) for label in labels}
    shipped = per[labels[0]]
    windows = shipped["windows"]

    def cap_free(label):
        v = per[label]
        return [not (a or b) for a, b in zip(v["cap_binds_into_worst"], v["cap_binds_after_worst"])]

    def within(label):
        v = per[label]
        return [ok for ok, free in zip(v["own_cut_within_five_points"], cap_free(label)) if free]

    # 3. the allocator's cut against the shipped setting's, by scheme and window
    deeper = {}
    for label in labels[1:]:
        deeper[label] = {n: [a > b for a, b in zip(per[label]["weight_cut"][n], shipped["weight_cut"][n])]
                         for n in risk_based}
    return {
        "settings": labels, "windows": windows,
        "cap_binds_into_worst": {label: per[label]["cap_binds_into_worst"] for label in labels},
        "cap_binds_share": {label: per[label]["cap_binds_share"] for label in labels},
        "cap_free_windows": {label: cap_free(label) for label in labels},
        # 1. where the cap is off both going in and coming out, the ratio predicts the cut
        "own_cut": {label: per[label]["own_cut"] for label in labels},
        "predicted_own_cut": shipped["predicted_own_cut"],
        "ratio_within_five_points_where_cap_free": {label: within(label) for label in labels},
        "ratio_holds_where_cap_free": {label: bool(within(label)) and all(within(label)) for label in labels},
        # 2. the hump
        "sleeve_sharpe": {label: per[label]["sleeve_sharpe"] for label in labels},
        "sleeve_best_window": {label: per[label]["sleeve_best_window"] for label in labels},
        "sleeve_sharpe_rises_with_window": {label: per[label]["sleeve_sharpe_rises_with_window"]
                                            for label in labels},
        # 3. the April and the allocator's cut
        "sleeve_april": {label: per[label]["sleeve_april"] for label in labels},
        "leverage_into": {label: per[label]["leverage_into"] for label in labels},
        "weight_cut": {label: per[label]["weight_cut"] for label in labels},
        "allocator_cuts_deeper_than_shipped": {label: {n: all(d) for n, d in deeper[label].items()}
                                               for label in labels[1:]},
        # 4. the gap to 1/N
        "sharpe_gap": {label: per[label]["sharpe_gap"] for label in labels},
        "risk_based_behind_equal_weight_everywhere": {
            label: all(g < 0 for n in risk_based for g in per[label]["sharpe_gap"][n]) for label in labels},
    }


def print_unpinned(sweep: dict, verdicts: dict) -> None:
    v = verdicts
    windows = v["windows"]
    print("\nUNPIN THE DIAL   section 21's sweep at settings where the cap is not the leverage")
    print("\n   the sleeve by itself at each setting: leverage into the worst month -> after, the cut it "
          "made on its own\n   (the ratio predicts " + ", ".join(f"{p:.0%}" for p in v["predicted_own_cut"])
          + f" at {', '.join(f'{w}mo' for w in windows)}), its scaled April, and its holdout Sharpe:")
    for label, s in sweep["settings"].items():
        by = {int(w): r for w, r in s["windows"].items()}
        binds = ", ".join(f"{by[w]['cap_binds_share']:.0%}" for w in windows)
        print(f"\n   {label}   (cap binds in {binds} of panel months)")
        print(f"{'window':>7} {'lev in':>7} {'lev out':>8} {'own cut':>8} {'April':>7} {'after':>7} {'Sharpe':>7}")
        for w in windows:
            r = by[w]
            cut = "   -" if r["own_cut"] is None else f"{r['own_cut']:>4.0%}"
            print(f"{w:>5}mo {r['leverage_into_worst']:>6.2f}x {r['leverage_after_worst']:>7.2f}x "
                  f"{cut:>8} {r['sleeve']['worst_month_return']:>+7.1%} "
                  f"{r['sleeve']['return_after_worst']:>+7.1%} {r['sleeve']['sharpe']:>+7.2f}")
    print("\n   the allocator's own cut as a share of its weight, HRP and min variance, and the Sharpe gap to 1/N:")
    print(f"\n{'setting':>15} " + " ".join(f"{w:>5}mo" for w in windows) + "   " + " ".join(f"{w:>5}mo" for w in windows))
    for label in v["settings"]:
        for n in ("HRP", "min variance"):
            cuts = " ".join("    -" if c != c else f"{c:>6.0%}" for c in v["weight_cut"][label][n])
            gaps = " ".join(f"{g:>+6.2f}" for g in v["sharpe_gap"][label][n])
            print(f"{label:>15} {cuts}   {gaps}   {n}")
    print("\n   the predictions:")
    print("   1. where the cap is off going in and coming out, the own cut is within five points of the ratio: "
          + ", ".join(f"{label} {sum(ok)}/{len(ok)}" for label, ok in v["ratio_within_five_points_where_cap_free"].items()))
    print("   2. the sleeve's Sharpe rises with the window: " + ", ".join(
        f"{label} {'yes' if ok else 'no'} (best {v['sleeve_best_window'][label]}mo)"
        for label, ok in v["sleeve_sharpe_rises_with_window"].items()))
    print("   3. the allocator cuts deeper than at the shipped setting, every window: " + ", ".join(
        f"{label} " + "/".join("yes" if ok else "no" for ok in d.values())
        for label, d in v["allocator_cuts_deeper_than_shipped"].items()))
    print("   4. every risk-based book behind 1/N at every window: " + ", ".join(
        f"{label} {'yes' if ok else 'no'}" for label, ok in v["risk_based_behind_equal_weight_everywhere"].items()))



# ---------- section 23: keep the mean ----------

# The two settings from section 22 where the cap is off the leverage going
# into April: the shipped target at twice the cap, and half the target at
# twice the cap (which is 10% / uncapped everywhere but 6 months).
KEPT_MEAN = (("10% / 10x", 0.10, 10.0), ("5% / 10x", 0.05, 10.0))
ESTIMATOR_ORDER = ("std", "rms", "ewma")


def estimator_sweep(volcarry: pd.Series, settings=KEPT_MEAN, estimators=ESTIMATOR_ORDER,
                    windows=SLEEVE_WINDOWS, lookback: int = LOOKBACK_MONTHS,
                    holdout_from: str = HOLDOUT_FROM) -> dict:
    """Section 22 again, with the sleeve sizing on an estimator that keeps
    the mean.

    Section 22 found the ratio that predicts the sleeve's own cut after
    April was written on the mean square while the sleeve sizes on a sample
    standard deviation, which subtracts the mean first, and that a Sharpe-2
    sleeve keeps a third of its variance there. Put the demeaned k in the
    formula and it lands on the measured multiplier at every window but
    six months. This runs the same sweep three ways: the sample std that
    shipped, a root mean square over the same window, and a RiskMetrics
    EWMA with the newest month weighted 1/window so April enters each
    estimate with the same weight it has in the window.

    Written before it ran:

    1. On the rms estimator the original ratio is right. Its k is the
       mean-square k the ratio was written on, so the sleeve's own cut is
       within three points of section 20's 46/36/28/24/12% at every window,
       six months included - the six-month overshoot in section 22 was the
       window's mean moving when April entered it, and this estimator has
       no mean to move.
    2. The rms sleeve is a smaller sleeve. Keeping the mean adds a third to
       the variance, so the vol is 15% higher and the leverage 13% lower at
       every window going into April: 6.9x at the shipped 36 months against
       7.9x, a scaled April near -15% against -17%, the cap binding less
       often, and the holdout Sharpe within 0.10 of the std row because the
       two estimates differ by a factor that moves slowly.
    3. The EWMA cuts what the window cuts. With the newest month at 1/w in
       both, April moves the two estimates by the same amount, so the EWMA's
       own cut is within two points of the rms window's at every w.
    4. And then it holds on. The window drops April after w months; the
       EWMA never does, it only decays it, and at the 6- and 12-month
       equivalents the window has forgotten April before the holdout ends
       while the EWMA still carries a quarter of it. So the EWMA re-levers
       less into the recovery and its holdout Sharpe is below the rms
       window's at 6 and 12, and within 0.05 of it at 24, 36 and 60, where
       neither has dropped the month yet.
    5. The allocator barely notices. Its HRP cut as a share of its weight is
       within five points of the std row at every window on both
       estimators: the series it is handed is the same raw sleeve at a
       slightly different leverage, and section 22's shallow cut at 6 and
       12 months was about the leverage swinging with each half-year's
       calm, which keeping the mean does not change.
    """
    out = {"holdout_from": holdout_from, "lookback": lookback, "settings": {}}
    for label, target, cap in settings:
        out["settings"][label] = {
            estimator: sleeve_window_sweep(volcarry, windows, lookback, holdout_from,
                                           cap=cap, target=target, estimator=estimator)
            for estimator in estimators}
    return out


def estimator_verdicts(sweep: dict, risk_based=RISK_BASED) -> dict:
    """The five predictions, scored at each setting."""
    out = {"settings": list(sweep["settings"]), "by_setting": {}}
    for label, by_est in sweep["settings"].items():
        per = {est: sleeve_window_verdicts(s, risk_based) for est, s in by_est.items()}
        windows = per["std"]["windows"]
        rows = {est: {int(w): r for w, r in by_est[est]["windows"].items()} for est in by_est}
        own = {est: per[est]["own_cut"] for est in per}
        # the ratio on the mean square, which is what the rms estimator uses,
        # and section 22's demeaned correction, which is what the std does
        mean_square = [rows["std"][w]["predicted_own_cut"] for w in windows]
        demeaned = [rows["std"][w]["predicted_own_cut_demeaned"] for w in windows]
        lev_in = {est: [rows[est][w]["leverage_into_worst"] for w in windows] for est in per}
        sharpe = {est: per[est]["sleeve_sharpe"] for est in per}
        hrp = {est: per[est]["weight_cut"]["HRP"] for est in per}
        v = {
            "windows": windows,
            "own_cut": own, "ratio_mean_square": mean_square, "ratio_demeaned": demeaned,
            # 1. the original ratio on the rms estimator
            "rms_within_three_points_of_ratio": [abs(a - b) < 0.03 for a, b in zip(own["rms"], mean_square)],
            "std_within_three_points_of_demeaned": [abs(a - b) < 0.03 for a, b in zip(own["std"], demeaned)],
            "vol_multiplier": {est: [rows[est][w]["vol_multiplier"] for w in windows] for est in per},
            "predicted_multiplier": [rows["std"][w]["predicted_multiplier"] for w in windows],
            "predicted_multiplier_demeaned": [rows["std"][w]["predicted_multiplier_demeaned"] for w in windows],
            "predicted_multiplier_exact": {est: [rows[est][w]["predicted_multiplier_exact"] for w in windows]
                                           for est in per if est != "std"},
            # the month going out, against the window's mean square (the rms row)
            "month_dropped": [rows["rms"][w]["month_dropped"] for w in windows],
            "month_dropped_return": [rows["rms"][w]["month_dropped_return"] for w in windows],
            "month_dropped_in_ordinary_months": [rows["rms"][w]["month_dropped_in_ordinary_months"] for w in windows],
            "k_on_own_estimate": {est: [rows[est][w]["worst_in_ordinary_months_own_estimate"] for w in windows]
                                  for est in per},
            # 2. the rms sleeve is smaller
            "leverage_into": lev_in,
            "rms_leverage_over_std": [a / b for a, b in zip(lev_in["rms"], lev_in["std"])],
            "rms_leverage_lower_everywhere": all(a < b for a, b in zip(lev_in["rms"], lev_in["std"])),
            "cap_binds_share": {est: per[est]["cap_binds_share"] for est in per},
            "sleeve_april": {est: per[est]["sleeve_april"] for est in per},
            "sleeve_sharpe": sharpe,
            "rms_sharpe_within_tenth_of_std": [abs(a - b) < 0.10 for a, b in zip(sharpe["rms"], sharpe["std"])],
            # 3. the ewma cuts what the rms window cuts
            "ewma_within_two_points_of_rms": [abs(a - b) < 0.02 for a, b in zip(own["ewma"], own["rms"])],
            # 4. and holds on: below the window at 6 and 12, level at 24+
            "ewma_sharpe_minus_rms": [a - b for a, b in zip(sharpe["ewma"], sharpe["rms"])],
            "ewma_below_rms_at_short_windows": all(sharpe["ewma"][i] < sharpe["rms"][i]
                                                   for i, w in enumerate(windows) if w <= 12),
            "ewma_level_with_rms_at_long_windows": all(abs(sharpe["ewma"][i] - sharpe["rms"][i]) < 0.05
                                                       for i, w in enumerate(windows) if w >= 24),
            "leverage_at_end": {est: [rows[est][w]["sleeve"]["mean_leverage_holdout"] for w in windows]
                                for est in per},
            # 5. the allocator's HRP cut within five points of the std row
            "hrp_cut": hrp,
            "scaled_april_in_allocator_months": {est: [rows[est][w]["scaled_worst_in_ordinary_months"] for w in windows]
                                                 for est in per},
            "hrp_weight_into": {est: [rows[est][w]["schemes"]["HRP"]["weight_into_worst"] for w in windows]
                                for est in per},
            "hrp_within_five_points_of_std": {est: [abs(a - b) < 0.05 for a, b in zip(hrp[est], hrp["std"])]
                                              for est in per if est != "std"},
            "sharpe_gap": {est: per[est]["sharpe_gap"] for est in per},
            "risk_based_behind_equal_weight_everywhere": {
                est: all(g < 0 for n in risk_based for g in per[est]["sharpe_gap"][n]) for est in per},
        }
        out["by_setting"][label] = v
    return out


def print_estimator(sweep: dict, verdicts: dict) -> None:
    print("\nKEEP THE MEAN   section 22's sweep with the sleeve sizing on the sample std, "
          "a root mean square, and an EWMA weighted 1/window on the newest month")
    for label, v in verdicts["by_setting"].items():
        windows = v["windows"]
        print(f"\n   {label}")
        print(f"{'window':>7} {'ratio':>6} {'demeaned':>9} " + " ".join(
            f"{est + ' cut':>9} {est + ' x':>7}" for est in ESTIMATOR_ORDER)
              + "   " + " ".join(f"{est + ' lev in':>12}" for est in ESTIMATOR_ORDER))
        for i, w in enumerate(windows):
            cuts = " ".join(f"{v['own_cut'][est][i]:>8.0%} {v['vol_multiplier'][est][i]:>7.2f}"
                            for est in ESTIMATOR_ORDER)
            levs = " ".join(f"{v['leverage_into'][est][i]:>11.2f}x" for est in ESTIMATOR_ORDER)
            print(f"{w:>5}mo {v['ratio_mean_square'][i]:>6.0%} {v['ratio_demeaned'][i]:>9.0%} {cuts}   {levs}")
        print(f"\n{'window':>7} " + " ".join(f"{est + ' April':>11} {est + ' Sharpe':>12}" for est in ESTIMATOR_ORDER)
              + "   " + " ".join(f"{est + ' HRP cut':>12}" for est in ESTIMATOR_ORDER))
        for i, w in enumerate(windows):
            cells = " ".join(f"{v['sleeve_april'][est][i]:>+11.1%} {v['sleeve_sharpe'][est][i]:>+12.2f}"
                             for est in ESTIMATOR_ORDER)
            hrp = " ".join(f"{v['hrp_cut'][est][i]:>12.0%}" for est in ESTIMATOR_ORDER)
            print(f"{w:>5}mo {cells}   {hrp}")
        print("   the month each window drops when April enters, in ordinary months of its mean square: " + ", ".join(
            f"{w}mo {m} ({k:.1f})" for w, m, k in zip(windows, v["month_dropped"], v["month_dropped_in_ordinary_months"])))
        print("   exact multiplier, rms 1 + (k - k_out)/w and ewma on its own k: rms "
              + "/".join(f"{m:.2f}" for m in v["predicted_multiplier_exact"]["rms"]) + ", ewma "
              + "/".join(f"{m:.2f}" for m in v["predicted_multiplier_exact"]["ewma"])
              + "; April on the ewma's own estimate " + "/".join(f"{k:.0f}" for k in v["k_on_own_estimate"]["ewma"]) + " months")
        print("   scaled April in ordinary months of the allocator's 24-month mean square: " + "; ".join(
            f"{est} " + "/".join(f"{k:.0f}" for k in v["scaled_april_in_allocator_months"][est]) for est in ESTIMATOR_ORDER))
        print("   cap binds in " + "; ".join(
            f"{est} " + "/".join(f"{s:.0%}" for s in v["cap_binds_share"][est]) for est in ESTIMATOR_ORDER)
              + " of panel months")
        print("\n   the predictions:")
        print(f"   1. rms cut within three points of the mean-square ratio: "
              f"{sum(v['rms_within_three_points_of_ratio'])} of {len(windows)} windows "
              f"(std within three of the demeaned one: {sum(v['std_within_three_points_of_demeaned'])})")
        print(f"   2. rms leverage lower at every window: {v['rms_leverage_lower_everywhere']}, "
              "ratio " + "/".join(f"{r:.2f}" for r in v["rms_leverage_over_std"])
              + f"; Sharpe within 0.10 of std at {sum(v['rms_sharpe_within_tenth_of_std'])} of {len(windows)}")
        print(f"   3. ewma cut within two points of rms: {sum(v['ewma_within_two_points_of_rms'])} of {len(windows)}")
        print(f"   4. ewma Sharpe below rms at 6 and 12: {v['ewma_below_rms_at_short_windows']}; "
              f"within 0.05 at 24+: {v['ewma_level_with_rms_at_long_windows']} "
              "(ewma - rms: " + "/".join(f"{d:+.2f}" for d in v["ewma_sharpe_minus_rms"]) + ")")
        print("   5. HRP cut within five points of std: " + ", ".join(
            f"{est} {sum(ok)}/{len(ok)}" for est, ok in v["hrp_within_five_points_of_std"].items())
              + "; risk-based behind 1/N everywhere: " + ", ".join(
            f"{est} {'yes' if ok else 'no'}" for est, ok in v["risk_based_behind_equal_weight_everywhere"].items()))



# ---------- section 24: match the age, and size the momentum sleeve the same way ----------

AGE_ORDER = ("rms", "ewma", "ewma_age")


def age_sweep(volcarry: pd.Series, settings=KEPT_MEAN, windows=SLEEVE_WINDOWS,
              lookback: int = LOOKBACK_MONTHS, holdout_from: str = HOLDOUT_FROM) -> dict:
    """Section 23 closed by saying an EWMA whose lambda is set by the mean
    age instead of the newest month's weight - lambda = 1 - 2/(w + 1), a
    half-life of twelve and a half months at the shipped 36 - "would
    behave like the rms window, not like this". This runs that row beside
    the two kept-mean rows of section 23 (which are re-run here and
    checked against the saved ones), and in a second pass sizes the
    momentum sleeve on the same estimator as the vol sleeve instead of
    leaving it on the sample std, which every section since 11 has done.

    Written before it ran:

    1. The mean-age EWMA is not the window with soft edges either. An
       exponential can match the window's mean age or its weight on the
       newest month, not both, and matching the age puts 2/(w + 1) on the
       month coming in - twice the window's 1/w. So its cut is
       sqrt((w - 1 + 2k) / (w + 1)) on its own k, exact by construction,
       and deeper than the rms window's at every w: with the window's k
       in it, 55 / 48 / 40 / 36 / 13% against 42 / 37 / 26 / 25 / 7 -
       within five points of the sample std's 57 / 43 / 37 / 33 / 16,
       which it reaches by a different road.
    2. Its estimate going into April is the window's. Same mean age, same
       level: leverage into April within 10% of the rms window's at every
       w, and April in ordinary months of its own estimate within three
       of the rms window's k.
    3. Its holdout Sharpe sits between the rms window and the newest-
       weight EWMA at 24 and 36 months - it cuts deeper than that EWMA
       and then re-levers on a 12.5-month half-life instead of carrying
       April for three years - and within 0.10 of the rms window at 6,
       where its half-life is two months and both have forgotten April by
       the summer.
    4. Sizing the momentum sleeve on the rms moves nothing. Its mean is
       under 1% of its mean square (a Sharpe-0.3 sleeve keeps s^2 / 12 of
       its variance there), so its leverage changes by under half a
       percent at every window, the allocator's HRP weight on it by under
       a point, and every book's holdout Sharpe by under 0.02. On the two
       EWMAs it moves more, because what they remember of the sleeve's
       own 2020 differs from the window's, but the sleeve runs de-levered
       at about 0.55x of a 10% target and its leverage into April stays
       within 10% of the std's; the book's Sharpe moves under 0.05.
    5. The allocator's HRP cut on the mean-age row is within five points
       of the rms row's 24 / 31 / 46 / 57 / 45%, because the series it is
       handed into April is the window's series.
    """
    out = {"holdout_from": holdout_from, "lookback": lookback, "settings": {}}
    for label, target, cap in settings:
        out["settings"][label] = {
            "vol_sleeve_only": {est: sleeve_window_sweep(volcarry, windows, lookback, holdout_from,
                                                         cap=cap, target=target, estimator=est)
                                for est in AGE_ORDER},
            "both_sleeves": {est: sleeve_window_sweep(volcarry, windows, lookback, holdout_from,
                                                      cap=cap, target=target, estimator=est,
                                                      xsmom_estimator=est)
                             for est in AGE_ORDER}}
    return out


def age_verdicts(sweep: dict, saved: dict | None = None, risk_based=RISK_BASED) -> dict:
    """The five predictions, scored at each setting. `saved` is section
    23's estimator_sweep, if the file has it, so the re-run rms and ewma
    rows can be checked against the ones already written down."""
    out = {"settings": list(sweep["settings"]), "by_setting": {}}
    for label, arms in sweep["settings"].items():
        solo, both = arms["vol_sleeve_only"], arms["both_sleeves"]
        per = {est: sleeve_window_verdicts(s, risk_based) for est, s in solo.items()}
        per_both = {est: sleeve_window_verdicts(s, risk_based) for est, s in both.items()}
        windows = per["rms"]["windows"]
        rows = {est: {int(w): r for w, r in solo[est]["windows"].items()} for est in solo}
        rows_both = {est: {int(w): r for w, r in both[est]["windows"].items()} for est in both}
        own = {est: per[est]["own_cut"] for est in per}
        sharpe = {est: per[est]["sleeve_sharpe"] for est in per}
        lev_in = {est: [rows[est][w]["leverage_into_worst"] for w in windows] for est in per}
        k_own = {est: [rows[est][w]["worst_in_ordinary_months_own_estimate"] for w in windows] for est in per}
        hrp = {est: per[est]["weight_cut"]["HRP"] for est in per}
        # the window's own cut with the mean-age EWMA's weight on April: what
        # prediction 1 said the row would be, before the estimate moved
        age_cut_on_window_k = [1 - 1 / np.sqrt((w - 1 + 2 * rows["rms"][w]["worst_in_ordinary_months_own_estimate"])
                                               / (w + 1)) for w in windows]
        saved_std = None
        if saved is not None and label in saved.get("settings", {}):
            std_rows = {int(w): r for w, r in saved["settings"][label]["std"]["windows"].items()}
            saved_std = {"own_cut": [std_rows[w]["own_cut"] for w in windows],
                         "sharpe": [std_rows[w]["sleeve"]["sharpe"] for w in windows],
                         "hrp_cut": [std_rows[w]["schemes"]["HRP"]["weight_cut_share"] for w in windows],
                         "xsmom_leverage_into": [std_rows[w]["xsmom"]["leverage_into_worst"]
                                                 if "xsmom" in std_rows[w] else None for w in windows]}
            match = {est: all(abs(rows[est][w]["own_cut"]
                                  - saved["settings"][label][est]["windows"][str(w)]["own_cut"]) < 1e-9
                              and abs(rows[est][w]["sleeve"]["sharpe"]
                                      - saved["settings"][label][est]["windows"][str(w)]["sleeve"]["sharpe"]) < 1e-9
                              for w in windows) for est in ("rms", "ewma")}
        else:
            match = None
        # 4. the momentum sleeve on the same estimator
        xs_lev = {est: {"std": [rows[est][w]["xsmom"]["leverage_into_worst"] for w in windows],
                        "same": [rows_both[est][w]["xsmom"]["leverage_into_worst"] for w in windows]}
                  for est in per}
        xs_ratio = {est: [a / b for a, b in zip(xs_lev[est]["same"], xs_lev[est]["std"])] for est in per}
        xs_hrp = {est: {"std": [rows[est][w]["schemes"]["HRP"]["xsmom_weight_into_worst"] for w in windows],
                        "same": [rows_both[est][w]["schemes"]["HRP"]["xsmom_weight_into_worst"] for w in windows]}
                  for est in per}
        names = list(rows["rms"][windows[0]]["schemes"])
        book_gap = {est: {n: [rows_both[est][w]["schemes"][n]["sharpe"] - rows[est][w]["schemes"][n]["sharpe"]
                              for w in windows] for n in names} for est in per}
        v = {
            "windows": windows,
            "section_23_rows_reproduced": match,
            "own_cut": own, "k_on_own_estimate": k_own,
            "vol_multiplier": {est: [rows[est][w]["vol_multiplier"] for w in windows] for est in per},
            "predicted_multiplier_exact": {est: [rows[est][w]["predicted_multiplier_exact"] for w in windows]
                                           for est in per},
            "newest_month_weight": {"window": [1 / w for w in windows],
                                    "ewma": [1 - ewma_lambda(w, "newest") for w in windows],
                                    "ewma_age": [1 - ewma_lambda(w, "age") for w in windows]},
            "half_life_months": {"ewma": [-np.log(2) / np.log(ewma_lambda(w, "newest")) for w in windows],
                                 "ewma_age": [-np.log(2) / np.log(ewma_lambda(w, "age")) for w in windows]},
            # 1. deeper than the rms window everywhere, near the std's cuts
            "age_cut_on_window_k": age_cut_on_window_k,
            "age_deeper_than_rms_everywhere": all(a > b for a, b in zip(own["ewma_age"], own["rms"])),
            "age_within_five_points_of_std": None if saved_std is None else
                [abs(a - b) < 0.05 for a, b in zip(own["ewma_age"], saved_std["own_cut"])],
            "std_own_cut": None if saved_std is None else saved_std["own_cut"],
            # 2. the window's estimate going in
            "leverage_into": lev_in,
            "age_leverage_over_rms": [a / b for a, b in zip(lev_in["ewma_age"], lev_in["rms"])],
            "age_leverage_within_tenth_of_rms": [abs(a / b - 1) < 0.10 for a, b in zip(lev_in["ewma_age"], lev_in["rms"])],
            "age_k_within_three_of_rms": [abs(a - b) < 3 for a, b in zip(k_own["ewma_age"], k_own["rms"])],
            # 3. the Sharpe between the two at 24 and 36, near the window at 6
            "sleeve_sharpe": sharpe,
            "std_sleeve_sharpe": None if saved_std is None else saved_std["sharpe"],
            "sleeve_april": {est: per[est]["sleeve_april"] for est in per},
            "leverage_holdout_mean": {est: [rows[est][w]["sleeve"]["mean_leverage_holdout"] for w in windows]
                                      for est in per},
            "age_between_rms_and_ewma_at_long_windows": all(
                min(sharpe["rms"][i], sharpe["ewma"][i]) <= sharpe["ewma_age"][i] <= max(sharpe["rms"][i], sharpe["ewma"][i])
                for i, w in enumerate(windows) if w in (24, 36)),
            "age_within_tenth_of_rms_at_six": all(abs(sharpe["ewma_age"][i] - sharpe["rms"][i]) < 0.10
                                                  for i, w in enumerate(windows) if w == 6),
            "age_sharpe_minus_rms": [a - b for a, b in zip(sharpe["ewma_age"], sharpe["rms"])],
            "age_sharpe_minus_ewma": [a - b for a, b in zip(sharpe["ewma_age"], sharpe["ewma"])],
            # 4. the momentum sleeve on the same estimator
            "xsmom_mean_share_of_mean_square": float(rows["rms"][windows[0]]["xsmom"]["mean_share_of_mean_square"]),
            "xsmom_leverage_into": xs_lev,
            "xsmom_leverage_ratio_same_over_std": xs_ratio,
            "xsmom_rms_moves_under_half_a_percent": all(abs(r - 1) < 0.005 for r in xs_ratio["rms"]),
            "xsmom_ewma_within_ten_percent": {est: all(abs(r - 1) < 0.10 for r in xs_ratio[est])
                                              for est in ("ewma", "ewma_age")},
            "xsmom_hrp_weight_into": xs_hrp,
            "xsmom_hrp_weight_moves_under_a_point_on_rms": all(
                abs(a - b) < 0.01 for a, b in zip(xs_hrp["rms"]["same"], xs_hrp["rms"]["std"])),
            "book_sharpe_change": book_gap,
            "book_sharpe_within_two_hundredths_on_rms": all(abs(g) < 0.02 for n in names for g in book_gap["rms"][n]),
            "book_sharpe_within_five_hundredths_on_ewmas": {est: all(abs(g) < 0.05 for n in names for g in book_gap[est][n])
                                                            for est in ("ewma", "ewma_age")},
            "xsmom_holdout_sharpe": {est: {"std": [rows[est][w]["xsmom"]["holdout_sharpe"] for w in windows],
                                           "same": [rows_both[est][w]["xsmom"]["holdout_sharpe"] for w in windows]}
                                     for est in per},
            # 5. the allocator's HRP cut on the mean-age row
            "hrp_cut": hrp,
            "hrp_cut_both": {est: per_both[est]["weight_cut"]["HRP"] for est in per_both},
            "age_hrp_within_five_points_of_rms": [abs(a - b) < 0.05 for a, b in zip(hrp["ewma_age"], hrp["rms"])],
            "scaled_april_in_allocator_months": {est: [rows[est][w]["scaled_worst_in_ordinary_months"] for w in windows]
                                                 for est in per},
            "hrp_weight_into": {est: [rows[est][w]["schemes"]["HRP"]["weight_into_worst"] for w in windows]
                                for est in per},
            "sharpe_gap": {est: per[est]["sharpe_gap"] for est in per},
            "risk_based_behind_equal_weight_everywhere": {
                est: all(g < 0 for n in risk_based for g in per_both[est]["sharpe_gap"][n]) for est in per_both},
        }
        out["by_setting"][label] = v
    return out


def print_age(sweep: dict, verdicts: dict) -> None:
    print("\nMATCH THE AGE   section 23's kept-mean rows beside an EWMA whose lambda is set by the "
          "mean age, 1 - 2/(w + 1), and then the momentum sleeve sized on the same estimator")
    for label, v in verdicts["by_setting"].items():
        windows = v["windows"]
        print(f"\n   {label}")
        if v["section_23_rows_reproduced"] is not None:
            print("   section 23's rms and ewma rows reproduced: " + ", ".join(
                f"{est} {'yes' if ok else 'NO'}" for est, ok in v["section_23_rows_reproduced"].items()))
        print(f"{'window':>7} {'w(new) win':>10} {'ewma':>6} {'age':>6} {'t1/2 ewma':>10} {'age':>6}   "
              + " ".join(f"{est + ' cut':>9} {'x':>5} {'k':>5}" for est in AGE_ORDER) + f" {'std cut':>8}")
        for i, w in enumerate(windows):
            cells = " ".join(f"{v['own_cut'][est][i]:>8.0%} {v['vol_multiplier'][est][i]:>5.2f} "
                             f"{v['k_on_own_estimate'][est][i]:>5.0f}" for est in AGE_ORDER)
            std = "" if v["std_own_cut"] is None else f"{v['std_own_cut'][i]:>8.0%}"
            print(f"{w:>5}mo {v['newest_month_weight']['window'][i]:>10.3f} {v['newest_month_weight']['ewma'][i]:>6.3f} "
                  f"{v['newest_month_weight']['ewma_age'][i]:>6.3f} {v['half_life_months']['ewma'][i]:>10.1f} "
                  f"{v['half_life_months']['ewma_age'][i]:>6.1f}   {cells} {std}")
        print(f"\n{'window':>7} " + " ".join(f"{est + ' in':>9} {est + ' Apr':>10} {est + ' Sh':>9}" for est in AGE_ORDER)
              + f" {'std Sh':>7}   " + " ".join(f"{est + ' HRP':>9}" for est in AGE_ORDER))
        for i, w in enumerate(windows):
            cells = " ".join(f"{v['leverage_into'][est][i]:>8.2f}x {v['sleeve_april'][est][i]:>+10.1%} "
                             f"{v['sleeve_sharpe'][est][i]:>+9.2f}" for est in AGE_ORDER)
            std = "" if v["std_sleeve_sharpe"] is None else f"{v['std_sleeve_sharpe'][i]:>+7.2f}"
            hrp = " ".join(f"{v['hrp_cut'][est][i]:>9.0%}" for est in AGE_ORDER)
            print(f"{w:>5}mo {cells} {std}   {hrp}")
        print("   the mean-age cut the window's own k would give: "
              + "/".join(f"{c:.0%}" for c in v["age_cut_on_window_k"]))
        print("   mean holdout leverage: " + "; ".join(
            f"{est} " + "/".join(f"{x:.1f}" for x in v["leverage_holdout_mean"][est]) for est in AGE_ORDER))
        print(f"\n   the momentum sleeve on the same estimator (its mean is "
              f"{v['xsmom_mean_share_of_mean_square']:.2%} of its mean square):")
        print(f"{'window':>7} " + " ".join(f"{est + ' lev':>10} {'same':>6} {'HRP w':>6} {'same':>6}" for est in AGE_ORDER))
        for i, w in enumerate(windows):
            cells = " ".join(f"{v['xsmom_leverage_into'][est]['std'][i]:>9.3f}x {v['xsmom_leverage_into'][est]['same'][i]:>5.3f}x "
                             f"{v['xsmom_hrp_weight_into'][est]['std'][i]:>6.2f} {v['xsmom_hrp_weight_into'][est]['same'][i]:>6.2f}"
                             for est in AGE_ORDER)
            print(f"{w:>5}mo {cells}")
        print("   largest book Sharpe change from re-sizing momentum: " + "; ".join(
            f"{est} {max(abs(g) for n in v['book_sharpe_change'][est] for g in v['book_sharpe_change'][est][n]):.3f}"
            for est in AGE_ORDER))
        print("\n   the predictions:")
        print(f"   1. age cut deeper than rms everywhere: {v['age_deeper_than_rms_everywhere']}; "
              + ("" if v["age_within_five_points_of_std"] is None else
                 f"within five points of std at {sum(v['age_within_five_points_of_std'])} of {len(windows)}"))
        print(f"   2. age leverage within 10% of rms at {sum(v['age_leverage_within_tenth_of_rms'])} of {len(windows)} "
              "(ratio " + "/".join(f"{r:.2f}" for r in v["age_leverage_over_rms"]) + "); "
              f"k within three at {sum(v['age_k_within_three_of_rms'])}")
        print(f"   3. age Sharpe between rms and ewma at 24 and 36: {v['age_between_rms_and_ewma_at_long_windows']}; "
              f"within 0.10 of rms at 6: {v['age_within_tenth_of_rms_at_six']} "
              "(age - rms " + "/".join(f"{d:+.2f}" for d in v["age_sharpe_minus_rms"])
              + ", age - ewma " + "/".join(f"{d:+.2f}" for d in v["age_sharpe_minus_ewma"]) + ")")
        print(f"   4. momentum on rms moves leverage under 0.5%: {v['xsmom_rms_moves_under_half_a_percent']}, "
              f"HRP weight under a point: {v['xsmom_hrp_weight_moves_under_a_point_on_rms']}, "
              f"book Sharpe under 0.02: {v['book_sharpe_within_two_hundredths_on_rms']}; on the EWMAs leverage within 10%: "
              + ", ".join(f"{e} {ok}" for e, ok in v["xsmom_ewma_within_ten_percent"].items())
              + ", book Sharpe under 0.05: " + ", ".join(f"{e} {ok}" for e, ok in v["book_sharpe_within_five_hundredths_on_ewmas"].items()))
        print(f"   5. age HRP cut within five points of rms at {sum(v['age_hrp_within_five_points_of_rms'])} of {len(windows)}; "
              "risk-based behind 1/N everywhere with both sleeves re-sized: " + ", ".join(
            f"{est} {'yes' if ok else 'no'}" for est, ok in v["risk_based_behind_equal_weight_everywhere"].items()))


# ---------- reporting ----------

def print_holdout(study: dict) -> None:
    s = study["sleeve"]
    a = study["sleeve_agreement"]
    print(f"\nTHE 2025 HOLDOUT   vol sleeve re-run to {s['extended_last']} (main study stops {s['shipped_last']})")
    print(f"\n   the re-run against the shipped series on the {a['months_compared']} months both have: "
          f"max |diff| {a['max_abs_diff'] * 1e4:.1f} bps ({a['max_abs_diff_month']}), "
          f"{a['sign_flips']} sign flips, corr {a['correlation']:.4f}")
    print(f"\n   the sleeve's {s['new_months']} new months: Sharpe {s['holdout']['sharpe']:+.2f} "
          f"against {s['seen']['sharpe']:+.2f} on the {len(study['sleeve_agreement']) and 'seen'} history; "
          f"worst {s['worst_new_month']} at {s['worst_new_month_return']:+.2%}, "
          f"which {s['months_worse_in_history']} months of the history beat on the downside")
    print(f"\n{'month':>8} {'ret':>8}")
    for d, v in s["path"]:
        print(f"{d:>8} {v:>+8.2%}")
    for lookback, lb in study["lookbacks"].items():
        print(f"\n   {lookback}-month window, capped 5x - Sharpe, and the bootstrap gap to equal weight:")
        print(f"\n{'scheme':>14} " + " ".join(f"{k:>30}" for k in ("seen", "holdout", "all")))
        print(f"{'':>14} " + " ".join(f"{lb[k]['months']:>5} mo {lb[k]['first']}..{lb[k]['last']:<10}" for k in ("seen", "holdout", "all")))
        for name in lb["all"]["schemes"]:
            cells = []
            for k in ("seen", "holdout", "all"):
                sc = lb[k]["schemes"][name]
                b = sc["vs_equal_weight"]
                cells.append(f"{sc['sharpe']:>+6.2f}  [{b['low']:>+5.2f},{b['high']:>+5.2f}] p{b['p_not_ahead']:.2f}")
            print(f"{name:>14} " + " ".join(f"{c:>30}" for c in cells))
        print("\n   worst holdout month: " + ", ".join(f"{n} {d}" for n, d in lb["worst_holdout_month"].items()))
        print("\n   the holdout month by month, and the weight on the vol sleeve going into it:")
        names = list(lb["holdout_path"])
        print(f"\n{'month':>8} " + " ".join(f"{n[:12]:>19}" for n in names))
        for k, (d, _) in enumerate(lb["holdout_path"][names[0]]):
            print(f"{d:>8} " + " ".join(
                f"{lb['holdout_path'][n][k][1]:>+8.2%} @{lb['volcarry_weight_path'][n][k][1]:>5.0%}" for n in names))


def main() -> None:
    study = holdout_study(LOOKBACKS, HOLDOUT_FROM)
    print_holdout(study)
    panel = capped_panel(frames(volcarry_extended()))
    study["window_sweep"] = window_sweep(panel, WINDOWS, HOLDOUT_FROM)
    study["window_verdicts"] = window_verdicts(study["window_sweep"])
    print_window_sweep(study["window_sweep"], study["window_verdicts"])
    study["sleeve_window_sweep"] = sleeve_window_sweep(volcarry_extended(), SLEEVE_WINDOWS,
                                                       LOOKBACK_MONTHS, HOLDOUT_FROM)
    study["sleeve_window_verdicts"] = sleeve_window_verdicts(study["sleeve_window_sweep"])
    print_sleeve_window_sweep(study["sleeve_window_sweep"], study["sleeve_window_verdicts"])
    study["unpinned_sweep"] = unpinned_sweep(volcarry_extended(), UNPINNED, SLEEVE_WINDOWS,
                                             LOOKBACK_MONTHS, HOLDOUT_FROM)
    study["unpinned_verdicts"] = unpinned_verdicts(study["unpinned_sweep"])
    print_unpinned(study["unpinned_sweep"], study["unpinned_verdicts"])
    study.update(estimator_section())
    study.update(age_section(study))
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(study, fh, indent=1, default=float)
    print(f"\nwrote {OUT}")


def estimator_section() -> dict:
    sweep = estimator_sweep(volcarry_extended(), KEPT_MEAN, ESTIMATOR_ORDER, SLEEVE_WINDOWS,
                            LOOKBACK_MONTHS, HOLDOUT_FROM)
    verdicts = estimator_verdicts(sweep)
    print_estimator(sweep, verdicts)
    return {"estimator_sweep": sweep, "estimator_verdicts": verdicts}


def age_section(study: dict | None = None) -> dict:
    sweep = age_sweep(volcarry_extended(), KEPT_MEAN, SLEEVE_WINDOWS, LOOKBACK_MONTHS, HOLDOUT_FROM)
    verdicts = age_verdicts(sweep, None if study is None else study.get("estimator_sweep"))
    print_age(sweep, verdicts)
    return {"age_sweep": sweep, "age_verdicts": verdicts}


def main_age_only() -> None:
    """Section 24 on its own, beside sections 19-23."""
    with open(OUT) as fh:
        study = json.load(fh)
    study.update(age_section(study))
    with open(OUT, "w") as fh:
        json.dump(study, fh, indent=1, default=float)
    print(f"\nwrote {OUT}")


def main_estimator_only() -> None:
    """Section 23 on its own, written beside the sections already in the
    file instead of over them, so a new row does not re-run (and re-round)
    four sections that have not changed."""
    with open(OUT) as fh:
        study = json.load(fh)
    study.update(estimator_section())
    with open(OUT, "w") as fh:
        json.dump(study, fh, indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    import sys
    if "--age" in sys.argv[1:]:
        main_age_only()
    elif "--estimator" in sys.argv[1:]:
        main_estimator_only()
    else:
        main()
