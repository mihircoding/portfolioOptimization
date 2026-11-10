"""Thirteen months nothing in multi_strategy.py has seen.

Usage:  python holdout.py
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

from multi_strategy import (BETA_WINDOW_MONTHS, LEVERAGE_CAP, LOOKBACK_MONTHS,
                            VOL_TARGET, VOL_WINDOW_MONTHS, allocators, beta_hedged,
                            estimate, market_monthly, paired_sharpe_bootstrap,
                            performance, statarb_monthly, statarb_trailing_vol,
                            trailing_vol, vol_targeted, volcarry_monthly,
                            xsmom_monthly)

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

def frames(volcarry: pd.Series, target: float = VOL_TARGET, window: int = VOL_WINDOW_MONTHS,
           beta_window: int = BETA_WINDOW_MONTHS) -> dict:
    """multi_strategy.sleeve_frames with the vol sleeve handed in, so the
    only thing different from section 11 is how long that sleeve runs."""
    market = market_monthly()
    xsmom = beta_hedged(xsmom_monthly(), market, beta_window)["hedged"].rename("xsmom")
    statarb = statarb_monthly()
    statarb.index = statarb.index.to_period("M").to_timestamp("M")
    volcarry = volcarry.copy()
    volcarry.index = volcarry.index.to_period("M").to_timestamp("M")
    return {"statarb": vol_targeted(statarb, statarb_trailing_vol(), target),
            "volcarry": vol_targeted(volcarry, trailing_vol(volcarry, window), target),
            "xsmom": vol_targeted(xsmom, trailing_vol(xsmom, window), target)}


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
    print(f"\n   the sleeve's vol estimate the month after, over the month before: " + ", ".join(
        f"{w}mo x{by[w]['sleeve_vol_multiplier']:.2f}" for w in windows))
    names = list(by[windows[0]]["schemes"])
    print(f"\n   weight on the sleeve going into {sweep['worst_month']} -> the month after (cut in points):")
    print(f"\n{'window':>7} " + " ".join(f"{n[:12]:>22}" for n in names))
    for w in windows:
        print(f"{w:>5}mo " + " ".join(
            f"{by[w]['schemes'][n]['weight_into_worst']:>6.0%} -> {by[w]['schemes'][n]['weight_after_worst']:>4.0%} "
            f"({by[w]['schemes'][n]['cut'] * 100:>+4.0f})" for n in names))
    print(f"\n   {sweep['worst_month']} return, then months behind equal weight of the {n_after} after it, "
          f"and the holdout Sharpe gap to equal weight [bootstrap 90%]:")
    print(f"\n{'window':>7} " + " ".join(f"{n[:12]:>28}" for n in names))
    for w in windows:
        cells = []
        for n in names:
            s = by[w]["schemes"][n]
            b = s["vs_equal_weight"]
            cells.append(f"{s['worst_month_return']:>+6.1%} {s['months_behind_after_worst']:>2}/{n_after} "
                         f"{s['sharpe_gap']:>+5.2f} [{b['low']:>+5.2f},{b['high']:>+5.2f}]")
        print(f"{w:>5}mo " + " ".join(f"{c:>28}" for c in cells))
    print(f"\n   the predictions:")
    for name in RISK_BASED:
        print(f"   {name:>13}: cut shrinks with the window {str(verdicts['cut_shrinks_with_window'][name]):<5} "
              f"(42mo cut is {verdicts['cut_longest_over_shortest'][name]:.2f} of 6mo's; 1/window says "
              f"{verdicts['one_over_window_would_give'][name]:.2f}); gap closes {str(verdicts['gap_closes_with_window'][name]):<5} "
              f"and stays negative {str(verdicts['gap_negative_at_every_window'][name]):<5} best window {verdicts['best_window_for'][name]}mo")
    print(f"   {'max Sharpe':>13}: heaviest in the sleeve going in at {verdicts['max_sharpe_heaviest_at']}mo; "
          f"best window {verdicts['best_window_for']['max Sharpe']}mo; gap closes "
          f"{verdicts['gap_closes_with_window']['max Sharpe']}, negative throughout "
          f"{verdicts['gap_negative_at_every_window']['max Sharpe']}")


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
        print(f"\n   worst holdout month: " + ", ".join(f"{n} {d}" for n, d in lb["worst_holdout_month"].items()))
        print(f"\n   the holdout month by month, and the weight on the vol sleeve going into it:")
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
    os.makedirs("results", exist_ok=True)
    with open(OUT, "w") as fh:
        json.dump(study, fh, indent=1, default=float)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
