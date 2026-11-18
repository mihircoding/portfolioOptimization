"""Sticky strike proper, and a limit on how fast the delta may move.

Usage:  python -m sleeves.volcarry.sticky_study        (reads results/spy_vix_daily.csv)

remark_study.py left two questions open. The strike-vol arm reads the
strike's vol off a smile centred on TODAY's spot (sticky moneyness: the
smile moves with the index, and a strike that has gone out of the money is
read on the wing). The other rule a desk uses is sticky strike: the smile
is pinned in strike space where it was sold, the index moves across it, and
a strike keeps its vol while the at-the-money vol changes because the money
has moved to a different point of the same smile. The caveat under the
strike-vol arm said a sticky-strike arm "should sit between capped and
strike-vol". This runs it.

What "sticky strike" means here, exactly. The call was sold at the money,
so on the day it was sold its strike WAS the level. Pin the fitted SPY smile
(the shape from ssvi_study.py, as remark_study.py uses it) at the strike.
On a later day the index is at S and VIX is the at-the-money vol at S, which
on the pinned smile is the point k = log(S / K). The level of the pinned
smile is therefore the number L that puts the smile's reading at k equal
to VIX - `sticky_strike_level()` solves for it by fixed point - and the
strike's own vol is that level, because the strike sits at the pinned
smile's k = 0. In a sell-off S is under K, log(S / K) is on the put side,
the put side is ABOVE the level, so the strike's vol sits below VIX by the
height of the put wing at that moneyness. The strike-vol arm reads the call
wing of a smile centred at S at the mirror-image moneyness, and the call
wing is lower, so the two rules disagree by the skew's asymmetry, not by
the skew.

The second thing is a limit on the hedge itself rather than on the vol.
The capped arm kept the down-marks and lost the up-marks by refusing to let
the vol rise; the note said a cap on how far the DELTA may move in a day
would slow the hedges that were right as well as the ones that were wrong,
and left it there. `delta_hedge(max_delta_move=...)` is that cap, on the
re-marked arm, at 0.05, 0.10 and 0.20 shares per option per day. The first
hedge and the liquidation are never limited.

Predictions, written before the run:

  9.  The sticky-strike arm's March 2020 is between the sold-at arm's
      (-2.70) and the strike-vol arm's (-3.85), and nearer the sold-at end:
      on the way down the strike's vol sits further below VIX than the
      strike-vol arm's call-wing read does, because the put wing is the
      high side of an equity smile. Its std is between 0.640 and 0.666.
  10. Over all hedge days the strike's vol under sticky strike averages
      BELOW VIX - the opposite sign to the strike-vol arm's +3.0 points -
      because the index spends more of a month's hedge days under the
      strike than over it only in sell-offs, but when it is over it the call
      side near the money is the smile's dip and the level is barely above
      VIX, while when it is under it the put side climbs fast. Its mean is
      within 0.03 of sold-at and it correlates with re-marked above 0.97.
  11. The delta limit: at 0.05 a day the re-marked arm's March 2020 gets
      more than half of the way back to the sold-at arm's (the sell-off
      moved the delta by more than 0.05 on most of its days) and its std
      falls below the re-marked arm's. The cost is in the calm months: in
      the 214 months where VIX did not rise ten points the limited arm's
      mean is below the re-marked arm's, and in the 24 spike months it is
      above. The limit saves the spike months and costs the calm ones.
  12. The limit binds on 25-45% of hedge days at 0.05, 5-15% at 0.10 and
      under 3% at 0.20, and at 0.20 the arm is the re-marked arm within
      0.01 on both mean and std.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from sleeves.volcarry.hedging import delta_hedge
from sleeves.volcarry.hedging_study import (DAILY_CACHE, HEDGE_DAYS, OPTION_SPREAD_VIX_ANCHOR,
                                            OPTION_SPREAD_VOL_PTS, RATE, T_YEARS,
                                            half_spread_vol_pts, load, months, summarize)
from sleeves.volcarry.remark_study import OUT as REMARK_OUT, SPY_SSVI, strike_vol
from sleeves.volcarry.ssvi import SSVIParams

OUT = "results/sticky_study.json"
LIMITS = (0.05, 0.10, 0.20)
ARMS = ("sold-at", "re-marked", "strike-vol", "sticky-strike") + tuple(f"limit-{c:.2f}" for c in LIMITS)


def sticky_strike_level(spot, strike, time_left, atm_vol, surface: SSVIParams = SPY_SSVI,
                        iterations: int = 12):
    """The level of a smile pinned at `strike` that reads `atm_vol` at `spot`.

    The pinned smile's at-the-money point is the strike. Today's index is at
    log-moneyness k = log(spot / strike) on it, and today's VIX is what the
    smile reads there. The smile's reading at k is strike_vol(spot=strike,
    strike=spot, ...) at a level L, so L is the fixed point of

        L  =  atm_vol - (strike_vol(strike, spot, t, L) - L)

    which converges in a handful of steps because the wing height moves
    slowly with the level. The strike's own vol is L. At spot == strike it
    is atm_vol exactly, whatever the surface. Floored at one vol point so
    the extrapolated last-week wing cannot drive it through zero. Arrays
    broadcast, so one call does a month of hedge days.
    """
    spot = np.asarray(spot, dtype=float)
    atm_vol = np.asarray(atm_vol, dtype=float)
    level = np.array(np.broadcast_to(atm_vol, np.broadcast(spot, atm_vol, time_left).shape), dtype=float)
    for _ in range(iterations):
        wing = strike_vol(strike, spot, time_left, level, surface) - level
        level = np.maximum(atm_vol - wing, 0.01)
    return level


def trade_sticky_ways(window: pd.DataFrame, option_type: str = "call",
                      option_spread_vol_pts: float = OPTION_SPREAD_VOL_PTS,
                      option_spread_vix_anchor: float | None = OPTION_SPREAD_VIX_ANCHOR,
                      limits: tuple = LIMITS) -> dict:
    """One month: the sold-at, re-marked and strike-vol arms again (so the
    tape can be checked against remark_study.json), the sticky-strike arm,
    and the re-marked arm under each per-day delta limit."""
    path = window["spy"].to_numpy(dtype=float)
    vix = window["vix"].to_numpy(dtype=float) / 100.0
    sigma = float(vix[0])
    strike = path[0]
    charged = half_spread_vol_pts(float(window["vix"].iloc[0]), option_spread_vol_pts,
                                  option_spread_vix_anchor)
    sold_at = max(sigma - charged / 100.0, 1e-6)
    scale = 100.0 / path[0]
    n_steps = len(path) - 1

    def run(hedge_vols, **kw):
        return delta_hedge(path, strike, T_YEARS, RATE, sold_at, option_type=option_type,
                           sigma_hedge=hedge_vols, **kw)

    time_left = T_YEARS - np.arange(n_steps) * T_YEARS / n_steps
    sold = run(sigma)
    marked = run(vix[:n_steps])
    own = strike_vol(path[:n_steps], strike, time_left, vix[:n_steps])
    off_skew = run(own)
    pinned = sticky_strike_level(path[:n_steps], strike, time_left, vix[:n_steps])
    sticky = run(pinned)
    under = path[:n_steps] < strike
    row = {"date": window.index[0], "implied": sigma, "realized_vol": sold["realized_vol"],
           "vix_max": float(vix[:n_steps].max()),
           "sold-at": sold["pnl"] * scale, "re-marked": marked["pnl"] * scale,
           "strike-vol": off_skew["pnl"] * scale, "sticky-strike": sticky["pnl"] * scale,
           # where each rule put the strike relative to VIX, in vol points
           "sticky_mean_pts": float((pinned - vix[:n_steps]).mean() * 100.0),
           "sticky_min_pts": float((pinned - vix[:n_steps]).min() * 100.0),
           "sticky_max_pts": float((pinned - vix[:n_steps]).max() * 100.0),
           "strike_vol_mean_pts": float((own - vix[:n_steps]).mean() * 100.0),
           "days_under_strike": int(under.sum()),
           "sticky_pts_under": float((pinned - vix[:n_steps])[under].mean() * 100.0) if under.any() else np.nan,
           "sticky_pts_over": float((pinned - vix[:n_steps])[~under].mean() * 100.0) if (~under).any() else np.nan,
           "strike_vol_pts_under": float((own - vix[:n_steps])[under].mean() * 100.0) if under.any() else np.nan,
           # how big the re-marked arm's daily delta moves were, so the limit
           # can be read against what it is limiting
           "max_daily_delta_move": float(np.abs(np.diff(marked["deltas"])).max()),
           "shares_re_marked": marked["shares_traded"]}
    moves = np.abs(np.diff(marked["deltas"]))
    for c in limits:
        limited = run(vix[:n_steps], max_delta_move=c)
        row[f"limit-{c:.2f}"] = limited["pnl"] * scale
        row[f"days_bound_{c:.2f}"] = int((moves > c + 1e-12).sum())
        row[f"shares_limit_{c:.2f}"] = limited["shares_traded"]
    return row


def agreement_with_remark_study(frame: pd.DataFrame, path: str = REMARK_OUT) -> dict:
    """The three repeated arms against what remark_study.py saved, so the
    two studies are shown to be on the same tape and the same arithmetic."""
    with open(path) as fh:
        saved = json.load(fh)
    out = {}
    for arm in ("sold-at", "re-marked", "strike-vol"):
        theirs = pd.Series(saved["monthly"][arm], index=saved["months"])
        mine = pd.Series(frame[arm].to_numpy(), index=[f"{d:%Y-%m}" for d in frame.index])
        both = pd.concat([mine.rename("new"), theirs.rename("old")], axis=1).dropna()
        out[arm] = {"months_compared": int(len(both)),
                    "max_abs_diff": float((both["new"] - both["old"]).abs().max())}
    return out


def verdicts(stats: dict, frame: pd.DataFrame) -> list[dict]:
    std = {a: stats[a]["std"] for a in ARMS}
    mean = {a: stats[a]["mean"] for a in ARMS}
    mar20 = frame.loc["2020-03"].iloc[0]
    spike = frame[(frame["vix_max"] - frame["implied"]) > 0.10]
    calm = frame[(frame["vix_max"] - frame["implied"]) <= 0.10]
    n_days = len(frame) * HEDGE_DAYS
    bound = {c: float(frame[f"days_bound_{c:.2f}"].sum() / n_days) for c in LIMITS}
    corr = float(frame["sticky-strike"].corr(frame["re-marked"]))
    gap = mar20["sold-at"] - mar20["re-marked"]
    recovered = (mar20["limit-0.05"] - mar20["re-marked"]) / gap if gap else np.nan
    return [
        {"claim": "sticky-strike's March 2020 is between sold-at and strike-vol, nearer sold-at; std between 0.640 and 0.666",
         "measured": f"2020-03 sold-at {mar20['sold-at']:+.3f}, sticky {mar20['sticky-strike']:+.3f}, strike-vol "
                     f"{mar20['strike-vol']:+.3f}; std sticky {std['sticky-strike']:.3f}",
         "held": bool(mar20["strike-vol"] < mar20["sticky-strike"] < mar20["sold-at"]
                      and abs(mar20["sticky-strike"] - mar20["sold-at"]) < abs(mar20["sticky-strike"] - mar20["strike-vol"])
                      and std["sold-at"] <= std["sticky-strike"] <= std["strike-vol"])},
        {"claim": "under sticky strike the strike's vol averages below VIX; mean within 0.03 of sold-at; correlation with re-marked above 0.97",
         "measured": f"mean offset {frame['sticky_mean_pts'].mean():+.2f} pts (strike-vol arm {frame['strike_vol_mean_pts'].mean():+.2f}); "
                     f"means sold-at {mean['sold-at']:+.3f}, sticky {mean['sticky-strike']:+.3f}; correlation {corr:.3f}",
         "held": bool(frame["sticky_mean_pts"].mean() < 0 and abs(mean["sticky-strike"] - mean["sold-at"]) < 0.03
                      and corr > 0.97)},
        {"claim": "a 0.05 limit gets more than half of March 2020 back, has a lower std than re-marked, loses in calm months and wins in spike months",
         "measured": f"2020-03 re-marked {mar20['re-marked']:+.3f}, limit-0.05 {mar20['limit-0.05']:+.3f} ({recovered:.0%} of the way to "
                     f"sold-at); std {std['limit-0.05']:.3f} vs {std['re-marked']:.3f}; calm months {calm['limit-0.05'].mean():+.3f} vs "
                     f"{calm['re-marked'].mean():+.3f}, spike months {spike['limit-0.05'].mean():+.3f} vs {spike['re-marked'].mean():+.3f}",
         "held": bool(recovered > 0.5 and std["limit-0.05"] < std["re-marked"]
                      and calm["limit-0.05"].mean() < calm["re-marked"].mean()
                      and spike["limit-0.05"].mean() > spike["re-marked"].mean())},
        {"claim": "the limit binds on 25-45% of days at 0.05, 5-15% at 0.10, under 3% at 0.20; limit-0.20 is re-marked within 0.01 on mean and std",
         "measured": ", ".join(f"{c:.2f}: {b:.1%}" for c, b in bound.items())
                     + f"; limit-0.20 mean {mean['limit-0.20']:+.3f} std {std['limit-0.20']:.3f} vs re-marked "
                       f"{mean['re-marked']:+.3f} / {std['re-marked']:.3f}",
         "held": bool(0.25 <= bound[0.05] <= 0.45 and 0.05 <= bound[0.10] <= 0.15 and bound[0.20] < 0.03
                      and abs(mean["limit-0.20"] - mean["re-marked"]) < 0.01
                      and abs(std["limit-0.20"] - std["re-marked"]) < 0.01)},
    ]


def main() -> None:
    data = load(end="2026-12-31", cache=DAILY_CACHE)
    windows = months(data)
    frame = pd.DataFrame([trade_sticky_ways(w) for w in windows]).set_index("date")
    stats = {a: summarize(frame.rename(columns={a: "pnl"})) for a in ARMS}
    print(f"Short one at-the-money SPY call a month, hedged daily: sticky strike, and the re-marked "
          f"hedge under a per-day delta limit.\n{len(frame)} trades, {frame.index[0].date()} to "
          f"{frame.index[-1].date()}, per $100 of underlying.\n")
    print(f"{'':<24} " + " ".join(f"{a:>13}" for a in ARMS))
    for key, label, fmt in (("mean", "average P&L", "{:>13.3f}"), ("std", "std dev of P&L", "{:>13.3f}"),
                            ("sharpe", "annualized Sharpe", "{:>13.2f}"), ("win_rate", "months profitable", "{:>13.1%}"),
                            ("worst", "worst month", "{:>13.3f}"), ("best", "best month", "{:>13.3f}")):
        print(f"{label:<24} " + " ".join(fmt.format(stats[a][key]) for a in ARMS))
    for label in ("2020-03", "2008-10"):
        r = frame.loc[label].iloc[0]
        print(f"{label:<24} " + " ".join(f"{r[a]:>13.3f}" for a in ARMS))
    spike = frame[(frame["vix_max"] - frame["implied"]) > 0.10]
    calm = frame.drop(spike.index)
    print(f"{f'{len(spike)} spike months':<24} " + " ".join(f"{spike[a].mean():>13.3f}" for a in ARMS))
    print(f"{f'{len(calm)} calm months':<24} " + " ".join(f"{calm[a].mean():>13.3f}" for a in ARMS))
    print(f"{'  beat sold-at in spikes':<24} " + " ".join(f"{int((spike[a] > spike['sold-at']).sum()):>13d}" for a in ARMS))
    print(f"{'shares traded':<24} " + " ".join(f"{v:>13.2f}" for v in
          (np.nan, frame["shares_re_marked"].mean(), np.nan, np.nan,
           *[frame[f"shares_limit_{c:.2f}"].mean() for c in LIMITS])))

    under = frame["days_under_strike"].sum()
    print(f"\nSticky strike put the strike's vol {frame['sticky_mean_pts'].mean():+.2f} points from VIX on average "
          f"(strike-vol arm: {frame['strike_vol_mean_pts'].mean():+.2f}); on the {under} hedge days the index was "
          f"under the strike {np.nanmean(frame['sticky_pts_under']):+.2f} (strike-vol {np.nanmean(frame['strike_vol_pts_under']):+.2f}), "
          f"on the {len(frame) * HEDGE_DAYS - under} it was over {np.nanmean(frame['sticky_pts_over']):+.2f}")
    m = frame.loc["2020-03"].iloc[0]
    print(f"  March 2020: sticky mean {m['sticky_mean_pts']:+.2f}, range {m['sticky_min_pts']:+.2f} to {m['sticky_max_pts']:+.2f}; "
          f"the re-marked delta's biggest daily move was {m['max_daily_delta_move']:.3f}")
    n_days = len(frame) * HEDGE_DAYS
    for c in LIMITS:
        b = frame[f"days_bound_{c:.2f}"]
        print(f"Limit {c:.2f} bound on {b.sum() / n_days:.1%} of hedge days, in {int((b == 0).sum())} months never, "
              f"in March 2020 on {int(m[f'days_bound_{c:.2f}'])} of 20")
    print(f"Median month's biggest daily delta move (re-marked): {frame['max_daily_delta_move'].median():.3f}; "
          f"share of months where it exceeded 0.20: {(frame['max_daily_delta_move'] > 0.20).mean():.0%}")

    agree = agreement_with_remark_study(frame)
    print("\nTape check against remark_study.json: " + "; ".join(
        f"{a} {v['months_compared']} months, max abs diff {v['max_abs_diff']:.2e}" for a, v in agree.items()))

    scored = verdicts(stats, frame)
    print("\nPredictions, scored:")
    for v in scored:
        print(f"  [{'held' if v['held'] else 'FAILED'}] {v['claim']}\n         measured: {v['measured']}")

    with open(OUT, "w") as fh:
        json.dump({"n": int(len(frame)), "from": str(frame.index[0].date()), "to": str(frame.index[-1].date()),
                   "limits": list(LIMITS),
                   "arms": {a: {k: float(v) for k, v in stats[a].items()} for a in ARMS},
                   "march_2020": {a: float(m[a]) for a in ARMS},
                   "october_2008": {a: float(frame.loc["2008-10"].iloc[0][a]) for a in ARMS},
                   "spike_months": {"n": int(len(spike)), "mean": {a: float(spike[a].mean()) for a in ARMS},
                                    "beat_sold_at": {a: int((spike[a] > spike["sold-at"]).sum()) for a in ARMS}},
                   "calm_months": {"n": int(len(calm)), "mean": {a: float(calm[a].mean()) for a in ARMS}},
                   "sticky_pts": {"mean": float(frame["sticky_mean_pts"].mean()),
                                  "under_strike": float(np.nanmean(frame["sticky_pts_under"])),
                                  "over_strike": float(np.nanmean(frame["sticky_pts_over"])),
                                  "strike_vol_mean": float(frame["strike_vol_mean_pts"].mean()),
                                  "strike_vol_under_strike": float(np.nanmean(frame["strike_vol_pts_under"])),
                                  "days_under_strike": int(under),
                                  "march_2020": {"mean": float(m["sticky_mean_pts"]), "min": float(m["sticky_min_pts"]),
                                                 "max": float(m["sticky_max_pts"])}},
                   "limit_binds": {f"{c:.2f}": {"share_of_days": float(frame[f"days_bound_{c:.2f}"].sum() / n_days),
                                                "months_never": int((frame[f"days_bound_{c:.2f}"] == 0).sum()),
                                                "march_2020_days": int(m[f"days_bound_{c:.2f}"]),
                                                "shares_traded": float(frame[f"shares_limit_{c:.2f}"].mean())}
                                   for c in LIMITS},
                   "shares_traded_re_marked": float(frame["shares_re_marked"].mean()),
                   "max_daily_delta_move_median": float(frame["max_daily_delta_move"].median()),
                   "correlation_sticky_re_marked": float(frame["sticky-strike"].corr(frame["re-marked"])),
                   "correlation_sticky_strike_vol": float(frame["sticky-strike"].corr(frame["strike-vol"])),
                   "tape_agreement": agree, "verdicts": scored,
                   "monthly": {a: [round(float(x), 5) for x in frame[a]] for a in ARMS},
                   "months": [f"{d:%Y-%m}" for d in frame.index]}, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
