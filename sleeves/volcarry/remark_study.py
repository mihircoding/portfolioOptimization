"""Which vol do you hedge with? The one you sold at, today's, or the true one.

Usage:  python -m sleeves.volcarry.remark_study        (reads results/spy_vix_daily.csv)

Every month of the hedging study sells an at-the-money call at VIX and then
hedges it to expiry with the delta computed at THAT vol - the vol the option
was sold at, never re-marked. The caveats paragraph has said since the study
was written that a desk re-hedges on current implied and that this "damps the
tails", and left it there. This runs the hedge three ways on the same tape:

    sold-at     delta at the inception VIX every day        (the study as shipped)
    re-marked   delta at that day's VIX                     (what a desk does)
    realized    delta at the vol the path actually had      (not tradable - it is
                                                            known only at expiry -
                                                            but it is the textbook
                                                            anchor)

and, since the first run, two more that each take one thing away from the
re-marked arm:

    strike-vol  delta at the STRIKE's own implied, not the index's: the day's
                VIX is the at-the-money level and the strike's vol is read off
                the SPY surface ssvi_study.py fitted, at the strike's
                log-moneyness and the time left. A call that has gone out of
                the money in a sell-off sits on the skew, not at the money.
    capped      delta at the day's VIX, but never above the vol it was sold
                at. The re-marked arm lost its tail by marking UP in a spike;
                this arm may mark down and may not mark up, which is the one
                change that tests whether the up-marks were the whole loss.

The price sold at is the same in all three, so the expected gap between
implied and realized is the same; what changes is the path the hedge takes
to collect it.

Predictions, written before the run:

  1. The mean P&L does not move: all three within 0.03 per $100 of each
     other. The hedge vol sets how the gap is collected, not the gap.
  2. Hedging at the realized vol has the SMALLEST standard deviation of
     P&L, and by a lot - under two thirds of the sold-at hedge's. This is
     the Ahmad-Wilmott result: with the true vol in the delta, the total
     P&L is the difference between the two Black-Scholes prices at
     inception, path by path, and only the daily-hedging error is left.
     Hedging at implied makes the P&L a gamma-weighted sum along the path,
     which is what the shipped study's 0.64 standard deviation is.
  3. Re-marking at the day's VIX sits between: VIX moves toward realized
     when realized moves, so the re-marked delta is nearer the true one
     than the inception delta is. The worst month (March 2020) improves.
  4. The ranking of months barely changes: the re-marked and sold-at P&Ls
     correlate above 0.95.

Scored on the first run: 1, 2 and 4 held, 3 failed - the realized-vol hedge
has the LARGEST std and re-marking makes the worst month 60% worse. The
second run adds the two arms above. Predictions, again written first:

  5. The strike-vol arm sits between sold-at and re-marked on both the std
     and the worst month. In a sell-off the strike is above the spot, a
     negative-rho surface puts a strike above the money below the
     at-the-money vol, so the strike's delta is pulled toward a half by
     less than VIX pulls it, and it buys less on the way down.
  6. The capped arm's worst month is the sold-at arm's to the cent: VIX
     never fell below the 33% the March 2020 call was sold at, so the cap
     binds on every day of that trade and the two hedges are the same
     hedge. Same for October 2008.
  7. The capped arm's std is below the sold-at arm's. What the cap keeps
     of re-marking is the down-marks, which lower the delta of an
     out-of-the-money call when the market calms; those are the months
     the re-marked arm won, and it keeps them without the spike months.
  8. Both new arms earn the sold-at mean within 0.03 - the vol in the
     delta still does not set the gap - and the strike-vol arm correlates
     with the re-marked arm above 0.98, because it is that arm with a
     skew adjustment of a few points.
"""

from __future__ import annotations

import json

import pandas as pd

import numpy as np

from sleeves.volcarry.black_scholes import price
from sleeves.volcarry.hedging import delta_hedge
from sleeves.volcarry.ssvi import SSVIParams
from sleeves.volcarry.hedging_study import (DAILY_CACHE, HEDGE_DAYS, OPTION_SPREAD_VIX_ANCHOR,
                                            OPTION_SPREAD_VOL_PTS, RATE, T_YEARS,
                                            half_spread_vol_pts, load, months, summarize)

OUT = "results/remark_study.json"
SHIPPED_MONTHLY = "results/volcarry_monthly_sp0p1_vix17p5.csv"
ARMS = ("sold-at", "re-marked", "realized")
NEW_ARMS = ("strike-vol", "capped")
ALL_ARMS = ARMS + NEW_ARMS

# The skew. Three shared SSVI parameters from ssvi_study.py's joint fit to
# 802 SPY quotes across six expiries (rho is the tilt, eta and gamma the
# width and how it decays as variance grows). Only the shape is used: the
# level of each day's surface is set by that day's VIX, so this is "the
# 2026 chain's skew, scaled by the surface's own power law to the day's
# variance". There is no historical surface on the tape, and a fitted shape
# is a better stand-in than a flat slope guessed from a textbook, because
# the width of a one-month smile in log-moneyness is not a constant - it
# is steep when variance is low and flattens as variance rises, which is
# exactly the regime the hedge is being asked about. The thetas and ts are
# placeholders: theta_at() is never called, every theta is passed directly.
SPY_SSVI = SSVIParams(rho=-0.4702, eta=0.7587, gamma=0.5843, thetas=(0.0,), ts=(0.0,))


def strike_vol(spot, strike, time_left, atm_vol, surface: SSVIParams = SPY_SSVI):
    """The implied vol of one strike, given the at-the-money vol that day.

    atm_vol sets the surface's level (theta = atm_vol^2 * T, the total
    variance at k = 0, where the SSVI braces collapse to 2 and give the
    at-the-money vol back exactly), and the surface's shared shape sets how
    far the strike's vol sits from it at k = log(K / S). All arrays
    broadcast, so one call does a whole month of hedge days.
    """
    spot = np.asarray(spot, dtype=float)
    time_left = np.maximum(np.asarray(time_left, dtype=float), 1e-6)
    atm_vol = np.asarray(atm_vol, dtype=float)
    k = np.log(strike / spot)
    theta = atm_vol ** 2 * time_left
    p = surface.phi(theta)
    root = np.sqrt((p * k + surface.rho) ** 2 + 1.0 - surface.rho ** 2)
    w = theta / 2.0 * (1.0 + surface.rho * p * k + root)
    return np.sqrt(np.maximum(w, 0.0) / time_left)


def trade_five_ways(window: pd.DataFrame, option_type: str = "call",
                    option_spread_vol_pts: float = OPTION_SPREAD_VOL_PTS,
                    option_spread_vix_anchor: float | None = OPTION_SPREAD_VIX_ANCHOR) -> dict:
    """One month, hedged at the sold-at vol, at the daily VIX, at the path's
    own realized vol, at the strike's own vol off the skew, and at the daily
    VIX capped at the sold-at vol. Same strike, same premium, same costs (none)."""
    path = window["spy"].to_numpy(dtype=float)
    vix = window["vix"].to_numpy(dtype=float) / 100.0
    sigma = float(vix[0])
    strike = path[0]
    charged = half_spread_vol_pts(float(window["vix"].iloc[0]), option_spread_vol_pts,
                                  option_spread_vix_anchor)
    sold_at = max(sigma - charged / 100.0, 1e-6)
    scale = 100.0 / path[0]
    n_steps = len(path) - 1

    def run(hedge_vols):
        return delta_hedge(path, strike, T_YEARS, RATE, sold_at, option_type=option_type,
                           sigma_hedge=hedge_vols)

    sold = run(sigma)
    marked = run(vix[:n_steps])
    realized = run(sold["realized_vol"])
    time_left = T_YEARS - np.arange(n_steps) * T_YEARS / n_steps
    own = strike_vol(path[:n_steps], strike, time_left, vix[:n_steps])
    off_skew = run(own)
    capped = run(np.minimum(vix[:n_steps], sigma))
    return {"date": window.index[0], "implied": sigma, "realized_vol": sold["realized_vol"],
            "vix_mean": float(vix[:n_steps].mean()), "vix_max": float(vix[:n_steps].max()),
            "sold-at": sold["pnl"] * scale, "re-marked": marked["pnl"] * scale,
            "realized": realized["pnl"] * scale,
            "strike-vol": off_skew["pnl"] * scale, "capped": capped["pnl"] * scale,
            # where the skew put the strike relative to the money, in vol
            # points, averaged over the hedge days and at the worst of them
            "skew_mean_pts": float((own - vix[:n_steps]).mean() * 100.0),
            "skew_min_pts": float((own - vix[:n_steps]).min() * 100.0),
            "skew_max_pts": float((own - vix[:n_steps]).max() * 100.0),
            "days_cap_binds": int((vix[:n_steps] > sigma).sum()),
            # the textbook number: the two prices at inception, which is what
            # the realized-vol hedge collects in continuous time
            "price_gap": (price(option_type, path[0], strike, T_YEARS, RATE, sold_at)
                          - price(option_type, path[0], strike, T_YEARS, RATE,
                                  sold["realized_vol"])) * scale,
            "shares_sold_at": sold["shares_traded"],
            "shares_re_marked": marked["shares_traded"]}


def agreement_with_shipped(frame: pd.DataFrame, shipped_path: str = SHIPPED_MONTHLY) -> dict:
    """The sold-at arm against the main study's cached months, so the tape
    this ran on is shown to be the tape the rest of the sleeve used."""
    shipped = pd.read_csv(shipped_path, index_col=0, parse_dates=True)["ret"] * 100.0
    mine = frame["sold-at"].copy()
    mine.index = mine.index.to_period("M").to_timestamp("M")
    both = pd.concat([mine.rename("new"), shipped.rename("old")], axis=1).dropna()
    diff = (both["new"] - both["old"]).abs()
    return {"months_compared": int(len(both)), "max_abs_diff": float(diff.max()),
            "mean_abs_diff": float(diff.mean()),
            "correlation": float(both["new"].corr(both["old"]))}


def verdicts(stats: dict, frame: pd.DataFrame) -> list[dict]:
    means = {a: stats[a]["mean"] for a in ARMS}
    stds = {a: stats[a]["std"] for a in ARMS}
    worst = {a: stats[a]["worst"] for a in ARMS}
    corr = float(frame["sold-at"].corr(frame["re-marked"]))
    spread = max(means.values()) - min(means.values())
    return [
        {"claim": "the mean P&L is the same in all three arms, within 0.03",
         "measured": ", ".join(f"{a} {m:+.3f}" for a, m in means.items()),
         "held": bool(spread < 0.03)},
        {"claim": "hedging at realized vol has the smallest std, under two thirds of sold-at",
         "measured": ", ".join(f"{a} {v:.3f}" for a, v in stds.items()),
         "held": bool(stds["realized"] == min(stds.values()) and stds["realized"] < stds["sold-at"] * 2 / 3)},
        {"claim": "re-marking sits between on std, and the worst month improves",
         "measured": f"std re-marked {stds['re-marked']:.3f}; worst sold-at {worst['sold-at']:+.3f}, "
                     f"re-marked {worst['re-marked']:+.3f}",
         "held": bool(stds["realized"] <= stds["re-marked"] <= stds["sold-at"]
                      and worst["re-marked"] > worst["sold-at"])},
        {"claim": "re-marked and sold-at P&Ls correlate above 0.95",
         "measured": f"{corr:.3f}", "held": bool(corr > 0.95)},
    ]


def skew_verdicts(stats: dict, frame: pd.DataFrame) -> list[dict]:
    stds = {a: stats[a]["std"] for a in ALL_ARMS}
    worst = {a: stats[a]["worst"] for a in ALL_ARMS}
    means = {a: stats[a]["mean"] for a in ALL_ARMS}
    oct08 = frame.loc["2008-10"].iloc[0] if "2008-10" in frame.index.strftime("%Y-%m") else None
    mar20 = frame.loc["2020-03"].iloc[0] if "2020-03" in frame.index.strftime("%Y-%m") else None
    same_tail = (mar20 is not None and oct08 is not None
                 and abs(mar20["capped"] - mar20["sold-at"]) < 0.01
                 and abs(oct08["capped"] - oct08["sold-at"]) < 0.01)
    corr = float(frame["strike-vol"].corr(frame["re-marked"]))
    return [
        {"claim": "the strike-vol arm sits between sold-at and re-marked on std and on the worst month",
         "measured": f"std {stds['sold-at']:.3f} / {stds['strike-vol']:.3f} / {stds['re-marked']:.3f}; "
                     f"worst {worst['sold-at']:+.3f} / {worst['strike-vol']:+.3f} / {worst['re-marked']:+.3f}",
         "held": bool(stds["sold-at"] <= stds["strike-vol"] <= stds["re-marked"]
                      and worst["re-marked"] <= worst["strike-vol"] <= worst["sold-at"])},
        {"claim": "the capped arm is the sold-at arm to the cent in March 2020 and October 2008",
         "measured": ("no such months" if mar20 is None or oct08 is None else
                      f"2020-03 {mar20['sold-at']:+.3f} vs {mar20['capped']:+.3f}, cap bound "
                      f"{int(mar20['days_cap_binds'])} of {HEDGE_DAYS} days; 2008-10 {oct08['sold-at']:+.3f} "
                      f"vs {oct08['capped']:+.3f}, {int(oct08['days_cap_binds'])} days"),
         "held": bool(same_tail)},
        {"claim": "the capped arm's std is below the sold-at arm's",
         "measured": f"capped {stds['capped']:.3f}, sold-at {stds['sold-at']:.3f}",
         "held": bool(stds["capped"] < stds["sold-at"])},
        {"claim": "both new arms earn the sold-at mean within 0.03, and strike-vol correlates with re-marked above 0.98",
         "measured": f"means sold-at {means['sold-at']:+.3f}, strike-vol {means['strike-vol']:+.3f}, capped "
                     f"{means['capped']:+.3f}; correlation {corr:.3f}",
         "held": bool(abs(means["strike-vol"] - means["sold-at"]) < 0.03
                      and abs(means["capped"] - means["sold-at"]) < 0.03 and corr > 0.98)},
    ]


def main() -> None:
    data = load(end="2026-12-31", cache=DAILY_CACHE)
    windows = months(data)
    frame = pd.DataFrame([trade_five_ways(w) for w in windows]).set_index("date")
    stats = {a: summarize(frame.rename(columns={a: "pnl"})) for a in ALL_ARMS}
    print(f"Short one at-the-money SPY call a month, sold at VIX less the spread, hedged daily "
          f"five ways.\n{len(frame)} trades, {frame.index[0].date()} to {frame.index[-1].date()}, "
          "per $100 of underlying.\n")
    print(f"{'':<24} " + " ".join(f"{a:>10}" for a in ALL_ARMS))
    for key, label, fmt in (("mean", "average P&L", "{:>10.3f}"), ("std", "std dev of P&L", "{:>10.3f}"),
                            ("sharpe", "annualized Sharpe", "{:>10.2f}"), ("win_rate", "months profitable", "{:>10.1%}"),
                            ("worst", "worst month", "{:>10.3f}"), ("best", "best month", "{:>10.3f}")):
        print(f"{label:<24} " + " ".join(fmt.format(stats[a][key]) for a in ALL_ARMS))
    gap = frame["price_gap"]
    print(f"\n{'price gap at inception':<24} {gap.mean():>10.3f}   (mean; std {gap.std(ddof=1):.3f}) - what the "
          "realized-vol hedge collects in continuous time")
    print(f"{'realized minus the gap':<24} {(frame['realized'] - gap).mean():>10.3f}   (mean; std "
          f"{(frame['realized'] - gap).std(ddof=1):.3f}) - the daily-hedging error alone")
    print(f"{'shares traded, sold-at':<24} {frame['shares_sold_at'].mean():>10.2f}   per option on one share, per month")
    print(f"{'shares traded, re-marked':<24} {frame['shares_re_marked'].mean():>10.2f}")

    worst = frame["sold-at"].idxmin()
    print(f"\nWorst sold-at month, {worst:%Y-%m}: " + ", ".join(f"{a} {frame.loc[worst, a]:+.3f}" for a in ALL_ARMS)
          + f"; implied {frame.loc[worst, 'implied']:.1%}, realized {frame.loc[worst, 'realized_vol']:.1%}, VIX ran to "
          f"{frame.loc[worst, 'vix_max']:.0%}")
    big = frame[(frame["vix_max"] - frame["implied"]) > 0.10]
    print(f"Months the VIX rose more than ten points inside the trade: {len(big)}; there the re-marked "
          f"hedge beat the sold-at hedge in {int((big['re-marked'] > big['sold-at']).sum())} and averaged "
          f"{big['re-marked'].mean():+.3f} against {big['sold-at'].mean():+.3f}")
    print(f"  in those months the strike-vol hedge averaged {big['strike-vol'].mean():+.3f} and beat sold-at in "
          f"{int((big['strike-vol'] > big['sold-at']).sum())}; the capped hedge averaged {big['capped'].mean():+.3f} "
          f"and beat it in {int((big['capped'] > big['sold-at']).sum())}")

    # where the skew put the strike: the whole point of the strike-vol arm
    # is that the strike's vol is not the index's, so say by how much
    print(f"\nStrike's vol minus the day's VIX, in vol points: mean over all hedge days "
          f"{frame['skew_mean_pts'].mean():+.2f}, month-means from {frame['skew_mean_pts'].min():+.2f} to "
          f"{frame['skew_mean_pts'].max():+.2f}; in {worst:%Y-%m} the mean was {frame.loc[worst, 'skew_mean_pts']:+.2f} "
          f"(day range {frame.loc[worst, 'skew_min_pts']:+.2f} to {frame.loc[worst, 'skew_max_pts']:+.2f})")
    above = int((frame["skew_mean_pts"] > 0).sum())
    print(f"  months where the strike sat ABOVE the money on average: {above} of {len(frame)}")
    binds = frame["days_cap_binds"]
    print(f"Cap on the re-mark bound on {binds.sum() / (len(frame) * (HEDGE_DAYS)):.0%} of hedge days; "
          f"{int((binds == HEDGE_DAYS).sum())} months on every day, {int((binds == 0).sum())} months on none")
    differs = frame[(frame["capped"] - frame["sold-at"]).abs() > 1e-6]
    print(f"  the capped arm differs from sold-at in {len(differs)} months, where it averaged "
          f"{differs['capped'].mean():+.3f} against {differs['sold-at'].mean():+.3f}")

    agree = agreement_with_shipped(frame)
    print(f"\nTape check: the sold-at arm against the main study's cached months, {agree['months_compared']} "
          f"in common: max abs diff {agree['max_abs_diff']:.4f}, correlation {agree['correlation']:.4f}")

    scored = verdicts(stats, frame)
    print("\nPredictions, scored (first run, three arms):")
    for v in scored:
        print(f"  [{'held' if v['held'] else 'FAILED'}] {v['claim']}\n         measured: {v['measured']}")
    scored_skew = skew_verdicts(stats, frame)
    print("\nPredictions, scored (second run, the strike-vol and capped arms):")
    for v in scored_skew:
        print(f"  [{'held' if v['held'] else 'FAILED'}] {v['claim']}\n         measured: {v['measured']}")

    with open(OUT, "w") as fh:
        json.dump({"n": int(len(frame)), "from": str(frame.index[0].date()), "to": str(frame.index[-1].date()),
                   "arms": {a: {k: float(v) for k, v in stats[a].items()} for a in ALL_ARMS},
                   "price_gap_mean": float(gap.mean()), "price_gap_std": float(gap.std(ddof=1)),
                   "hedging_error_std": float((frame["realized"] - gap).std(ddof=1)),
                   "shares_traded": {"sold-at": float(frame["shares_sold_at"].mean()),
                                     "re-marked": float(frame["shares_re_marked"].mean())},
                   "worst_sold_at_month": f"{worst:%Y-%m}",
                   "worst_month": {a: float(frame.loc[worst, a]) for a in ALL_ARMS},
                   "vix_spike_months": {"n": int(len(big)),
                                        "re_marked_wins": int((big["re-marked"] > big["sold-at"]).sum()),
                                        "strike_vol_wins": int((big["strike-vol"] > big["sold-at"]).sum()),
                                        "capped_wins": int((big["capped"] > big["sold-at"]).sum()),
                                        "mean": {a: float(big[a].mean()) for a in ALL_ARMS}},
                   "correlation_sold_at_re_marked": float(frame["sold-at"].corr(frame["re-marked"])),
                   "correlation_strike_vol_re_marked": float(frame["strike-vol"].corr(frame["re-marked"])),
                   "skew_pts": {"mean": float(frame["skew_mean_pts"].mean()),
                                "months_above_money": above,
                                "worst_month_mean": float(frame.loc[worst, "skew_mean_pts"]),
                                "worst_month_min": float(frame.loc[worst, "skew_min_pts"]),
                                "worst_month_max": float(frame.loc[worst, "skew_max_pts"])},
                   "cap": {"share_of_days_bound": float(binds.sum() / (len(frame) * HEDGE_DAYS)),
                           "months_bound_every_day": int((binds == HEDGE_DAYS).sum()),
                           "months_never_bound": int((binds == 0).sum()),
                           "months_differing_from_sold_at": int(len(differs))},
                   "tape_agreement": agree, "verdicts": scored, "skew_verdicts": scored_skew,
                   "monthly": {a: [round(float(x), 5) for x in frame[a]] for a in ALL_ARMS},
                   "months": [f"{d:%Y-%m}" for d in frame.index]}, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
