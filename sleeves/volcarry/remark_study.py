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
"""

from __future__ import annotations

import json

import pandas as pd

from sleeves.volcarry.black_scholes import price
from sleeves.volcarry.hedging import delta_hedge
from sleeves.volcarry.hedging_study import (DAILY_CACHE, OPTION_SPREAD_VIX_ANCHOR,
                                            OPTION_SPREAD_VOL_PTS, RATE, T_YEARS,
                                            half_spread_vol_pts, load, months, summarize)

OUT = "results/remark_study.json"
SHIPPED_MONTHLY = "results/volcarry_monthly_sp0p1_vix17p5.csv"
ARMS = ("sold-at", "re-marked", "realized")


def trade_three_ways(window: pd.DataFrame, option_type: str = "call",
                     option_spread_vol_pts: float = OPTION_SPREAD_VOL_PTS,
                     option_spread_vix_anchor: float | None = OPTION_SPREAD_VIX_ANCHOR) -> dict:
    """One month, hedged at the sold-at vol, at the daily VIX, and at the
    path's own realized vol. Same strike, same premium, same costs (none)."""
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
    return {"date": window.index[0], "implied": sigma, "realized_vol": sold["realized_vol"],
            "vix_mean": float(vix[:n_steps].mean()), "vix_max": float(vix[:n_steps].max()),
            "sold-at": sold["pnl"] * scale, "re-marked": marked["pnl"] * scale,
            "realized": realized["pnl"] * scale,
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


def main() -> None:
    data = load(end="2026-12-31", cache=DAILY_CACHE)
    windows = months(data)
    frame = pd.DataFrame([trade_three_ways(w) for w in windows]).set_index("date")
    stats = {a: summarize(frame.rename(columns={a: "pnl"})) for a in ARMS}
    print(f"Short one at-the-money SPY call a month, sold at VIX less the spread, hedged daily "
          f"three ways.\n{len(frame)} trades, {frame.index[0].date()} to {frame.index[-1].date()}, "
          "per $100 of underlying.\n")
    print(f"{'':<24} " + " ".join(f"{a:>10}" for a in ARMS))
    for key, label, fmt in (("mean", "average P&L", "{:>10.3f}"), ("std", "std dev of P&L", "{:>10.3f}"),
                            ("sharpe", "annualized Sharpe", "{:>10.2f}"), ("win_rate", "months profitable", "{:>10.1%}"),
                            ("worst", "worst month", "{:>10.3f}"), ("best", "best month", "{:>10.3f}")):
        print(f"{label:<24} " + " ".join(fmt.format(stats[a][key]) for a in ARMS))
    gap = frame["price_gap"]
    print(f"\n{'price gap at inception':<24} {gap.mean():>10.3f}   (mean; std {gap.std(ddof=1):.3f}) - what the "
          "realized-vol hedge collects in continuous time")
    print(f"{'realized minus the gap':<24} {(frame['realized'] - gap).mean():>10.3f}   (mean; std "
          f"{(frame['realized'] - gap).std(ddof=1):.3f}) - the daily-hedging error alone")
    print(f"{'shares traded, sold-at':<24} {frame['shares_sold_at'].mean():>10.2f}   per option on one share, per month")
    print(f"{'shares traded, re-marked':<24} {frame['shares_re_marked'].mean():>10.2f}")

    worst = frame["sold-at"].idxmin()
    print(f"\nWorst sold-at month, {worst:%Y-%m}: sold-at {frame.loc[worst, 'sold-at']:+.3f}, re-marked "
          f"{frame.loc[worst, 're-marked']:+.3f}, realized {frame.loc[worst, 'realized']:+.3f}; implied "
          f"{frame.loc[worst, 'implied']:.1%}, realized {frame.loc[worst, 'realized_vol']:.1%}, VIX ran to "
          f"{frame.loc[worst, 'vix_max']:.0%}")
    big = frame[(frame["vix_max"] - frame["implied"]) > 0.10]
    print(f"Months the VIX rose more than ten points inside the trade: {len(big)}; there the re-marked "
          f"hedge beat the sold-at hedge in {int((big['re-marked'] > big['sold-at']).sum())} and averaged "
          f"{big['re-marked'].mean():+.3f} against {big['sold-at'].mean():+.3f}")

    agree = agreement_with_shipped(frame)
    print(f"\nTape check: the sold-at arm against the main study's cached months, {agree['months_compared']} "
          f"in common: max abs diff {agree['max_abs_diff']:.4f}, correlation {agree['correlation']:.4f}")

    scored = verdicts(stats, frame)
    print("\nPredictions, scored:")
    for v in scored:
        print(f"  [{'held' if v['held'] else 'FAILED'}] {v['claim']}\n         measured: {v['measured']}")

    with open(OUT, "w") as fh:
        json.dump({"n": int(len(frame)), "from": str(frame.index[0].date()), "to": str(frame.index[-1].date()),
                   "arms": {a: {k: float(v) for k, v in stats[a].items()} for a in ARMS},
                   "price_gap_mean": float(gap.mean()), "price_gap_std": float(gap.std(ddof=1)),
                   "hedging_error_std": float((frame["realized"] - gap).std(ddof=1)),
                   "shares_traded": {"sold-at": float(frame["shares_sold_at"].mean()),
                                     "re-marked": float(frame["shares_re_marked"].mean())},
                   "worst_sold_at_month": f"{worst:%Y-%m}",
                   "worst_month": {a: float(frame.loc[worst, a]) for a in ARMS},
                   "vix_spike_months": {"n": int(len(big)),
                                        "re_marked_wins": int((big["re-marked"] > big["sold-at"]).sum()),
                                        "mean": {a: float(big[a].mean()) for a in ARMS}},
                   "correlation_sold_at_re_marked": float(frame["sold-at"].corr(frame["re-marked"])),
                   "tape_agreement": agree, "verdicts": scored,
                   "monthly": {a: [round(float(x), 5) for x in frame[a]] for a in ARMS},
                   "months": [f"{d:%Y-%m}" for d in frame.index]}, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
