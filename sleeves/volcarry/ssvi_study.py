"""Six sound smiles are not a sound surface.

Usage:  python ssvi_study.py [TICKER]

svi.py fits each expiry separately under the constraint that its own implied
density stays positive. The README's version of what that bought was careful
about the strike direction and silent about the other one, and the other one
has a name: total variance must not fall as maturity rises at fixed
moneyness, or a calendar spread has a negative price. Nothing in a
per-expiry fit knows about the expiry next to it, so nothing in it prevents
that.

Table 1 asks whether the concern is real or theoretical by taking the six
independent SVI slices the project already fits and checking them against each
other. Table 2 fits the same quotes as one SSVI surface, where both conditions
are closed-form statements about three shared parameters and are enforced
during the fit. Table 3 is the price of that, per expiry, in volatility
points, because 9 parameters cannot follow 802 quotes the way 30 can.

One thing worth reporting because it came out differently from expected. Of
the two closed-form calendar conditions, the exotic-looking one - a bound on
d(theta*phi)/dtheta - turns out to be free for the power-law phi. Working the
derivative through, the ratio it bounds collapses to (1-gamma)/(1+theta),
which is under 1 for any admissible gamma, while the bound never falls below
1.04. It cannot bind. So the entire calendar guarantee comes from the plain
half: theta non-decreasing in maturity. That is the half a per-expiry fit
cannot even express, because each slice solves for its own level knowing
nothing about the slice next to it, and it is what the joint fit has to force.

The conclusion is the same shape as the one in the README's SVI section and
it is worth stating in the same breath as the result: the constrained surface
is not a better repricer, it is a surface you can differentiate. Consistency
across expiries is what lets you price a calendar spread, a variance swap or
anything else that touches two maturities at once, and the per-strike residual
is what a desk carries on top of it.
"""

import sys

import numpy as np
import pandas as pd

from sleeves.volcarry import ssvi
from sleeves.volcarry import svi
from sleeves.volcarry import vol_surface as vs


def per_expiry_fits(surface: pd.DataFrame) -> list[dict]:
    """The existing per-slice fit, one call per expiry, unchanged."""
    out = []
    for expiry, smile in surface.groupby("expiry", sort=False):
        fit = svi.fit_smile(smile)
        fit["expiry"] = expiry
        fit["n"] = int(smile["iv"].notna().sum())
        out.append(fit)
    return sorted(out, key=lambda f: f["T"])


def slice_calendar_violations(fits: list[dict], k_lo: float = -0.4,
                              k_hi: float = 0.4, n: int = 81) -> dict:
    """Do the independently fitted slices cross each other?

    Reads total variance straight off each fitted slice on a common
    log-moneyness grid and walks up the expiries. A violation is a grid point
    where a longer maturity prices LESS total variance than a shorter one,
    which is a calendar spread you would be paid to own.

    The grid is deliberately narrow, plus or minus 0.4 in log-moneyness. Wider
    than that and the short expiries are being extrapolated far past any quote
    they were fitted to, so a crossing out there says more about extrapolation
    than about the surface.
    """
    grid = np.linspace(k_lo, k_hi, n)
    curves = [p["params"].total_variance(grid) for p in fits]
    rows, worst = [], 0.0
    for (a, wa), (b, wb) in zip(zip(fits, curves), zip(fits[1:], curves[1:])):
        bad = wb < wa - 1e-12
        if bad.any():
            gap = float(np.max((wa - wb)[bad]))
            worst = max(worst, gap)
            rows.append({"from": a["expiry"], "to": b["expiry"],
                         "points": int(bad.sum()), "worst_w": gap,
                         "at_k": float(grid[bad][np.argmax((wa - wb)[bad])])})
    return {"pairs": rows, "total": sum(r["points"] for r in rows),
            "grid": n * (len(fits) - 1), "worst_w": worst}


def vol_point_error(params: ssvi.SSVIParams, sl: dict, theta: float) -> dict:
    """SSVI's repricing error on one slice, in volatility points."""
    model_w = params.total_variance(sl["k"], theta)
    model_iv = np.sqrt(np.maximum(model_w, 0.0) / sl["T"])
    err = (model_iv - sl["iv"]) * 100
    return {"rmse": float(np.sqrt(np.mean(err ** 2))),
            "worst": float(np.max(np.abs(err)))}


def main() -> None:
    ticker = sys.argv[1] if len(sys.argv) > 1 else "SPY"
    print(f"Building the surface for {ticker}...")
    surface = vs.build_surface(ticker, max_expiries=6, verbose=True)
    if surface.empty:
        print("no usable quotes")
        return

    fits = per_expiry_fits(surface)
    slices = ssvi.slices_from_surface(surface)

    print("\n\n1. Six independently fitted slices, checked against each other\n")
    print(f"{'expiry':<12} {'days':>5} {'quotes':>7} {'own density':>12} "
          f"{'vol rmse':>9} {'atm vol':>8}")
    for f in fits:
        atm = float(np.sqrt(f["params"].total_variance(0.0) / f["T"]))
        print(f"{str(f['expiry']):<12} {f['T'] * 365:>5.0f} {f['n']:>7} "
              f"{'clean' if f['butterfly_free'] else 'NEGATIVE':>12} "
              f"{f['rmse_vol_points'] * 100:>8.2f}p {atm:>7.1%}")

    cal = slice_calendar_violations(fits)
    print(f"\nEvery slice has a positive density on its own. Between slices, "
          f"{cal['total']} of\n{cal['grid']} grid points price a calendar "
          f"spread at a negative value.")
    for r in cal["pairs"]:
        print(f"    {r['from']} -> {r['to']}: {r['points']} points, worst "
              f"total variance drop {r['worst_w']:.5f} at k={r['at_k']:+.2f}")
    if not cal["pairs"]:
        print("    (none on this day's quotes - the condition is not "
              "guaranteed, only unviolated)")

    print("\n\n2. The same quotes as one SSVI surface\n")
    fit = ssvi.fit_ssvi(slices)
    p = fit["params"]
    print(f"  rho    {p.rho:>+8.4f}   one skew tilt for the whole surface; "
          f"negative is the equity sign")
    print(f"  eta    {p.eta:>8.4f}   smile width scale")
    print(f"  gamma  {p.gamma:>8.4f}   how the width decays as variance grows")
    print(f"\n  {fit['n_params']} parameters against "
          f"{len(fits) * 5} for six independent slices, "
          f"{fit['n_quotes']} quotes.")
    print(f"\n{'expiry':<12} {'days':>5} {'theta':>9} {'atm vol':>8}")
    for sl, theta in zip(slices, p.thetas):
        print(f"{str(sl['expiry']):<12} {sl['T'] * 365:>5.0f} {theta:>9.5f} "
              f"{np.sqrt(theta / sl['T']):>7.1%}")

    print("\n  conditions, from the parameters:")
    print(f"    butterfly (theta*phi*(1+|rho|) < 4)        "
          f"{'satisfied' if fit['butterfly_ok'] else 'FAILED'}")
    print(f"    calendar  (d(theta*phi)/dtheta in range)   "
          f"{'satisfied' if fit['calendar_ok'] else 'FAILED'}"
          f"   (free for this phi - see the header)")
    print(f"    theta non-decreasing in maturity           "
          f"{'satisfied' if fit['theta_monotone'] else 'FAILED'}"
          f"   (this is the one doing the work)")
    print("\n  verified independently, on the fitted surface rather than on "
          "the parameters:")
    print(f"    negative-density grid points   "
          f"{ssvi.density_violations(p)}   (svi.durrleman_g, 400 strikes x "
          f"{len(slices)} expiries)")
    print(f"    negative calendar spreads      "
          f"{ssvi.calendar_violations(p)}   (81 strikes x "
          f"{len(slices) - 1} adjacent pairs)")

    print("\n\n3. What the constraint costs, in volatility points\n")
    print(f"{'expiry':<12} {'days':>5} {'per-slice SVI':>15} "
          f"{'SSVI surface':>14} {'worst SSVI':>12}")
    for f, sl, theta in zip(fits, slices, p.thetas):
        e = vol_point_error(p, sl, theta)
        print(f"{str(f['expiry']):<12} {f['T'] * 365:>5.0f} "
              f"{f['rmse_vol_points'] * 100:>14.2f}p {e['rmse']:>13.2f}p "
              f"{e['worst']:>11.2f}p")

    free = ssvi.fit_ssvi(slices, enforce=False)
    fp = free["params"]
    ok = ssvi.admissible(fp.rho, fp.eta, fp.gamma)
    worse = (fit["rmse_w"] / free["rmse_w"] - 1.0) * 100 if free["rmse_w"] else 0.0
    print(f"\nDropping the arbitrage conditions from the search fits better, "
          f"as it has to: total\nvariance RMSE {free['rmse_w']:.5f} against "
          f"{fit['rmse_w']:.5f}, so enforcing them costs "
          f"{worse:.1f}%\nof fit. The unconstrained surface is "
          f"{'admissible anyway on this day' if ok else 'NOT admissible'}"
          f"{'' if ok else ' - which is the thing enforcement exists to prevent'}"
          f", and it\nwould be arbitrageable "
          f"{'nowhere' if ok else 'at ' + str(ssvi.density_violations(fp) + ssvi.calendar_violations(fp)) + ' of the grid points checked above'}.")

    print("\nThe honest summary: 9 parameters reprice a liquid SPY chain worse "
          "than 30 do, and\nthat is not a defect being apologised for. A "
          "per-expiry fit has no opinion about\nwhat a calendar spread is "
          "worth, because it never saw two expiries at once. This\nsurface "
          "does, and cannot price one at a negative value. Desks run the "
          "constrained\nsurface for the shape and carry a per-strike "
          "correction on top of it.")


if __name__ == "__main__":
    main()
