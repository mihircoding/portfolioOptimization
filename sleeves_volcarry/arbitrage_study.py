"""Check a live option chain for arbitrage across strikes, and see whether
interpolating the smile invents any.

Usage:  python arbitrage_study.py [TICKER ...]     (default: SPY)

Two questions, one run.

1. Do the quotes themselves violate the no-arbitrage conditions on the call
   price - monotonicity, the slope bound, convexity? Raw violation counts on
   a real chain are always non-zero and always misread. The number that
   matters is how many survive paying the bid-ask spread on every leg, and
   that number is small.

2. A surface is quoted at a few dozen strikes and used at any strike, so
   something interpolates. vol_surface.atm_vol() interpolates total variance
   and remarks that interpolating volatility instead can break those same
   conditions between the quoted points. This prices both interpolations on
   a fine grid and counts what comes out negative. The finding is that the
   choice barely matters next to the noise already in the quotes: linear
   interpolation of either quantity produces negative densities between the
   strikes, because the quoted points are not convex to begin with.
"""

import sys

import pandas as pd

import arbitrage as arb
import svi
import vol_surface as vs


def report(ticker: str) -> None:
    print(f"\n{'=' * 78}\n{ticker}\n{'=' * 78}")
    surface = vs.build_surface(ticker, max_expiries=6, verbose=True)
    if surface.empty:
        print("  no usable quotes")
        return

    table = arb.check_surface(surface)
    print(f"\n{'expiry':<12} {'days':>5} {'strikes':>8} {'verticals':>18} "
          f"{'butterflies':>18} {'worst':>9} {'density':>8}")
    print(f"{'':<12} {'':>5} {'':>8} {'raw':>9} {'tradable':>8} "
          f"{'raw':>9} {'tradable':>8} {'$':>9} {'mass':>8}")
    for _, r in table.iterrows():
        print(f"{r['expiry']:<12} {r['T'] * 365:>5.0f} {r['n_strikes']:>8} "
              f"{r['vertical_violations']:>9} {r['vertical_tradable']:>8} "
              f"{r['butterfly_violations']:>9} {r['butterfly_tradable']:>8} "
              f"{r['worst_butterfly']:>9.2f} {r['density_mass']:>8.3f}")

    raw = int(table["butterfly_violations"].sum() + table["vertical_violations"].sum())
    live = int(table["butterfly_tradable"].sum() + table["vertical_tradable"].sum())
    print(f"\n{raw} conditions violated by the mid prices; {live} by more than the")
    print("bid-ask spread of the legs you would have to trade. The gap between those")
    print("two numbers is what a screen built on mids and no spread filter reports.")
    print("\nDensity mass is the implied distribution integrated over the quoted")
    print("strikes: near 1 when the chain covers the tails, less when it stops short.")

    print(f"\n{'expiry':<12} {'interpolated in variance':>26} "
          f"{'interpolated in vol':>22}")
    print(f"{'':<12} {'bad butterflies':>16} {'worst $':>9} "
          f"{'bad butterflies':>16} {'worst $':>9}")
    for expiry, smile in surface.groupby("expiry", sort=False):
        in_var = arb.interpolated_smile_violations(smile, in_variance=True)
        in_vol = arb.interpolated_smile_violations(smile, in_variance=False)
        print(f"{expiry:<12} {in_var['violations']:>16} {in_var['worst']:>9.3f} "
              f"{in_vol['violations']:>16} {in_vol['worst']:>9.3f}")
    print("\n200 strikes per expiry, priced from the interpolated vol. Both columns")
    print("are the same quotes; only the quantity interpolated between them differs,")
    print("and the answer is that it hardly matters. Straight lines between quoted")
    print("IVs inherit whatever non-convexity the quotes already have, and that")
    print("dominates the choice of what to interpolate.")

    svi_section(surface)


def svi_section(surface) -> None:
    """Fit each expiry with SVI and re-run the same measurement."""
    print(f"\n{'-' * 78}\nSVI fit, constrained to be butterfly-free\n{'-' * 78}")
    print(f"{'expiry':<12} {'n':>4} {'fit err':>9} {'worst':>7} {'in':>8} "
          f"{'bad butterflies':>16} {'min g':>8} {'wings':>13}")
    print(f"{'':<12} {'':>4} {'vol pts':>9} {'vol pt':>7} {'spread':>8} "
          f"{'linear':>7} {'svi':>8} {'':>8} {'left':>6} {'right':>6}")

    fitted = 0
    for expiry, smile in surface.groupby("expiry", sort=False):
        try:
            fit = svi.fit_smile(smile)
        except (ValueError, RuntimeError) as e:
            print(f"{expiry:<12} {'':>4} no fit: {e}")
            continue
        fitted += 1
        linear = arb.interpolated_smile_violations(smile, in_variance=True)
        after = svi.grid_violations(smile, fit)
        q = svi.fit_quality(smile, fit)
        print(f"{expiry:<12} {fit['n_quotes']:>4} "
              f"{fit['rmse_vol_points'] * 100:>9.2f} "
              f"{fit['max_vol_error'] * 100:>7.2f} "
              f"{q['inside_spread']:>4}/{q['n']:<3} "
              f"{linear['violations']:>7} {after['violations']:>8} "
              f"{fit['min_g']:>8.3f} "
              f"{fit['left_slope']:>6.2f} {fit['right_slope']:>6.2f}")

    if not fitted:
        return
    print("\nFit error is in volatility POINTS, so 0.30 means the fitted smile is")
    print("three tenths of a vol away from the quote - well inside a chain whose")
    print("quotes move by more than that between one refresh and the next. 'in")
    print("spread' counts the strikes the fitted price lands between the bid and the")
    print("ask, which is the only resolution the market actually has.")
    print("\nThe butterfly columns are the point. Linear interpolation invents")
    print("negative densities between the quoted strikes; the SVI fit cannot,")
    print("because the search only ever accepted parameters whose g(k) stays")
    print("non-negative, and g(k) is a formula in those parameters rather than")
    print("something measured after the fact.")
    print("\nThe wing slopes are Lee's bound: total variance cannot grow faster")
    print("than 2 per unit of log-moneyness in either wing without admitting")
    print("arbitrage. A spline has no opinion about that and will break it happily.")

    print("\nWhat the vega weighting buys\n")
    print(f"{'expiry':<12} {'weighting':<10} {'rmse':>7} {'worst miss':>11} "
          f"{'in spread':>10}")
    for expiry, smile in surface.groupby("expiry", sort=False):
        for scheme in ("vega", "equal"):
            try:
                fit = svi.fit_smile(smile, weight_by=scheme)
            except (ValueError, RuntimeError):
                continue
            q = svi.fit_quality(smile, fit)
            print(f"{expiry:<12} {scheme:<10} "
                  f"{fit['rmse_vol_points'] * 100:>7.2f} "
                  f"{q['worst_in_spreads']:>10.1f}x {q['inside_spread']:>6}/"
                  f"{q['n']:<4}")
    print("\nEqual weighting wins on RMSE and loses on the thing that matters. It")
    print("spends its accuracy on deep wing quotes, whose implied vol is mostly")
    print("rounding because the option barely responds to volatility at all, and")
    print("pays for it at the strikes anyone trades. 'worst miss' is the largest")
    print("repricing error measured in half-spreads of that strike.")


def main() -> None:
    tickers = sys.argv[1:] or ["SPY"]
    pd.set_option("display.width", 200)
    for ticker in tickers:
        report(ticker)


if __name__ == "__main__":
    main()
