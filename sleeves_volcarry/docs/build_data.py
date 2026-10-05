"""Builds docs/data.js for the GitHub Pages site.

Most of that page is computed in the browser from a JavaScript port of the
pricers, so it needs no data at all. Two sections are the exception, because
they are statements about what actually happened rather than what a formula
says:

  surface  - pulls a live SPY chain, runs it through vol_surface.py (the same
             code the Streamlit app uses) and records when it was taken.
  hedging  - runs hedging_study.py's monthly SPY trades from 2007 to 2024, plus
             a simulated check of how hedging error scales with the number of
             rebalances, using hedging.py for both.

Re-run it and the page updates:

    python docs/build_data.py            # both
    python docs/build_data.py hedging    # just one; the rest of data.js is kept

If the live chain can't be fetched (Yahoo rate limits often), the surface
already in data.js is kept rather than wiped.
"""

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import hedging  # noqa: E402
import hedging_study as study  # noqa: E402
import arbitrage as arb  # noqa: E402
import vol_surface as vsurf  # noqa: E402

TICKER = "SPY"
N_EXPIRIES = 6
OUT = Path(__file__).resolve().parent / "data.js"
SURFACE_KEYS = ("ticker", "as_of", "n_quotes", "smiles", "terms", "flat",
                "calendar_violations", "arbitrage")

# The simulated rebalancing check: a one-month at-the-money call sold at 20%
# on a stock that realizes exactly 20%, so the only thing left in the P&L is
# the error from hedging in steps. Each path has 12 points a trading day, and
# every N below divides 252, so each hedge lands on a point of the path.
SIM_PATHS = 2000
SIM_STEPS = 252
SIM_REBALANCES = (1, 2, 3, 4, 6, 7, 12, 21, 42, 84, 252)
SIM_SIGMA = 0.20


def arbitrage_data(surface):
    """The cross-strike no-arbitrage checks, plus one implied density curve.

    Runs on the surface already fetched, so the page costs one trip to Yahoo
    rather than two.
    """
    table = arb.check_surface(surface)
    rows = []
    for _, r in table.iterrows():
        smile = surface[surface["expiry"] == r["expiry"]]
        in_var = arb.interpolated_smile_violations(smile, in_variance=True)
        in_vol = arb.interpolated_smile_violations(smile, in_variance=False)
        rows.append({
            "expiry": r["expiry"], "days": round(float(r["T"]) * 365),
            "strikes": int(r["n_strikes"]),
            "vertical": int(r["vertical_violations"]),
            "vertical_tradable": int(r["vertical_tradable"]),
            "butterfly": int(r["butterfly_violations"]),
            "butterfly_tradable": int(r["butterfly_tradable"]),
            "worst": round(float(r["worst_butterfly"]), 3),
            "density_mass": round(float(r["density_mass"]), 4),
            "interp_variance": in_var["violations"],
            "interp_vol": in_vol["violations"],
            "interp_grid": in_var["n"],
        })

    # The density picture is drawn from the expiry with the most strikes -
    # the one whose distribution is actually resolved rather than sketched.
    best = max(rows, key=lambda r: r["strikes"])
    smile = surface[surface["expiry"] == best["expiry"]]
    detail = arb.check_smile(smile)
    b = detail["butterflies"]
    forward = float(smile["forward"].iloc[0])

    return {
        "rows": rows,
        "raw": int(sum(r["vertical"] + r["butterfly"] for r in rows)),
        "tradable": int(sum(r["vertical_tradable"] + r["butterfly_tradable"]
                            for r in rows)),
        "density": {
            "expiry": best["expiry"], "days": best["days"], "forward": round(forward, 2),
            "points": [[round(float(k), 2), round(float(d), 6)]
                       for k, d in zip(b["strike"], b["density"])],
            "mass": best["density_mass"],
        },
    }


def svi_data(surface):
    """Fit each expiry with SVI and re-run the butterfly count on the fit.

    Runs on the surface already fetched, same as arbitrage_data, so the page
    still costs one trip to Yahoo. The fitted curve for the best-resolved
    expiry is carried too, so the page can draw the fit through the quotes
    rather than only quoting a residual.
    """
    import svi

    rows, curve = [], None
    for expiry, smile in surface.groupby("expiry", sort=False):
        try:
            fit = svi.fit_smile(smile)
        except (ValueError, RuntimeError) as e:
            print(f"    {expiry}: no SVI fit ({e})")
            continue
        linear = arb.interpolated_smile_violations(smile, in_variance=True)
        after = svi.grid_violations(smile, fit)
        quality = svi.fit_quality(smile, fit)
        equal = svi.fit_smile(smile, weight_by="equal")
        equal_q = svi.fit_quality(smile, equal)

        rows.append({
            "expiry": expiry,
            "days": round(float(smile["T"].iloc[0]) * 365),
            "quotes": fit["n_quotes"],
            "rmse_vol": round(fit["rmse_vol_points"] * 100, 3),
            "max_vol": round(fit["max_vol_error"] * 100, 3),
            "inside": quality["inside_spread"],
            "n": quality["n"],
            "worst_spreads": round(quality["worst_in_spreads"], 1),
            "linear_bad": linear["violations"],
            "svi_bad": after["violations"],
            "min_g": round(fit["min_g"], 4),
            "left": round(fit["left_slope"], 3),
            "right": round(fit["right_slope"], 3),
            "rho": round(fit["params"].rho, 3),
            "equal_rmse": round(equal["rmse_vol_points"] * 100, 3),
            "equal_worst": round(equal_q["worst_in_spreads"], 1),
        })

        if curve is None or fit["n_quotes"] > curve["quotes"]:
            s_ = smile.dropna(subset=["iv"]).sort_values("strike")
            F = float(s_["forward"].iloc[0])
            T = float(s_["T"].iloc[0])
            k = np.log(s_["strike"].values / F)
            grid = np.linspace(k.min() - 0.08, k.max() + 0.08, 160)
            curve = {
                "expiry": expiry, "quotes": fit["n_quotes"], "forward": round(F, 2),
                "days": round(T * 365),
                "points": [[round(float(x), 4), round(float(v), 4)]
                           for x, v in zip(k, s_["iv"].values)],
                "fit": [[round(float(x), 4),
                         round(float(fit["params"].implied_vol(x, T)), 4)]
                        for x in grid],
            }

    return {"rows": rows, "curve": curve,
            "linear_total": sum(r["linear_bad"] for r in rows),
            "svi_total": sum(r["svi_bad"] for r in rows)}


def surface_data():
    print(f"pulling {TICKER} chains...")
    surface = vsurf.build_surface(TICKER, max_expiries=N_EXPIRIES, verbose=True)
    if surface.empty:
        return None

    terms = vsurf.term_structure(surface)

    smiles = []
    for expiry, smile in surface.groupby("expiry", sort=False):
        s = smile.sort_values("moneyness")
        smiles.append({
            "expiry": expiry,
            "days": round(float(s["T"].iloc[0]) * 365),
            "forward": round(float(s["forward"].iloc[0]), 2),
            "points": [[round(float(m), 4), round(float(v), 4)]
                       for m, v in zip(s["moneyness"], s["iv"])],
        })

    front = surface[surface["expiry"] == terms.iloc[0]["expiry"]]
    flat = float(front["iv"].median())
    err = vsurf.flat_vol_error(front, flat)

    data = {
        "ticker": TICKER,
        "as_of": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        "n_quotes": int(len(surface)),
        "smiles": smiles,
        "terms": [{k: (round(float(v), 5) if isinstance(v, (float, np.floating)) else v)
                   for k, v in row.items()} for row in terms.to_dict("records")],
        "flat": {
            "expiry": terms.iloc[0]["expiry"],
            "sigma": round(flat, 4),
            "points": [[round(float(m), 4), round(float(e), 3)]
                       for m, e in zip(err["moneyness"], err["error"])],
            "worst": round(float(err["error"].abs().max()), 2),
            "median_abs": round(float(err["error"].abs().median()), 3),
        },
        "calendar_violations": int(len(vsurf.calendar_arbitrage(surface))),
        "arbitrage": arbitrage_data(surface),
        "svi": svi_data(surface),
    }

    print(f"  {data['n_quotes']} quotes over {len(smiles)} expiries, "
          f"{data['calendar_violations']} calendar violations")
    for row in data["terms"]:
        print(f"  {row['expiry']}  atm {row['atm']:.2%}  "
              f"RR {row['risk_reversal']:+.2%}  fly {row['butterfly']:+.2%}")
    a = data["arbitrage"]
    print(f"  arbitrage: {a['raw']} violations on mids, {a['tradable']} through "
          f"the spread")
    for row in a["rows"]:
        print(f"    {row['expiry']}  {row['strikes']:>4} strikes  "
              f"fly {row['butterfly']:>3}/{row['butterfly_tradable']:<3} "
              f"density mass {row['density_mass']:.3f}")
    return data


def simulated_rebalancing():
    """Std of the hedged P&L against the number of rebalances, on GBM paths."""
    T = study.HEDGE_DAYS / 252
    rng = np.random.default_rng(7)
    dt = T / SIM_STEPS
    shocks = rng.normal(0.0, 1.0, size=(SIM_PATHS, SIM_STEPS))
    log_path = np.cumsum(-0.5 * SIM_SIGMA ** 2 * dt
                         + SIM_SIGMA * np.sqrt(dt) * shocks, axis=1)
    paths = 100.0 * np.concatenate([np.ones((SIM_PATHS, 1)), np.exp(log_path)], axis=1)

    rows = []
    for n in SIM_REBALANCES:
        pnl = np.array([hedging.delta_hedge(p, 100.0, T, 0.0, SIM_SIGMA,
                                            rebalance_every=SIM_STEPS // n)["pnl"]
                        for p in paths])
        rows.append({"n": n, "mean": round(float(pnl.mean()), 4),
                     "std": round(float(pnl.std(ddof=1)), 4)})
        print(f"  {n:>4} rebalances  mean {pnl.mean():+.4f}  std {pnl.std(ddof=1):.4f}")

    premium = hedging.delta_hedge(paths[0], 100.0, T, 0.0, SIM_SIGMA)["premium"]
    # Slope of log(std) on log(N) where the asymptotics apply. Theory says -0.5.
    tail = [r for r in rows if r["n"] >= 4]
    slope = np.polyfit(np.log([r["n"] for r in tail]),
                       np.log([r["std"] for r in tail]), 1)[0]
    print(f"  log-log slope from N = 4 up: {slope:.3f}")
    return {"paths": SIM_PATHS, "sigma": SIM_SIGMA, "premium": round(float(premium), 4),
            "slope": round(float(slope), 3), "rows": rows}


def hedging_data():
    print("running the monthly SPY hedge...")
    windows = study.months(study.load())
    base = pd.DataFrame([study.trade(w) for w in windows]).set_index("date")
    stats = study.summarize(base)

    gap_r = float(np.corrcoef(base["variance_gap"], base["pnl"])[0, 1])
    gap_slope, gap_intercept = np.polyfit(base["variance_gap"], base["pnl"], 1)
    gamma_r = float(np.corrcoef(base["gamma_pnl"], base["pnl"])[0, 1])

    frequency = [{"every": 1, "mean": round(float(base["pnl"].mean()), 3),
                  "worst": round(float(base["pnl"].min()), 3)}]
    for every in study.FREQUENCIES:
        table = pd.DataFrame([study.trade(w, rebalance_every=every) for w in windows])
        diff = table["pnl"].to_numpy() - base["pnl"].to_numpy()
        frequency.append({
            "every": every,
            "mean": round(float(table["pnl"].mean()), 3),
            "vs_daily": round(float(diff.mean()), 3),
            "err_std": round(float(diff.std(ddof=1)), 3),
            "scaled": round(float(diff.std(ddof=1) / np.sqrt(every)), 3),
            "worst": round(float(table["pnl"].min()), 3),
        })

    costs = []
    for bps in study.COSTS_BPS:
        table = pd.DataFrame([study.trade(w, cost_bps=bps) for w in windows])
        s = study.summarize(table)
        costs.append({"bps": bps, "mean": round(float(s["mean"]), 3),
                      "sharpe": round(float(s["sharpe"]), 2)})

    data = {
        "n": stats["n"],
        "first": base.index[0].strftime("%Y-%m"),
        "last": base.index[-1].strftime("%Y-%m"),
        "mean": round(float(stats["mean"]), 3),
        "premium": round(float(base["premium"].mean()), 3),
        "std": round(float(stats["std"]), 3),
        "sharpe": round(float(stats["sharpe"]), 2),
        "win_rate": round(stats["win_rate"], 3),
        "worst": round(float(stats["worst"]), 3),
        "worst_month": base["pnl"].idxmin().strftime("%Y-%m"),
        "best": round(float(stats["best"]), 3),
        "best_month": base["pnl"].idxmax().strftime("%Y-%m"),
        "implied": round(float(base["implied"].mean()), 4),
        "realized": round(float(base["realized"].mean()), 4),
        "implied_above": round(float((base["implied"] > base["realized"]).mean()), 3),
        "gap_r2": round(gap_r ** 2, 3),
        "gap_intercept": round(float(gap_intercept), 3),
        "gamma_r": round(gamma_r, 4),
        "gamma_mae": round(float((base["gamma_pnl"] - base["pnl"]).abs().mean()), 3),
        "months": [[d.strftime("%Y-%m"), round(float(p), 3), round(float(g), 3)]
                   for d, p, g in zip(base.index, base["pnl"], base["gamma_pnl"])],
        "worst5": [[d.strftime("%Y-%m"), round(float(row["implied"]), 3),
                    round(float(row["realized"]), 3), round(float(row["pnl"]), 3)]
                   for d, row in base.nsmallest(5, "pnl").iterrows()],
        "frequency": frequency,
        "costs": costs,
    }
    print(f"  {data['n']} trades, mean {data['mean']}, Sharpe {data['sharpe']}, "
          f"gamma r {data['gamma_r']}, variance-gap r^2 {data['gap_r2']}")
    print("simulating rebalance frequency...")
    data["sim"] = simulated_rebalancing()
    return data


def main():
    parts = sys.argv[1:] or ["surface", "hedging"]
    unknown = set(parts) - {"surface", "hedging"}
    if unknown:
        raise SystemExit(f"unknown section(s): {', '.join(sorted(unknown))}")

    data = {}
    if OUT.exists():
        text = OUT.read_text(encoding="utf-8").strip()
        data = json.loads(text[len("window.DATA = "):].rstrip(";"))

    if "surface" in parts:
        surface = surface_data()
        if surface is None:
            print("no usable quotes - Yahoo is probably rate limiting; "
                  "keeping the surface already in data.js")
        else:
            data.update(surface)
    if "hedging" in parts:
        data["hedging"] = hedging_data()

    if not all(k in data for k in SURFACE_KEYS):
        raise SystemExit("data.js has no volatility surface and none could be fetched")

    OUT.write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
