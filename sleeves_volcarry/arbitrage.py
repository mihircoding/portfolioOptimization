"""Does the quoted chain admit arbitrage across strikes?

vol_surface.py already checks the time direction: total variance must not
fall as maturity rises, or a calendar spread is free money. This is the
other direction, across strikes at one expiry, and it is the one that
actually catches bad data.

Three conditions, all of them statements about the call price C(K) that no
model is needed to derive - they follow from the payoff alone:

    1. C is decreasing in K.       A call struck lower is worth more.
    2. Its slope is at least -e^(-rT). A call spread cannot cost more than
       the strike difference it can pay.
    3. C is convex in K.           The butterfly K1/K2/K3, long the wings
       and short two of the body, pays out nothing worse than zero, so it
       cannot cost less than nothing.

Condition 3 is the interesting one, because the second derivative of the
call price *is* the risk-neutral density (Breeden & Litzenberger, 1978):

    q(K) = e^(rT) d2C/dK2

so a butterfly quoted at a negative price is the market assigning negative
probability to a price range. That never means the market is wrong; it means
the three quotes were not all alive at the same instant, or one leg is stale,
or the forward used to convert puts into calls is off.

Which is why everything here is measured against the bid-ask spread of the
legs involved. A violation smaller than what you would pay to trade it is a
quoting artifact, not an opportunity, and reporting those two as the same
number is how a screen ends up with hundreds of "arbitrages" a day and no
trades. Both counts are reported separately below.

Puts are converted to calls through put-call parity against the same forward
the surface was built on, so one continuous call curve spans the whole strike
range rather than two half-curves that meet at the money.
"""

import numpy as np
import pandas as pd


def call_curve(smile: pd.DataFrame) -> pd.DataFrame:
    """One call-price curve per expiry, OTM puts converted through parity.

    C - P = e^(-rT)(F - K). The OTM put at a low strike and the (unquoted,
    illiquid) call at that strike carry the same information; parity is the
    exchange rate between them, and it uses the forward the chain itself
    implied rather than a dividend guess.

    Returns strike / call / half_spread, sorted by strike. `half_spread` is
    carried through because every test below needs to know whether a
    violation is bigger than the cost of trading it.
    """
    if smile.empty:
        return pd.DataFrame(columns=["strike", "call", "half_spread"])

    F = float(smile["forward"].iloc[0])
    disc = float(smile["discount"].iloc[0])

    out = smile[["strike", "type", "mid", "spread"]].copy()
    is_put = out["type"] == "put"
    out["call"] = np.where(is_put, out["mid"] + disc * (F - out["strike"]), out["mid"])
    # parity moves the price, not the uncertainty in it: the spread of the
    # quoted leg is the spread of the synthetic call
    out["half_spread"] = out["spread"] / 2.0
    out = out[["strike", "call", "half_spread"]].sort_values("strike")
    return out.groupby("strike", as_index=False).mean().reset_index(drop=True)


def vertical_checks(curve: pd.DataFrame, discount: float = 1.0) -> pd.DataFrame:
    """Conditions 1 and 2, on every adjacent pair of strikes.

    slope = dC/dK must lie in [-discount, 0]. Above zero a call spread has a
    negative price for a non-negative payoff; below -discount it costs more
    than the discounted maximum it can ever pay.
    """
    rows = []
    k = curve["strike"].to_numpy(float)
    c = curve["call"].to_numpy(float)
    h = curve["half_spread"].to_numpy(float)
    for i in range(len(k) - 1):
        dk = k[i + 1] - k[i]
        slope = (c[i + 1] - c[i]) / dk
        # how far outside the band, in dollars of option premium
        excess = max(0.0, c[i + 1] - c[i], -(c[i + 1] - c[i]) - discount * dk)
        cost = h[i] + h[i + 1]
        rows.append({"k_lo": k[i], "k_hi": k[i + 1], "slope": slope,
                     "excess": excess, "cost": cost,
                     "violation": excess > 0, "tradable": excess > cost})
    return pd.DataFrame(rows)


def butterfly_checks(curve: pd.DataFrame, rate: float = 0.0,
                     T: float = 0.0) -> pd.DataFrame:
    """Condition 3, plus the risk-neutral density, on every strike triplet.

    With unequal strike spacing the butterfly is weighted so its payoff is a
    tent of height (K2-K1)(K3-K2)/(K3-K1) and never negative:

        w1 = (K3-K2)/(K3-K1),  w3 = (K2-K1)/(K3-K1),  cost = w1*C1 - C2 + w3*C3

    `density` is that cost turned into a probability density: the same
    second divided difference, compounded forward. Summed against the strike
    spacing it integrates to 1 over a complete strike range, and to less
    than 1 over a quoted range that stops short of the tails - which is a
    useful reading of how much of the distribution the chain covers.
    """
    k = curve["strike"].to_numpy(float)
    c = curve["call"].to_numpy(float)
    h = curve["half_spread"].to_numpy(float)

    rows = []
    for i in range(1, len(k) - 1):
        k1, k2, k3 = k[i - 1], k[i], k[i + 1]
        w1 = (k3 - k2) / (k3 - k1)
        w3 = (k2 - k1) / (k3 - k1)
        cost = w1 * c[i - 1] - c[i] + w3 * c[i + 1]
        # trading it means crossing the spread on all three legs, the body twice
        trade_cost = w1 * h[i - 1] + 2 * h[i] + w3 * h[i + 1]
        second_diff = 2 * cost / ((k2 - k1) * (k3 - k2))
        rows.append({
            "strike": k2, "k_lo": k1, "k_hi": k3,
            "butterfly": cost, "cost": trade_cost,
            "density": float(np.exp(rate * T) * second_diff),
            "width": (k3 - k1) / 2.0,
            "violation": cost < 0, "tradable": cost < -trade_cost,
        })
    return pd.DataFrame(rows)


def check_smile(smile: pd.DataFrame) -> dict:
    """Run all three conditions on one expiry's quotes."""
    curve = call_curve(smile)
    if len(curve) < 3:
        return {"expiry": smile["expiry"].iloc[0] if len(smile) else None,
                "n_strikes": len(curve), "usable": False}

    disc = float(smile["discount"].iloc[0])
    rate = float(smile["rate"].iloc[0])
    T = float(smile["T"].iloc[0])

    verticals = vertical_checks(curve, disc)
    butterflies = butterfly_checks(curve, rate, T)
    # signed, not clipped: quoting noise puts a positive and a negative
    # butterfly next to each other, and clipping would keep only the half
    # that inflates the answer
    mass = float((butterflies["density"] * butterflies["width"]).sum())

    return {
        "expiry": smile["expiry"].iloc[0],
        "T": T, "n_strikes": len(curve), "usable": True,
        "vertical_violations": int(verticals["violation"].sum()),
        "vertical_tradable": int(verticals["tradable"].sum()),
        "butterfly_violations": int(butterflies["violation"].sum()),
        "butterfly_tradable": int(butterflies["tradable"].sum()),
        "worst_butterfly": float(butterflies["butterfly"].min()),
        "density_mass": mass,
        "verticals": verticals, "butterflies": butterflies, "curve": curve,
    }


def check_surface(surface: pd.DataFrame) -> pd.DataFrame:
    """check_smile() for every expiry, one row each."""
    rows = []
    for _, smile in surface.groupby("expiry", sort=False):
        r = check_smile(smile)
        r.pop("verticals", None); r.pop("butterflies", None); r.pop("curve", None)
        rows.append(r)
    out = pd.DataFrame(rows)
    return out.sort_values("T").reset_index(drop=True) if "T" in out else out


def interpolated_smile_violations(smile: pd.DataFrame, n: int = 200,
                                  in_variance: bool = True) -> dict:
    """Does filling the gaps between quoted strikes invent arbitrage?

    A surface is quoted at maybe 40 strikes and used at any strike, so
    something has to interpolate. atm_vol() interpolates total variance and
    says in passing that interpolating volatility instead can break the
    no-arbitrage conditions between the quoted points. This measures that
    claim: interpolate one way or the other onto a fine strike grid, price
    the grid with Black-Scholes, and count the butterflies that come out
    negative on strikes where nothing was ever quoted.

    Returns the count and the worst butterfly, in dollars.
    """
    import black_scholes as bs

    s = smile.dropna(subset=["iv"]).sort_values("strike")
    if len(s) < 3:
        return {"violations": 0, "worst": 0.0, "n": 0}

    F = float(s["forward"].iloc[0])
    disc = float(s["discount"].iloc[0])
    rate = float(s["rate"].iloc[0])
    T = float(s["T"].iloc[0])
    spot_equiv = F * disc

    grid = np.linspace(s["strike"].min(), s["strike"].max(), n)
    if in_variance:
        w = np.interp(grid, s["strike"], s["iv"] ** 2 * T)
        iv = np.sqrt(np.maximum(w, 0) / T)
    else:
        iv = np.interp(grid, s["strike"], s["iv"])

    calls = np.array([float(bs.price("call", spot_equiv, K, T, rate, v))
                      for K, v in zip(grid, iv)])
    curve = pd.DataFrame({"strike": grid, "call": calls,
                          "half_spread": np.zeros(n)})
    b = butterfly_checks(curve, rate, T)
    return {"violations": int(b["violation"].sum()),
            "worst": float(b["butterfly"].min()), "n": len(b)}
