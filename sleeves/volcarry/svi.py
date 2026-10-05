"""Fit the smile under the no-arbitrage constraint instead of joining dots.

arbitrage_study.py ends on a promise: linear interpolation between quoted
implied vols produces negative butterfly prices between the strikes, because
the quoted points are not convex to begin with, and a surface you can price
a book with has to be FITTED under the convexity constraint rather than
joined up dot to dot. This is that fit.

The parameterization is Gatheral's SVI (stochastic volatility inspired), in
raw form. For log-moneyness k = log(K/F) it models TOTAL VARIANCE, w = sigma^2 T:

    w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )

Five parameters, and each one is a feature of the smile you can point at:

    a       the overall level of variance
    b       how fast the wings rise (the slope out of the money)
    rho     the tilt - negative leans the left wing up, which is what an
            equity index does, because the crash is on the put side
    m       where the minimum sits, in log-moneyness
    sigma   how rounded the bottom is; sigma -> 0 gives a kink at m

Why this function and not a spline. It is a hyperbola, so it is convex in k
by construction and goes linear in both wings - which is exactly the shape
the no-arbitrage conditions demand. Lee's moment formula caps the asymptotic
slope of total variance at 2 per unit of log-moneyness; a polynomial or a
cubic spline has no such shape, will happily turn over in the wings, and
prices negative densities there no matter how well it fits the quotes.

Fitting is nonlinear in five parameters, so a bad start point matters. The
approach here is the standard one: for FIXED (m, sigma), the model is linear
in (a, b*rho, b), so those three come from a small least-squares solve;
only (m, sigma) get searched. That turns a 5-dimensional non-convex problem
into a 2-dimensional one over a well-behaved surface.

What is checked afterwards is the part that matters. A fit that reproduces
the quotes but prices a negative butterfly is worse than useless, so the
result carries Gatheral-Jacquier's g(k) - the density's sign, in closed form
from the fitted parameters - and the fit is rejected if it goes negative
anywhere on the quoted range.
"""

from dataclasses import dataclass

import numpy as np

# Weight floor for the vega weighting below. An option with no vega has an
# implied vol that is not a measurement, and letting it into a least squares
# on equal terms with an at-the-money quote is how a fit ends up chasing the
# noisiest points on the chain.
MIN_WEIGHT = 1e-4


@dataclass(frozen=True)
class SVIParams:
    a: float
    b: float
    rho: float
    m: float
    sigma: float

    def total_variance(self, k):
        k = np.asarray(k, dtype=float)
        return self.a + self.b * (self.rho * (k - self.m)
                                  + np.sqrt((k - self.m) ** 2 + self.sigma ** 2))

    def implied_vol(self, k, T: float):
        w = np.maximum(self.total_variance(k), 1e-12)
        return np.sqrt(w / T)

    @property
    def left_slope(self) -> float:
        """Asymptotic slope of total variance in the left wing. Lee's bound
        says this cannot exceed 2 without admitting arbitrage."""
        return self.b * (1.0 - self.rho)

    @property
    def right_slope(self) -> float:
        return self.b * (1.0 + self.rho)

    def as_dict(self) -> dict:
        return {"a": self.a, "b": self.b, "rho": self.rho, "m": self.m,
                "sigma": self.sigma}


def durrleman_g(p: SVIParams, k) -> np.ndarray:
    """Gatheral-Jacquier's g(k): the risk-neutral density, up to a positive factor.

        g(k) = (1 - k w'/(2w))^2 - (w'/4)(1/w + 1/4) w' + w''/2

    A smile is free of butterfly arbitrage exactly when g >= 0 everywhere
    (and the density integrates to one, which the wings handle). The value
    of having a parametric smile is that this is a FORMULA, evaluated at any
    k without pricing a single option - where arbitrage.py has to build
    three option prices per strike and difference them.
    """
    k = np.asarray(k, dtype=float)
    d = k - p.m
    root = np.sqrt(d ** 2 + p.sigma ** 2)
    w = p.a + p.b * (p.rho * d + root)
    w1 = p.b * (p.rho + d / root)                       # dw/dk
    w2 = p.b * p.sigma ** 2 / root ** 3                 # d2w/dk2
    w = np.maximum(w, 1e-12)
    return (1 - k * w1 / (2 * w)) ** 2 - (w1 / 4) * (1 / w + 0.25) * w1 + w2 / 2


def is_arbitrage_free(p: SVIParams, k_lo: float = -1.5, k_hi: float = 1.5,
                      n: int = 601) -> dict:
    """Butterfly and wing checks on the fitted parameters alone.

    The grid is deliberately wider than any chain's quoted strikes: the
    reason to fit a function is to use it where nothing was quoted, so the
    check has to cover there too.
    """
    grid = np.linspace(k_lo, k_hi, n)
    g = durrleman_g(p, grid)
    w = p.total_variance(grid)
    return {
        "butterfly_free": bool(np.all(g >= -1e-10)),
        "min_g": float(g.min()),
        "min_g_at": float(grid[int(np.argmin(g))]),
        "positive_variance": bool(np.all(w > 0)),
        "left_slope": p.left_slope,
        "right_slope": p.right_slope,
        "lee_bound_ok": bool(max(p.left_slope, p.right_slope) <= 2.0 + 1e-9),
    }


def _linear_stage(k, w, weights, m: float, sigma: float):
    """Best (a, b, rho) for a fixed (m, sigma), by weighted least squares.

    With y = (k - m)/sigma, the model is

        w = a + b*sigma*rho*y + b*sigma*sqrt(y^2 + 1)

    which is linear in (a, c1 = b*sigma*rho, c2 = b*sigma). Solving it
    directly is what makes the outer search two-dimensional. The parameter
    bounds (b >= 0, |rho| <= 1, a >= 0) are imposed by clipping afterwards
    rather than by a constrained solve - on real chains the unconstrained
    solution is almost always already inside, and when it is not, the outer
    search sees the worse fit and walks away from that (m, sigma).
    """
    y = (k - m) / sigma
    design = np.column_stack([np.ones_like(y), y, np.sqrt(y ** 2 + 1.0)])
    sw = np.sqrt(weights)
    coef, *_ = np.linalg.lstsq(design * sw[:, None], w * sw, rcond=None)
    a, c1, c2 = coef

    c2 = max(c2, 1e-8)
    rho = np.clip(c1 / c2, -0.999, 0.999)
    b = c2 / sigma
    a = max(a, 1e-8)
    return SVIParams(float(a), float(b), float(rho), float(m), float(sigma))


def _weighted_rmse(p: SVIParams, k, w, weights) -> float:
    resid = p.total_variance(k) - w
    return float(np.sqrt(np.sum(weights * resid ** 2) / np.sum(weights)))


def fit_svi(k, total_variance, weights=None, enforce_no_arbitrage: bool = True,
            n_m: int = 41, n_sigma: int = 31) -> dict:
    """Fit one expiry's smile.

    k                log-moneyness of each quote
    total_variance   sigma^2 * T at each quote
    weights          relative confidence per quote; vega or 1/spread are the
                     two defensible choices and both beat equal weighting

    The outer search is a grid over (m, sigma) followed by a local refine,
    rather than a gradient method from one start point. Five-parameter SVI
    has well-documented local minima; a 41x31 grid over a range set by the
    data costs about a thousand three-parameter least-squares solves, which
    is microseconds, and removes the failure mode entirely.

    With enforce_no_arbitrage the search only ever accepts a candidate whose
    g(k) stays non-negative. That is the whole point of the exercise: the
    constraint is imposed during the fit, so the result cannot be a surface
    that fits beautifully and prices a negative probability.
    """
    k = np.asarray(k, dtype=float)
    w = np.asarray(total_variance, dtype=float)
    if len(k) < 5:
        raise ValueError("need at least 5 quotes to fit 5 parameters")
    weights = (np.ones_like(k) if weights is None
               else np.maximum(np.asarray(weights, dtype=float), MIN_WEIGHT))

    spread = max(k.max() - k.min(), 1e-3)
    m_grid = np.linspace(k.min() - 0.25 * spread, k.max() + 0.25 * spread, n_m)
    sigma_grid = np.geomspace(0.01 * spread, 2.0 * spread, n_sigma)

    best, best_err = None, np.inf
    rejected = 0
    for m in m_grid:
        for sigma in sigma_grid:
            cand = _linear_stage(k, w, weights, m, sigma)
            err = _weighted_rmse(cand, k, w, weights)
            if err >= best_err:
                continue
            if enforce_no_arbitrage and not is_arbitrage_free(
                    cand, k.min() - 0.5, k.max() + 0.5)["butterfly_free"]:
                rejected += 1
                continue
            best, best_err = cand, err

    if best is None:
        raise RuntimeError("no arbitrage-free SVI fit found on this smile")

    best, best_err = _refine(best, k, w, weights, best_err,
                             enforce_no_arbitrage)

    checks = is_arbitrage_free(best, k.min() - 0.5, k.max() + 0.5)

    # Residuals in sqrt(total variance), which is sigma * sqrt(T) and NOT a
    # volatility. fit_smile() knows T and converts; nothing here does, and a
    # key called "vol" that holds sigma*sqrt(T) is how a table ends up with a
    # different unit in every row. See fit_smile().
    resid = np.sqrt(np.maximum(best.total_variance(k), 1e-12)) - np.sqrt(w)
    return {
        "params": best,
        "rmse_total_var": best_err,
        "rmse_sqrt_total_var": float(np.sqrt(np.mean(resid ** 2))),
        "max_sqrt_total_var_error": float(np.max(np.abs(resid))),
        "rejected_candidates": rejected,
        "n_quotes": len(k),
        **checks,
    }


def _refine(p: SVIParams, k, w, weights, err: float, enforce: bool,
            rounds: int = 60) -> tuple:
    """Shrinking local search around (m, sigma). Two parameters, so a
    coordinate walk is both adequate and easy to read; there is nothing here
    a gradient would buy that is worth the extra machinery."""
    step_m = 0.05 * max(k.max() - k.min(), 1e-3)
    step_s = 0.5 * p.sigma
    for _ in range(rounds):
        improved = False
        for dm, ds in ((step_m, 0), (-step_m, 0), (0, step_s), (0, -step_s)):
            m, sigma = p.m + dm, p.sigma + ds
            if sigma <= 1e-6:
                continue
            cand = _linear_stage(k, w, weights, m, sigma)
            e = _weighted_rmse(cand, k, w, weights)
            if e < err and (not enforce or is_arbitrage_free(
                    cand, k.min() - 0.5, k.max() + 0.5)["butterfly_free"]):
                p, err, improved = cand, e, True
        if not improved:
            step_m *= 0.5
            step_s *= 0.5
            if step_m < 1e-6:
                break
    return p, err


def fit_smile(smile, weight_by: str = "vega") -> dict:
    """Fit a vol_surface.smile_for_expiry() frame.

    weight_by:
      "vega"    weight each quote by its Black-Scholes vega. A wing option
                whose price barely moves with volatility has an implied vol
                that is mostly quantization noise, and vega weighting is the
                statement that the fit should care about a quote in
                proportion to how much the quote actually knows.
      "spread"  weight by 1 / half-spread, which says the same thing in the
                market's own units and needs no model to compute.
      "equal"   no weighting, for comparison - see how much worse the wings
                drag the at-the-money fit.
    """
    from sleeves.volcarry import black_scholes as bs

    s = smile.dropna(subset=["iv"]).sort_values("strike")
    T = float(s["T"].iloc[0])
    F = float(s["forward"].iloc[0])
    disc = float(s["discount"].iloc[0])
    rate = float(s["rate"].iloc[0])
    k = np.log(s["strike"].values / F)
    w = s["iv"].values ** 2 * T

    if weight_by == "vega":
        weights = np.array([bs.vega(F * disc, K, T, rate, v)
                            for K, v in zip(s["strike"], s["iv"])])
    elif weight_by == "spread":
        half = (s["ask"] - s["bid"]).values / 2 if "ask" in s else None
        weights = (1.0 / np.maximum(half, 1e-4) if half is not None
                   else np.ones_like(k))
    else:
        weights = np.ones_like(k)

    out = fit_svi(k, w, weights)
    out["T"] = T
    out["forward"] = F
    out["expiry"] = s["expiry"].iloc[0] if "expiry" in s else None

    # Repricing error in implied VOLATILITY, which needs T and so cannot be
    # computed inside fit_svi(). This used to be reported straight out of there
    # in sqrt(total variance) units under the name "rmse_vol_points", which is
    # sigma * sqrt(T) - so the same fit quality read as a different number at
    # every maturity, and a one-week expiry looked about seven times better
    # than it was. Dividing by sqrt(T) puts every row in the same unit, which
    # is the only way the per-expiry table in the README means anything.
    root_t = np.sqrt(T)
    out["rmse_vol_points"] = out["rmse_sqrt_total_var"] / root_t
    out["max_vol_error"] = out["max_sqrt_total_var_error"] / root_t
    return out


def grid_violations(smile, fit: dict, n: int = 200) -> dict:
    """Butterfly violations on a fine strike grid, priced from the fit.

    The same measurement arbitrage_study.py already runs on the linearly
    interpolated smile, so the two numbers sit next to each other and mean
    the same thing. The expected answer is zero, and it should be zero for a
    structural reason rather than a lucky one: the fit was only ever allowed
    to accept parameters whose g(k) is non-negative.
    """
    from sleeves.volcarry import arbitrage as arb
    from sleeves.volcarry import black_scholes as bs
    import pandas as pd

    s = smile.dropna(subset=["iv"]).sort_values("strike")
    F = float(s["forward"].iloc[0])
    disc = float(s["discount"].iloc[0])
    rate = float(s["rate"].iloc[0])
    T = float(s["T"].iloc[0])
    spot_equiv = F * disc
    p = fit["params"]

    grid = np.linspace(s["strike"].min(), s["strike"].max(), n)
    iv = p.implied_vol(np.log(grid / F), T)
    calls = np.array([float(bs.price("call", spot_equiv, K, T, rate, v))
                      for K, v in zip(grid, iv)])
    curve = pd.DataFrame({"strike": grid, "call": calls,
                          "half_spread": np.zeros(n)})
    b = arb.butterfly_checks(curve, rate, T)
    return {"violations": int(b["violation"].sum()),
            "worst": float(b["butterfly"].min()), "n": len(b)}


def fit_quality(smile, fit: dict) -> dict:
    """Does the fitted surface reprice the chain inside its own spreads?

    The right yardstick for a smile fit is not an r-squared, it is whether
    the model price lands between the bid and the ask. A fit that is inside
    the spread everywhere is indistinguishable from the market at the only
    resolution the market has; one that misses by more than the spread is
    claiming an edge, and on a liquid chain that claim is nearly always the
    model being wrong rather than the market.
    """
    from sleeves.volcarry import black_scholes as bs

    s = smile.dropna(subset=["iv"]).sort_values("strike")
    F = float(s["forward"].iloc[0])
    disc = float(s["discount"].iloc[0])
    rate = float(s["rate"].iloc[0])
    T = float(s["T"].iloc[0])
    p = fit["params"]

    iv = p.implied_vol(np.log(s["strike"].values / F), T)
    model = np.array([float(bs.price(t, F * disc, K, T, rate, v))
                      for t, K, v in zip(s["type"], s["strike"], iv)])
    err = np.abs(model - s["mid"].values)
    half = s["spread"].values / 2.0
    return {
        "inside_spread": int(np.sum(err <= half)),
        "n": len(s),
        "worst_error": float(err.max()),
        "worst_in_spreads": float(np.max(err / np.maximum(half, 1e-6))),
        "mean_error": float(err.mean()),
    }
