"""One surface, not six smiles. No butterfly AND no calendar arbitrage.

svi.py fits each expiry on its own, under the constraint that its implied
density stays positive. That removes butterfly arbitrage everywhere, and it
says nothing at all about the direction the strikes do not run in. Six slices
each individually sound can still cross each other in maturity, and total
variance that falls as maturity rises is a calendar spread with a negative
price - free money, so the surface is wrong. The project has checked for that
since vol_surface.calendar_arbitrage() was written, and checking is not the
same as excluding.

Excluding it needs the surface to be one object with one set of parameters.
This is Gatheral and Jacquier's SSVI (2014), which is the standard way to do
it:

    w(k, theta) = theta/2 * { 1 + rho*phi(theta)*k
                              + sqrt( (phi(theta)*k + rho)^2 + 1 - rho^2 ) }

theta is the AT-THE-MONEY TOTAL VARIANCE of that expiry - put k=0 above and
the braces collapse to 2, leaving w = theta - so the term structure is a
parameter rather than an output. Everything else is shared across the whole
surface: one correlation rho for the skew's tilt, and one function phi(theta)
saying how the smile's width changes as variance grows. The power law is the
usual choice:

    phi(theta) = eta / ( theta^gamma * (1 + theta)^(1-gamma) )

So the parameter count is 3 + (one theta per expiry). Six independent SVI
slices cost 30. That is the trade the whole file is about, and it is a real
trade in both directions - fewer parameters cannot fit the quotes as closely,
and fewer parameters are what make the conditions below expressible at all.

What makes SSVI worth the loss of freedom is that both no-arbitrage
conditions are closed-form statements about those three shared parameters,
rather than something to check afterwards on a grid:

  butterfly (positive density everywhere, every expiry):
        theta * phi(theta) * (1 + |rho|) < 4
        theta * phi(theta)^2 * (1 + |rho|) <= 4
  calendar (total variance non-decreasing in maturity at every k):
        theta_t non-decreasing in t, and
        0 <= d/dtheta [ theta * phi(theta) ]
             <= (1/rho^2) * (1 + sqrt(1 - rho^2)) * phi(theta)

The second calendar condition is worth working out rather than quoting,
because for the power law it turns out to be free. Differentiating
theta*phi = eta * theta^(1-gamma) * (1+theta)^(gamma-1) and dividing by phi
collapses almost everything:

    d/dtheta [ theta*phi(theta) ] / phi(theta)  =  (1 - gamma) / (1 + theta)

So the left-hand side is bounded above by 1 - gamma, which is below 1 for any
gamma in (0,1). The right-hand side, (1 + sqrt(1-rho^2)) / rho^2, is minimised
at |rho| = 1 where it equals 1, and is above 1.04 everywhere the fit can
actually go. The condition cannot bind. calendar_ok() is kept, and checks it
numerically rather than asserting the algebra, but it will not be what rejects
a candidate.

Which means the calendar guarantee comes entirely from the OTHER half: theta
non-decreasing in maturity. That half is not free, it is the thing the fit has
to enforce, and it is exactly the thing a per-expiry fit has no way to even
express - each slice solves for its own level with no knowledge of the slice
next to it. The interesting constraint here was never the exotic one.

These are enforced during the fit: a candidate (rho, eta, gamma) that fails
any of them on the range of theta being fitted is never accepted, and theta
is monotone by construction because the fit solves for its increments. The
surface therefore cannot contain either kind of arbitrage for a structural
reason. ssvi_study.py then verifies that independently with the project's own
checkers - vol_surface.calendar_arbitrage() and svi.durrleman_g() - because a
theorem quoted from a paper and a theorem implemented correctly are different
claims.
"""

from dataclasses import dataclass

import numpy as np

from sleeves.volcarry.svi import SVIParams, durrleman_g

# Where the conditions get checked. theta is total variance, so this spans an
# ATM vol of about 6% at a week out to 60% at two years - comfortably wider
# than any equity index surface, and the fit only ever needs the conditions to
# hold over the theta it actually uses.
THETA_GRID = np.geomspace(1e-4, 1.5, 200)


@dataclass(frozen=True)
class SSVIParams:
    """The whole surface: three shared numbers plus one theta per expiry."""

    rho: float
    eta: float
    gamma: float
    thetas: tuple          # ATM total variance, in the expiries' own order
    ts: tuple              # the maturities those thetas belong to

    def phi(self, theta):
        """Smile width as a function of variance level. Power law."""
        theta = np.asarray(theta, dtype=float)
        return self.eta / (theta ** self.gamma * (1.0 + theta) ** (1.0 - self.gamma))

    def total_variance(self, k, theta: float) -> np.ndarray:
        k = np.asarray(k, dtype=float)
        p = float(self.phi(theta))
        root = np.sqrt((p * k + self.rho) ** 2 + 1.0 - self.rho ** 2)
        return theta / 2.0 * (1.0 + self.rho * p * k + root)

    def implied_vol(self, k, theta: float, T: float) -> np.ndarray:
        return np.sqrt(np.maximum(self.total_variance(k, theta), 0.0) / T)

    def theta_at(self, T: float) -> float:
        """ATM total variance at any maturity, interpolated in the term structure.

        Linear in T between the fitted expiries, which keeps it monotone
        wherever the fitted thetas are monotone - the property the calendar
        condition needs. Flat extrapolation past the ends rather than a
        straight line, because a linear extrapolation of variance can go
        negative and there is no information out there to justify a slope.
        """
        return float(np.interp(T, self.ts, self.thetas))

    def as_svi_slice(self, theta: float) -> SVIParams:
        """The same slice written in raw SVI parameters.

        Every SSVI slice IS a raw SVI slice; matching the two forms term by
        term gives

            b = theta*phi/2,  m = -rho/phi,  sigma = sqrt(1-rho^2)/phi,
            a = theta*(1-rho^2)/2

        which is worth having for two reasons. It shows the two models are the
        same function with different freedom - SSVI is raw SVI with four of the
        five parameters tied to each other and to the term structure - and it
        lets the density check already written in svi.py be pointed at this
        surface unchanged, instead of being written a second time.
        """
        p = float(self.phi(theta))
        return SVIParams(a=theta * (1.0 - self.rho ** 2) / 2.0,
                         b=theta * p / 2.0,
                         rho=self.rho,
                         m=-self.rho / p,
                         sigma=np.sqrt(1.0 - self.rho ** 2) / p)


def butterfly_ok(rho: float, eta: float, gamma: float,
                 thetas=THETA_GRID) -> bool:
    """theta*phi*(1+|rho|) < 4 and theta*phi^2*(1+|rho|) <= 4."""
    theta = np.asarray(thetas, dtype=float)
    phi = eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))
    tp = theta * phi
    factor = 1.0 + abs(rho)
    return bool(np.all(tp * factor < 4.0) and np.all(tp * phi * factor <= 4.0))


def calendar_ok(rho: float, eta: float, gamma: float,
                thetas=THETA_GRID) -> bool:
    """0 <= d(theta*phi)/dtheta <= (1+sqrt(1-rho^2))*phi/rho^2.

    For the power-law phi this is satisfied by every admissible parameter set -
    the derivation is in the module docstring, and the short version is that the
    ratio reduces to (1-gamma)/(1+theta), which is under 1, while the bound is
    over 1.04. So this function is expected never to return False for the phi
    written above, and that is a fact about the family rather than a bug here.

    It is kept, and differentiated numerically rather than by hand, for the
    reason the numerical form is the right one: the condition has to hold for
    whatever phi someone swaps in later, and a family with a different shape
    can fail it. Checking the algebra of the phi in front of it is what makes
    this a guard instead of a comment.
    """
    theta = np.asarray(thetas, dtype=float)
    phi = eta / (theta ** gamma * (1.0 + theta) ** (1.0 - gamma))
    tp = theta * phi
    d = np.gradient(tp, theta)
    if rho == 0.0:
        return bool(np.all(d >= -1e-12))
    upper = (1.0 + np.sqrt(1.0 - rho ** 2)) / rho ** 2 * phi
    return bool(np.all(d >= -1e-12) and np.all(d <= upper + 1e-12))


def admissible(rho: float, eta: float, gamma: float) -> bool:
    """Both conditions, plus the parameters' own domains."""
    if not (-1.0 < rho < 1.0 and eta > 0.0 and 0.0 < gamma < 1.0):
        return False
    return butterfly_ok(rho, eta, gamma) and calendar_ok(rho, eta, gamma)


def _thetas_for(rho: float, eta: float, gamma: float, slices: list[dict],
                n_grid: int = 60) -> np.ndarray:
    """Best theta per expiry, then forced monotone in maturity.

    Each theta is a one-dimensional weighted least squares over its own slice,
    solved by a coarse grid and a local refinement. It is one dimension because
    everything else in w(k, theta) is already fixed by (rho, eta, gamma), and
    a one-dimensional search cannot get stuck the way the five-parameter fit
    in svi.py can.

    The running maximum at the end is how the first calendar condition gets
    enforced: theta cannot fall as maturity rises, so if the unconstrained best
    fit for a later expiry came out below an earlier one, it is raised to meet
    it. That costs fit quality on that expiry, which is the point - the
    alternative is a surface with a negative calendar spread in it.
    """
    thetas = []
    for sl in slices:
        k, w, weight = sl["k"], sl["w"], sl["weights"]
        atm = float(np.interp(0.0, k, w))
        lo, hi = max(atm * 0.25, 1e-6), atm * 4.0
        best, best_err = atm, np.inf
        for _ in range(3):
            for cand in np.linspace(lo, hi, n_grid):
                p = eta / (cand ** gamma * (1.0 + cand) ** (1.0 - gamma))
                root = np.sqrt((p * k + rho) ** 2 + 1.0 - rho ** 2)
                model = cand / 2.0 * (1.0 + rho * p * k + root)
                err = float(np.sum(weight * (model - w) ** 2))
                if err < best_err:
                    best, best_err = cand, err
            span = (hi - lo) / n_grid
            lo, hi = max(best - span, 1e-6), best + span
        thetas.append(best)
    return np.maximum.accumulate(np.asarray(thetas))


def _surface_error(params: SSVIParams, slices: list[dict]) -> float:
    """Weighted sum of squared total-variance error over every quote."""
    total = 0.0
    for sl, theta in zip(slices, params.thetas):
        model = params.total_variance(sl["k"], theta)
        total += float(np.sum(sl["weights"] * (model - sl["w"]) ** 2))
    return total


def fit_ssvi(slices: list[dict], enforce: bool = True) -> dict:
    """Fit one SSVI surface to every slice at once.

    slices: [{"k", "w", "weights", "T"}, ...] in increasing maturity. k is
    log-moneyness, w is total variance - the same quantities svi.fit_svi()
    takes, one entry per expiry instead of one call per expiry.

    The search is over (rho, eta, gamma) only, three numbers, with the thetas
    solved exactly inside each evaluation. Nelder-Mead over a starting grid,
    because the outer problem is three-dimensional and non-convex and a
    gradient method walks straight into the boundary of the admissible set.

    enforce=False drops the arbitrage conditions from the search, so the study
    can put a number on what they cost in fit quality. It is not a mode anyone
    should price a book in.
    """
    from scipy.optimize import minimize

    def objective(x) -> float:
        rho, eta, gamma = float(x[0]), float(x[1]), float(x[2])
        if not (-0.999 < rho < 0.999 and 1e-3 < eta < 50.0
                and 0.01 < gamma < 0.99):
            return 1e12
        if enforce and not admissible(rho, eta, gamma):
            return 1e12
        thetas = _thetas_for(rho, eta, gamma, slices)
        p = SSVIParams(rho, eta, gamma, tuple(thetas),
                       tuple(sl["T"] for sl in slices))
        return _surface_error(p, slices)

    best_x, best_err = None, np.inf
    for rho0 in (-0.8, -0.5, -0.2):
        for eta0 in (0.5, 1.0, 2.0):
            for gamma0 in (0.3, 0.5):
                res = minimize(objective, [rho0, eta0, gamma0],
                               method="Nelder-Mead",
                               options={"xatol": 1e-4, "fatol": 1e-10,
                                        "maxiter": 800})
                if res.fun < best_err:
                    best_x, best_err = res.x, float(res.fun)

    rho, eta, gamma = (float(best_x[0]), float(best_x[1]), float(best_x[2]))
    thetas = _thetas_for(rho, eta, gamma, slices)
    params = SSVIParams(rho, eta, gamma, tuple(thetas),
                        tuple(sl["T"] for sl in slices))

    n_quotes = sum(len(sl["k"]) for sl in slices)
    return {
        "params": params,
        "sse": best_err,
        "rmse_w": float(np.sqrt(best_err / n_quotes)),
        "n_quotes": n_quotes,
        "n_params": 3 + len(slices),
        "butterfly_ok": butterfly_ok(rho, eta, gamma),
        "calendar_ok": calendar_ok(rho, eta, gamma),
        "theta_monotone": bool(np.all(np.diff(thetas) >= -1e-12)),
    }


def slices_from_surface(surface, weight_by: str = "vega") -> list[dict]:
    """Turn a vol_surface.build_surface() frame into fit_ssvi()'s input.

    Same weighting choice as svi.fit_smile(), for the same reason: a wing
    quote whose price hardly moves with volatility has an implied vol that is
    mostly rounding, and weighting by vega is the statement that the fit
    should trust a quote in proportion to how much it knows.
    """
    from sleeves.volcarry import black_scholes as bs

    out = []
    for expiry, raw in surface.groupby("expiry", sort=False):
        s = raw.dropna(subset=["iv"]).sort_values("strike")
        if len(s) < 5:
            continue
        T = float(s["T"].iloc[0])
        F = float(s["forward"].iloc[0])
        disc = float(s["discount"].iloc[0])
        rate = float(s["rate"].iloc[0])
        if weight_by == "vega":
            weights = np.array([bs.vega(F * disc, K, T, rate, v)
                                for K, v in zip(s["strike"], s["iv"])])
            weights = np.maximum(weights, 1e-4)
        else:
            weights = np.ones(len(s))
        out.append({"expiry": expiry, "T": T, "forward": F,
                    "k": np.log(s["strike"].values / F),
                    "w": s["iv"].values ** 2 * T,
                    "iv": s["iv"].values,
                    "strike": s["strike"].values,
                    "weights": weights})
    return sorted(out, key=lambda sl: sl["T"])


def density_violations(params: SSVIParams, k_lo: float = -1.5,
                       k_hi: float = 1.5, n: int = 400) -> int:
    """How many grid points have a negative implied density, over all expiries.

    Runs svi.durrleman_g() on each slice through as_svi_slice(), so this is
    the project's existing check applied to the new surface rather than a
    second implementation of it. Should be zero when butterfly_ok() is true,
    and that agreement is the point of running it.
    """
    grid = np.linspace(k_lo, k_hi, n)
    bad = 0
    for theta in params.thetas:
        bad += int(np.sum(durrleman_g(params.as_svi_slice(theta), grid) < 0))
    return bad


def calendar_violations(params: SSVIParams, k_lo: float = -0.4,
                        k_hi: float = 0.4, n: int = 81) -> int:
    """Grid points where total variance falls between consecutive expiries.

    The direct reading of the condition, on the fitted surface, rather than on
    the parameters. Should be zero, and unlike the parameter conditions this
    one needs no theorem to interpret.
    """
    grid = np.linspace(k_lo, k_hi, n)
    bad = 0
    for earlier, later in zip(params.thetas, params.thetas[1:]):
        w0 = params.total_variance(grid, earlier)
        w1 = params.total_variance(grid, later)
        bad += int(np.sum(w1 < w0 - 1e-12))
    return bad
