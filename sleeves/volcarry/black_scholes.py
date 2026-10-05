"""
Black-Scholes option pricing, implemented from scratch.

The only outside math help is scipy.stats.norm for the standard normal
CDF/PDF - everything else (d1, d2, prices, greeks, implied vol) is done
by hand from the closed-form equations.

Notation used everywhere in this project:
    S     - spot price of the underlying
    K     - strike price
    T     - time to expiry in YEARS (30 days -> 30/365)
    r     - annualized risk-free interest rate (0.05 = 5%)
    sigma - annualized volatility of the underlying (0.20 = 20%)
    q     - annualized continuous dividend yield (0.02 = 2%), default 0

Dividends matter because the stock price drops by the dividend on each
ex-dividend date. The standard fix (Merton's extension) is to discount
the spot by e^(-qT) everywhere it appears: you effectively price the
option on the stock minus the dividends you won't receive by holding
the option instead of the stock. Set q=0 and every formula collapses
back to plain Black-Scholes.
"""

import numpy as np
from scipy.stats import norm


def d1_d2(S, K, T, r, sigma, q=0.0):
    """
    The two probability-ish terms at the heart of Black-Scholes.

    d1 = [ln(S/K) + (r - q + sigma^2/2) * T] / (sigma * sqrt(T))
    d2 = d1 - sigma * sqrt(T)

    Loosely: N(d2) is the risk-neutral probability the option finishes
    in the money, and N(d1) is that probability adjusted for the size
    of the payoff when it does.
    """
    S = np.asarray(S, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    # avoid division by zero when T or sigma is 0 (option at expiry /
    # no movement) - clip to a tiny positive number instead
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(sigma, 1e-10)

    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return d1, d2


def call_price(S, K, T, r, sigma, q=0.0):
    """C = S*e^(-qT)*N(d1) - K*e^(-rT)*N(d2)"""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma, q=0.0):
    """P = K*e^(-rT)*N(-d2) - S*e^(-qT)*N(-d1)  (put-call parity form)"""
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    return K * np.exp(-r * T) * norm.cdf(-d2) - S * np.exp(-q * T) * norm.cdf(-d1)


def price(option_type, S, K, T, r, sigma, q=0.0):
    """Convenience wrapper: option_type is 'call' or 'put'."""
    if option_type == "call":
        return call_price(S, K, T, r, sigma, q)
    return put_price(S, K, T, r, sigma, q)


# ---------------------------------------------------------------------------
# Greeks - the partial derivatives of the price formula.
# Each one answers "how much does the option price move if X changes a bit?"
# ---------------------------------------------------------------------------

def delta(option_type, S, K, T, r, sigma, q=0.0):
    """
    dPrice/dSpot. Call delta is e^(-qT)*N(d1) (0 to 1), put delta is
    e^(-qT)*(N(d1)-1) (-1 to 0). A delta of 0.6 means the option gains
    ~$0.60 when the stock gains $1.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return np.exp(-q * T) * norm.cdf(d1)
    return np.exp(-q * T) * (norm.cdf(d1) - 1.0)


def gamma(S, K, T, r, sigma, q=0.0):
    """
    dDelta/dSpot - how fast delta itself changes. Same for calls and
    puts. Highest for at-the-money options near expiry.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(T, 1e-10)
    sigma = np.maximum(np.asarray(sigma, dtype=float), 1e-10)
    return np.exp(-q * T) * norm.pdf(d1) / (S * sigma * np.sqrt(T))


def vega(S, K, T, r, sigma, q=0.0):
    """
    dPrice/dVol, quoted per 1 percentage point of volatility (hence the
    /100). Same for calls and puts.
    """
    d1, _ = d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(T, 1e-10)
    return S * np.exp(-q * T) * norm.pdf(d1) * np.sqrt(T) / 100.0


def theta(option_type, S, K, T, r, sigma, q=0.0):
    """
    dPrice/dTime, quoted per calendar DAY (hence the /365). Almost
    always negative for long options - they lose value as time passes.
    """
    d1, d2 = d1_d2(S, K, T, r, sigma, q)
    T = np.maximum(T, 1e-10)
    common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
    if option_type == "call":
        yearly = (common
                  - r * K * np.exp(-r * T) * norm.cdf(d2)
                  + q * S * np.exp(-q * T) * norm.cdf(d1))
    else:
        yearly = (common
                  + r * K * np.exp(-r * T) * norm.cdf(-d2)
                  - q * S * np.exp(-q * T) * norm.cdf(-d1))
    return yearly / 365.0


def rho(option_type, S, K, T, r, sigma, q=0.0):
    """
    dPrice/dRate, quoted per 1 percentage point of interest rate.
    Calls gain when rates rise, puts lose.
    """
    _, d2 = d1_d2(S, K, T, r, sigma, q)
    if option_type == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100.0
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100.0


def all_greeks(option_type, S, K, T, r, sigma, q=0.0):
    """All five greeks in one dict - used by the UI."""
    return {
        "Delta": float(delta(option_type, S, K, T, r, sigma, q)),
        "Gamma": float(gamma(S, K, T, r, sigma, q)),
        "Vega": float(vega(S, K, T, r, sigma, q)),
        "Theta": float(theta(option_type, S, K, T, r, sigma, q)),
        "Rho": float(rho(option_type, S, K, T, r, sigma, q)),
    }


# ---------------------------------------------------------------------------
# Implied volatility - invert the formula: given a market price, find the
# sigma that makes Black-Scholes spit out that price. There is no closed
# form, so we use simple bisection (robust, no derivatives needed).
# ---------------------------------------------------------------------------

def implied_vol(option_type, market_price, S, K, T, r, q=0.0,
                lo=1e-4, hi=5.0, tol=1e-8, max_iter=100, min_vega=1e-6):
    """
    Invert the pricing formula for sigma: find the volatility that makes
    the model price equal the observed market price.

    Newton-Raphson with a guarded bisection fallback. Price is strictly
    increasing in sigma, so a sign change over [lo, hi] guarantees exactly
    one root and bisection will always find it - but bisection needs ~45
    pricing calls to reach a tight answer, and a volatility surface means
    several hundred inversions per chain. Newton uses the derivative we
    already have in closed form (vega) and gets there in 3-5.

    The guard matters. Vega collapses toward zero for deep in- or
    out-of-the-money options, and dividing by a near-zero derivative sends
    Newton somewhere useless. So every Newton step is checked against the
    bracket: if it lands outside, or vega is too small to trust, the step
    is discarded and a bisection step is taken instead. The bracket narrows
    on every iteration either way, so the fallback keeps converging rather
    than starting over.

    `tol` is a tolerance on SIGMA, not on price - this is the one place the
    distinction bites. Stopping when the price error is small sounds right
    and is wrong for exactly the options where vega is small: a deep ITM
    call is worth intrinsic-plus-epsilon at 3% vol and at 80% vol alike, so
    "the price matches to a millionth" can be true a long way from the
    right volatility. Converging on sigma instead makes the answer mean
    what it says.

    `min_vega` is the other half of that, in vega()'s own units (price
    change per percentage point of vol). When vega at the solution is below
    it the quote genuinely does not determine a volatility - there is no
    number to return, and returning one anyway is how a garbage IV ends up
    plotted on a surface. np.nan is the honest answer, and so it is also
    the answer for a quote below intrinsic value or above the underlying,
    which real chains produce constantly.
    """
    f = lambda sig: price(option_type, S, K, T, r, sig, q) - market_price

    # f is strictly increasing in sigma, which is what makes all of this
    # work: f < 0 always means "sigma too low", never anything else, so the
    # bracket update below needs no sign bookkeeping.
    f_lo, f_hi = f(lo), f(hi)
    if f_lo > 0 or f_hi < 0:       # price outside what any vol in range gives
        return float("nan")
    if f_lo == 0.0:
        # The floor already reproduces the price exactly, which on a real
        # chain means the quote is at discounted intrinsic and any small
        # vol fits it equally well. Fall through to the vega check, which
        # is what turns that into nan rather than a fake 0.01%.
        lo = hi = float(lo)

    # Brenner-Subrahmanyam (1988): for an at-the-money option,
    # C ~ 0.4 * S * sigma * sqrt(T), so sigma ~ 2.5 * (C/S) / sqrt(T).
    # Rough away from the money, but a starting point near the answer is
    # most of what Newton needs, and the bracket catches it when it isn't.
    sigma = float(np.clip(2.5 * (market_price / S) / np.sqrt(max(T, 1e-10)), lo, hi))

    for _ in range(max_iter):
        f_sigma = f(sigma)
        # vega() is quoted per percentage point, so the actual derivative
        # dPrice/dsigma is 100x it. Newton with the quoted number takes
        # hundred-fold steps and then spends the rest of its iterations
        # being rescued by the bracket, which is a slow way to run
        # bisection.
        dp_dsigma = 100.0 * float(vega(S, K, T, r, sigma, q))

        # Newton's own error estimate: |f| / (dPrice/dsigma) is roughly how
        # far sigma still is from the root, in volatility units. Stopping on
        # that rather than on the bracket width is what lets Newton finish
        # early - it converges from one side, so the bracket can stay wide
        # long after sigma is correct to nine decimals.
        if dp_dsigma > 1e-12 and abs(f_sigma) / dp_dsigma < tol:
            break

        if f_sigma < 0:            # keep the bracket tight around the root
            lo = sigma
        else:
            hi = sigma
        if hi - lo < tol:
            sigma = 0.5 * (lo + hi)
            break

        step = sigma - f_sigma / dp_dsigma if dp_dsigma > 1e-12 else None
        # A Newton step outside the bracket is a step into territory where
        # the function has no root; bisect instead of chasing it.
        sigma = step if step is not None and lo < step < hi else 0.5 * (lo + hi)

    sigma = float(sigma)
    if float(vega(S, K, T, r, sigma, q)) < min_vega:
        return float("nan")        # the quote carries no volatility information
    return sigma
