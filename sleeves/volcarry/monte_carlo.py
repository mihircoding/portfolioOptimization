"""
Monte Carlo option pricing - the same answer as Black-Scholes, reached a
completely different way.

Black-Scholes assumes the stock follows geometric Brownian motion (GBM):
random walk in log-space with drift. Under the risk-neutral measure the
terminal price after time T is

    S_T = S * exp( (r - q - sigma^2/2) * T  +  sigma * sqrt(T) * Z )

where Z is a standard normal draw. So instead of solving the math in
closed form, we can literally simulate that equation many thousands of
times, average the option payoffs, and discount back to today:

    price ~ e^(-rT) * mean( payoff(S_T) )

By the law of large numbers this converges to the Black-Scholes price as
the number of simulations grows - watching that happen on the Monte Carlo
page is the best evidence the closed-form model is doing what it claims.
"""

import numpy as np


def terminal_prices(S, T, r, sigma, n_paths, seed=0, q=0.0):
    """
    Draw n_paths terminal stock prices from the risk-neutral GBM
    distribution in one vectorized shot. A fixed seed keeps the page
    reproducible - change it to see a different random sample.
    """
    rng = np.random.default_rng(seed)
    z = rng.standard_normal(n_paths)
    return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)


def mc_price(option_type, S, K, T, r, sigma, n_paths, seed=0, q=0.0):
    """
    Monte Carlo price plus its standard error. The standard error
    (std of the discounted payoffs / sqrt(n)) tells you how much the
    estimate would wiggle if you re-ran with a different seed - the
    true price should sit within ~2 standard errors 95% of the time.
    """
    s_t = terminal_prices(S, T, r, sigma, n_paths, seed, q)
    if option_type == "call":
        payoffs = np.maximum(s_t - K, 0.0)
    else:
        payoffs = np.maximum(K - s_t, 0.0)
    discounted = np.exp(-r * T) * payoffs
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))


def sample_paths(S, T, r, sigma, n_paths=50, n_steps=100, seed=0, q=0.0):
    """
    Full price paths (not just endpoints) for the chart - each path is
    built step by step with the same GBM equation applied over small
    time slices dt. Returns (time grid, paths matrix [n_paths x n_steps+1]).
    """
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    z = rng.standard_normal((n_paths, n_steps))
    increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
    log_paths = np.cumsum(increments, axis=1)
    paths = S * np.exp(np.hstack([np.zeros((n_paths, 1)), log_paths]))
    return np.linspace(0.0, T, n_steps + 1), paths


def _implied_z(s_t, S, T, r, sigma, q=0.0):
    """Recover each path's standard-normal draw Z from its terminal price.

    terminal_prices() builds S_T = S * exp((r-q-sigma^2/2)*T + sigma*sqrt(T)*Z)
    from a Z it doesn't return. Every Greek estimator below needs that same
    Z (it's the random variable the derivatives below are actually taken
    with respect to), so instead of duplicating terminal_prices' RNG call
    sequence and hoping the two never drift apart, invert the one formula
    algebraically. Whatever S_T came from, this recovers exactly the Z that
    produced it.
    """
    T = max(T, 1e-10)
    sigma = max(sigma, 1e-10)
    return (np.log(s_t / S) - (r - q - 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def pathwise_delta(option_type, S, K, T, r, sigma, n_paths, seed=0, q=0.0):
    """Monte Carlo delta by differentiating the PATH, not the price formula.

    S_T is a smooth (differentiable) function of S: dS_T/dS = S_T / S,
    because S is just a multiplicative factor out front of the GBM formula.
    Swap the order of expectation and differentiation (valid here because
    the payoff is differentiable except at the single point S_T = K, which
    happens with probability zero under a continuous distribution):

        delta = e^(-rT) * E[ dPayoff/dS_T * dS_T/dS ]
              = e^(-rT) * E[ 1{S_T>K} * S_T/S ]                    (call)
              = e^(-rT) * E[ -1{S_T<K} * S_T/S ]                   (put)

    Returns (estimate, standard_error) - same convention as mc_price, and
    for the same reason: this is still a Monte Carlo estimate with sampling
    noise, not a closed-form number, and reporting it without a standard
    error would overstate how precisely it's known.
    """
    s_t = terminal_prices(S, T, r, sigma, n_paths, seed, q)
    pathwise_deriv = s_t / S
    if option_type == "call":
        sample = np.where(s_t > K, pathwise_deriv, 0.0)
    else:
        sample = np.where(s_t < K, -pathwise_deriv, 0.0)
    discounted = np.exp(-r * T) * sample
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))


def pathwise_vega(option_type, S, K, T, r, sigma, n_paths, seed=0, q=0.0):
    """Monte Carlo vega, same pathwise idea as pathwise_delta but
    differentiating S_T with respect to sigma instead of S:

        dS_T/dsigma = S_T * (sqrt(T)*Z - sigma*T)

    Z is recovered from each path via _implied_z rather than regenerated,
    so this uses exactly the same random draws terminal_prices produced -
    no new randomness, no reason for a mismatched Z to sneak in a bias.
    Divided by 100 at the end for the same reason black_scholes.vega() is:
    quoted per 1 percentage point of volatility, not per 1.0 (100 vol points).
    """
    s_t = terminal_prices(S, T, r, sigma, n_paths, seed, q)
    z = _implied_z(s_t, S, T, r, sigma, q)
    ds_dsigma = s_t * (np.sqrt(T) * z - sigma * T)

    if option_type == "call":
        sample = np.where(s_t > K, ds_dsigma, 0.0)
    else:
        sample = np.where(s_t < K, -ds_dsigma, 0.0)
    discounted = np.exp(-r * T) * sample / 100.0
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))


def likelihood_ratio_gamma(option_type, S, K, T, r, sigma, n_paths, seed=0, q=0.0):
    """Monte Carlo gamma - and the reason this file has TWO different ways
    of computing a Greek by simulation instead of one.

    Gamma is a SECOND derivative of the price with respect to S, which
    means a FIRST derivative of the payoff's slope. But a call or put
    payoff has a kink at S_T = K: its slope jumps from 0 to 1 (or -1 to 0)
    right there. Differentiate that kink and you get a Dirac delta, not a
    number - pathwise differentiation, which worked fine for delta and
    vega, breaks down completely for gamma. This is a known, structural
    limitation of the pathwise method for any Greek involving the second
    derivative of a discontinuous-slope payoff, not a bug to work around.

    The likelihood-ratio (score function) method sidesteps this entirely by
    differentiating the PROBABILITY DENSITY of S_T instead of the payoff,
    which stays perfectly smooth even when the payoff doesn't:

        price(S) = e^(-rT) * integral( payoff(s) * f(s; S) ds )
        d/dS price(S) = e^(-rT) * E[ payoff(S_T) * score(S) ]
        score(S) = d/dS[ln f(S_T; S)] = Z / (S*sigma*sqrt(T))

    Differentiating the score once more (S_T's log is Normal(ln S + drift,
    sigma^2 T), so this is a standard score/Fisher-information calculation)
    gives the weight used for gamma:

        gamma = e^(-rT)/(S^2 sigma^2 T) * E[ payoff(S_T) * (Z^2 - 1 - Z*sigma*sqrt(T)) ]

    This weight does not depend on option_type or on the payoff being
    differentiable at all - it would work identically for a digital
    option's genuinely discontinuous payoff, which is exactly the case
    pathwise differentiation cannot handle even in principle.
    """
    s_t = terminal_prices(S, T, r, sigma, n_paths, seed, q)
    z = _implied_z(s_t, S, T, r, sigma, q)
    weight = (z**2 - 1.0 - z * sigma * np.sqrt(T)) / (S**2 * sigma**2 * T)

    payoffs = np.maximum(s_t - K, 0.0) if option_type == "call" else np.maximum(K - s_t, 0.0)
    discounted = np.exp(-r * T) * payoffs * weight
    return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))


def mc_greeks(option_type, S, K, T, r, sigma, n_paths, seed=0, q=0.0):
    """Delta, vega and gamma by simulation, bundled the way
    black_scholes.all_greeks() bundles the closed-form versions - each
    paired with its Monte Carlo standard error, since (unlike the
    closed-form Greeks) these are estimates, not exact numbers."""
    d, d_se = pathwise_delta(option_type, S, K, T, r, sigma, n_paths, seed, q)
    v, v_se = pathwise_vega(option_type, S, K, T, r, sigma, n_paths, seed, q)
    g, g_se = likelihood_ratio_gamma(option_type, S, K, T, r, sigma, n_paths, seed, q)
    return {
        "Delta": (d, d_se),
        "Vega": (v, v_se),
        "Gamma": (g, g_se),
    }


def convergence_curve(option_type, S, K, T, r, sigma, max_paths, seed=0, q=0.0):
    """
    MC estimate as a function of sample size, computed by taking running
    means over ONE big draw (so the curve is smooth and cheap). Returns
    (sample sizes, estimates) for the convergence chart.
    """
    s_t = terminal_prices(S, T, r, sigma, max_paths, seed, q)
    if option_type == "call":
        payoffs = np.maximum(s_t - K, 0.0)
    else:
        payoffs = np.maximum(K - s_t, 0.0)
    discounted = np.exp(-r * T) * payoffs
    running_mean = np.cumsum(discounted) / np.arange(1, max_paths + 1)
    # log-spaced sample points so the x-axis reads nicely from 100 to max
    start = min(100, max_paths)
    idx = np.unique(np.geomspace(start, max_paths, 200).astype(int)) - 1
    return idx + 1, running_mean[idx]
