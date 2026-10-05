"""Delta hedging: what a Black-Scholes price is actually worth.

Verify with:  python test_sanity.py

Everything else in this repo answers "what is this option worth?" That question
has an answer only because of a claim hiding inside the derivation: that an
option can be replicated by continuously trading the underlying, so its price
is the cost of that replication and nothing else. Nobody hedges continuously.

This module does the replication for real - discretely, on actual price paths -
and reports what the hedge misses. Two results come out of it, and they are the
two things a derivatives desk is in the business of knowing.

**Hedging error is not noise, it is a bet on variance.** Sell an option at 20%
implied vol and hedge it, and the money you make or lose is decided almost
entirely by whether the stock realizes more or less than 20%. In continuous
time it is exactly

    P&L  =  integral of  1/2 * Gamma * S^2 * (sigma_implied^2 - sigma_realized^2) dt

so a delta-hedged option is a position in the *difference between two
volatilities*, weighted by gamma. `gamma_pnl()` computes the discrete version
term by term, and the test suite checks it reproduces the actual simulated P&L.
That identity is why traders talk about being "long gamma" instead of "long
calls": once the delta is hedged away, the direction of the stock is gone and
only its agitation is left.

**Hedging less often doesn't cost you money, it costs you certainty.** The
expected P&L barely moves with rebalance frequency; the standard deviation
grows like the square root of the interval (Boyle & Emanuel, 1980). Hedging
weekly instead of daily is not a cheaper version of the same trade, it is the
same trade with wider error bars.
"""

import numpy as np

from black_scholes import delta, gamma, price

__all__ = ["delta_hedge", "gamma_pnl", "realized_vol"]


def realized_vol(path: np.ndarray, steps_per_year: float = 252.0) -> float:
    """Annualized standard deviation of log returns along one path.

    Zero mean, not the sample mean: the replication argument cares about the
    quadratic variation of the path, and subtracting an estimated drift would
    remove part of the thing being measured. On a 30-day window the sample
    mean is nearly all noise anyway, and it biases the estimate downward.
    """
    log_returns = np.diff(np.log(path))
    return float(np.sqrt(np.mean(log_returns ** 2) * steps_per_year))


def delta_hedge(path, K, T, r, sigma_implied, option_type="call", q=0.0,
                sigma_hedge=None, rebalance_every=1, cost_bps=0.0,
                steps_per_year=252.0):
    """Sell one option at `sigma_implied`, hedge it along `path`, report the P&L.

    path : the underlying's price at each step, path[0] being the trade date and
        path[-1] expiry. Length sets the number of steps; `T` sets the calendar
        time they span.
    sigma_hedge : the volatility used to compute the hedge ratio. Defaults to
        `sigma_implied`, which is what a desk does. Setting it to something else
        answers a different and genuinely interesting question, because the
        *price* you sold at and the *delta* you hedge with do not have to come
        from the same number.
    rebalance_every : hedge every n steps. 1 is daily on daily data.
    cost_bps : proportional transaction cost on every share traded, including
        the initial hedge and the final liquidation.

    Returns a dict with the P&L and enough of the path to check it:
    'pnl', 'premium', 'payoff', 'n_rebalances', 'shares_traded', 'costs',
    'realized_vol', 'cash', 'deltas'.

    Sign convention: short the option, long the hedge. A positive P&L means the
    premium collected exceeded the cost of replicating the payoff.
    """
    path = np.asarray(path, dtype=float)
    n_steps = len(path) - 1
    if n_steps < 1:
        raise ValueError("path needs at least two points")
    if sigma_hedge is None:
        sigma_hedge = sigma_implied

    dt = T / n_steps
    premium = price(option_type, path[0], K, T, r, sigma_implied, q)

    cash = premium
    shares = 0.0
    shares_traded = 0.0
    costs = 0.0
    deltas = []
    n_rebalances = 0

    for step in range(n_steps):
        time_left = T - step * dt
        # Hedge only on rebalance steps; on the others the position is simply
        # carried, which is what makes a weekly hedge different from a daily one.
        if step % rebalance_every == 0:
            d = delta(option_type, path[step], K, max(time_left, 1e-12), r,
                      sigma_hedge, q)
            trade = d - shares
            cash -= trade * path[step]
            trade_cost = abs(trade) * path[step] * cost_bps / 10_000
            cash -= trade_cost
            costs += trade_cost
            shares_traded += abs(trade)
            shares = d
            n_rebalances += 1
        deltas.append(shares)

        # Carry: cash earns r, the stock position earns the dividend yield. Both
        # are part of the replication cost, and leaving them out is the most
        # common way a hand-rolled hedging simulation quietly drifts.
        cash *= np.exp(r * dt)
        cash += shares * path[step] * q * dt

    payoff = (max(path[-1] - K, 0.0) if option_type == "call"
              else max(K - path[-1], 0.0))
    liquidation_cost = abs(shares) * path[-1] * cost_bps / 10_000
    costs += liquidation_cost
    cash += shares * path[-1] - liquidation_cost
    shares_traded += abs(shares)

    return {
        "pnl": float(cash - payoff),
        "premium": float(premium),
        "payoff": float(payoff),
        "n_rebalances": n_rebalances,
        "shares_traded": float(shares_traded),
        "costs": float(costs),
        "realized_vol": realized_vol(path, steps_per_year),
        "cash": float(cash),
        "deltas": np.asarray(deltas),
    }


def gamma_pnl(path, K, T, r, sigma_implied, option_type="call", q=0.0,
              sigma_hedge=None):
    """The gamma decomposition of the same P&L, term by term.

        sum over steps of  1/2 * Gamma_t * S_t^2 * (sigma^2 dt - (dS/S)^2)

    Each term is one step's answer to "did the stock move more or less than the
    implied vol said it would?", scaled by how much the position cared. Summing
    them reproduces the simulated hedge P&L to within the second-order terms the
    expansion drops, which is what the test asserts.

    Worth reading the sign slowly. Short an option means short gamma, so a term
    is *positive* when the realized move was smaller than implied. Selling
    options is selling insurance against movement; you are paid when nothing
    happens.

    Returns (total, per_step_array).
    """
    path = np.asarray(path, dtype=float)
    n_steps = len(path) - 1
    if sigma_hedge is None:
        sigma_hedge = sigma_implied
    dt = T / n_steps

    terms = np.empty(n_steps)
    for step in range(n_steps):
        time_left = max(T - step * dt, 1e-12)
        g = gamma(path[step], K, time_left, r, sigma_hedge, q)
        move = (path[step + 1] - path[step]) / path[step]
        terms[step] = 0.5 * g * path[step] ** 2 * (sigma_implied ** 2 * dt - move ** 2)
    return float(terms.sum()), terms
