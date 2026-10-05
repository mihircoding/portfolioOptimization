"""
American and European option pricing via a Cox-Ross-Rubinstein (CRR)
binomial tree, implemented from scratch.

Black-Scholes (black_scholes.py) only prices European exercise - it can't,
because the closed-form solution assumes the holder can't act early. Most
US listed single-stock options are American-style, so this module exists
to answer two questions the Black-Scholes model structurally can't:

    1. How much is early exercise actually worth? (the "early-exercise
       premium" - American price minus European price, same inputs)
    2. Does a discrete-time tree agree with the continuous-time
       closed form as it's refined? (a cross-validation of both models,
       not just a new feature)

Notation matches black_scholes.py: S, K, T, r, sigma, q. `steps` is the
number of time slices in the tree - more steps means finer approximation
of continuous time, at the cost of O(steps^2) work.
"""

import numpy as np


def crr_price(option_type, exercise, S, K, T, r, sigma, q=0.0, steps=200):
    """
    Price an option on a `steps`-step CRR binomial tree.

    option_type : 'call' or 'put'
    exercise    : 'european' (only exercisable at expiry) or
                  'american' (exercisable at any node)

    The tree: at each step the stock moves up by a factor u = e^(sigma*sqrt(dt))
    or down by d = 1/u. u and d are chosen so that, as steps -> infinity, the
    tree's terminal distribution converges to the same lognormal distribution
    Black-Scholes assumes - that convergence is what test_sanity.py checks.

    p is the RISK-NEUTRAL up-probability, not a real-world one - it's solved
    for so the discounted expected stock price grows at the risk-free rate
    net of dividends, exactly the same no-arbitrage argument Black-Scholes
    is built on:

        p = (e^((r-q)*dt) - d) / (u - d)

    Valuation walks the tree backwards from expiry: at each node, the value
    is the discounted risk-neutral expectation of the two children. For an
    American option, that continuation value is then compared against the
    payoff of exercising *right now* at that node, and the larger of the two
    wins - the one place American and European pricing actually diverge.
    """
    if steps < 1:
        raise ValueError("steps must be a positive integer")

    dt = T / steps
    u = np.exp(sigma * np.sqrt(dt))
    d = 1.0 / u
    disc = np.exp(-r * dt)
    p = (np.exp((r - q) * dt) - d) / (u - d)

    if not (0.0 <= p <= 1.0):
        # can happen with extreme inputs (huge dt, tiny sigma) - the tree's
        # up/down move no longer brackets the risk-neutral drift, so there's
        # no arbitrage-free probability to solve for. Real market data won't
        # hit this; it's here so a bad input fails loudly instead of quietly
        # returning a nonsense negative-probability price.
        raise ValueError(
            f"risk-neutral probability {p:.4f} outside [0, 1] - steps too "
            "coarse for these inputs (try more steps or check T/sigma)")

    # terminal stock prices: j down-moves out of `steps` total moves,
    # j = 0 is all-up (top of tree), j = steps is all-down (bottom)
    j = np.arange(steps + 1)
    S_terminal = S * u ** (steps - j) * d ** j

    if option_type == "call":
        values = np.maximum(S_terminal - K, 0.0)
    else:
        values = np.maximum(K - S_terminal, 0.0)

    # walk backward from expiry to today, one step at a time
    for i in range(steps - 1, -1, -1):
        values = disc * (p * values[:-1] + (1.0 - p) * values[1:])

        if exercise == "american":
            j = np.arange(i + 1)
            S_node = S * u ** (i - j) * d ** j
            intrinsic = (np.maximum(S_node - K, 0.0) if option_type == "call"
                        else np.maximum(K - S_node, 0.0))
            values = np.maximum(values, intrinsic)

    return float(values[0])


def early_exercise_premium(option_type, S, K, T, r, sigma, q=0.0, steps=200):
    """American price minus European price on the same tree (same u, d, p,
    so the only difference between the two calls is whether early exercise
    is checked). Always >= 0 - the right to do something extra can't make
    an option worth less.
    """
    american = crr_price(option_type, "american", S, K, T, r, sigma, q, steps)
    european = crr_price(option_type, "european", S, K, T, r, sigma, q, steps)
    return american - european
