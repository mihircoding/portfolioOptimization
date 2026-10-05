"""
Multi-leg option strategies.

Every strategy is just a list of "legs". A leg is a dict:
    kind   - 'call', 'put' or 'stock'
    strike - strike price (ignored for stock)
    qty    - +1 for long (you bought it), -1 for short (you sold it),
             -2 for two shorts, etc.

With that one representation we only need two generic functions:
    strategy_cost()   - what the position costs to open today (Black-Scholes
                        for the option legs, spot price for stock legs)
    strategy_payoff() - what the position is worth at expiry for a range
                        of final stock prices

PnL at expiry = payoff - cost. That single line drives every chart and
heatmap on the strategies page.
"""

import numpy as np

import black_scholes as bs


def _leg(kind, strike, qty):
    return {"kind": kind, "strike": strike, "qty": qty}


def build_strategy(name, S, K):
    """
    Build the legs for a named strategy around current spot S and a
    "central" strike K. Wing strikes are placed at sensible fixed
    percentages of K so every strategy works for any stock price.
    Returns (legs, one-line description).
    """
    K = float(K)
    lo, hi = round(K * 0.90, 2), round(K * 1.10, 2)      # 10% wings
    lo2, hi2 = round(K * 0.80, 2), round(K * 1.20, 2)    # 20% wings

    strategies = {
        "Covered Call": (
            [_leg("stock", None, 1), _leg("call", hi, -1)],
            f"Own 100 shares, sell a {hi} call against them. You collect the "
            "premium but give up any upside past the strike.",
        ),
        "Married Put": (
            [_leg("stock", None, 1), _leg("put", lo, 1)],
            f"Own 100 shares, buy a {lo} put as insurance. Losses are capped "
            "at the put strike, minus the premium you paid.",
        ),
        "Bull Call Spread": (
            [_leg("call", K, 1), _leg("call", hi, -1)],
            f"Buy the {K} call, sell the {hi} call. Cheaper than a naked "
            "call, but profit is capped at the upper strike.",
        ),
        "Bear Put Spread": (
            [_leg("put", K, 1), _leg("put", lo, -1)],
            f"Buy the {K} put, sell the {lo} put. A defined-risk bet that "
            "the stock falls, with profit capped at the lower strike.",
        ),
        "Protective Collar": (
            [_leg("stock", None, 1), _leg("put", lo, 1), _leg("call", hi, -1)],
            f"Own shares, buy the {lo} put, fund it by selling the {hi} "
            "call. Cheap downside protection in exchange for capped upside.",
        ),
        "Long Straddle": (
            [_leg("call", K, 1), _leg("put", K, 1)],
            f"Buy the {K} call AND the {K} put. Profits from a big move in "
            "either direction; loses the double premium if nothing happens.",
        ),
        "Long Strangle": (
            [_leg("call", hi, 1), _leg("put", lo, 1)],
            f"Buy the {hi} call and the {lo} put. Like a straddle but "
            "cheaper - needs an even bigger move to pay off.",
        ),
        "Long Call Butterfly": (
            [_leg("call", lo, 1), _leg("call", K, -2), _leg("call", hi, 1)],
            f"Buy {lo} call, sell two {K} calls, buy {hi} call. Max profit "
            "if the stock pins the middle strike at expiry.",
        ),
        "Iron Condor": (
            [_leg("put", lo2, 1), _leg("put", lo, -1),
             _leg("call", hi, -1), _leg("call", hi2, 1)],
            f"Sell the {lo} put and {hi} call, buy the {lo2} put and {hi2} "
            "call as wings. You keep the credit if the stock stays in the "
            "middle range.",
        ),
        "Iron Butterfly": (
            [_leg("put", lo, 1), _leg("put", K, -1),
             _leg("call", K, -1), _leg("call", hi, 1)],
            f"Sell the {K} straddle, buy the {lo} put / {hi} call as wings. "
            "Bigger credit than a condor but a much narrower profit zone.",
        ),
    }
    return strategies[name]


STRATEGY_NAMES = [
    "Covered Call", "Married Put", "Bull Call Spread", "Bear Put Spread",
    "Protective Collar", "Long Straddle", "Long Strangle",
    "Long Call Butterfly", "Iron Condor", "Iron Butterfly",
]


def strategy_cost(legs, S, T, r, sigma):
    """
    Net cost to open the position today. Long legs cost money (positive),
    short legs bring in premium (negative). A negative total means you
    open the trade for a CREDIT (e.g. iron condor).
    """
    total = 0.0
    for leg in legs:
        if leg["kind"] == "stock":
            total += leg["qty"] * S
        else:
            total += leg["qty"] * float(
                bs.price(leg["kind"], S, leg["strike"], T, r, sigma)
            )
    return total


def strategy_payoff(legs, spot_grid):
    """
    Value of the position at expiry for each final price in spot_grid.
    At expiry the options are worth exactly their intrinsic value:
    call -> max(S-K, 0), put -> max(K-S, 0).
    """
    spot_grid = np.asarray(spot_grid, dtype=float)
    payoff = np.zeros_like(spot_grid)
    for leg in legs:
        if leg["kind"] == "stock":
            payoff += leg["qty"] * spot_grid
        elif leg["kind"] == "call":
            payoff += leg["qty"] * np.maximum(spot_grid - leg["strike"], 0.0)
        else:  # put
            payoff += leg["qty"] * np.maximum(leg["strike"] - spot_grid, 0.0)
    return payoff


def strategy_value(legs, spot_grid, T, r, sigma):
    """
    Mark-to-model value of the position BEFORE expiry (time T left),
    pricing each option leg with Black-Scholes at every spot in the grid.
    Used for the strategy PnL heatmap over spot x volatility.
    """
    spot_grid = np.asarray(spot_grid, dtype=float)
    value = np.zeros_like(spot_grid)
    for leg in legs:
        if leg["kind"] == "stock":
            value += leg["qty"] * spot_grid
        else:
            value += leg["qty"] * bs.price(
                leg["kind"], spot_grid, leg["strike"], T, r, sigma
            )
    return value
