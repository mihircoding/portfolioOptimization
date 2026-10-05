"""Tests for charging the option's bid-ask on the monthly sale.

Run with:  python -m pytest tests/volcarry/test_option_spread.py -q

The sleeve used to sell its option at the mid while charging bid-ask on every
share the hedge traded, which made it the cheaper of the two sleeves the
allocator compares for a reason that was an omission rather than a result.
`option_spread_vol_pts` closes that, and the claim it makes is narrow: the
option is sold at the BID and hedged at the MID, so the premium falls and
nothing else about the trade changes.

Narrow claims are the testable kind. These run on a constructed price path
rather than a download, so they check the wiring instead of recording a
sample - and the cost is checked against the vega arithmetic quoted in
hedging_study, so the parameter stays something a reader can verify rather
than a number to take on trust.
"""

import numpy as np
import pytest

from sleeves.volcarry.black_scholes import price, vega
from sleeves.volcarry.hedging_study import HEDGE_DAYS, RATE, T_YEARS, trade

import pandas as pd

SPOT = 400.0
VIX = 18.0


def window(drift: float = 0.0, seed: int = 0) -> pd.DataFrame:
    """One month of daily prices, plus a VIX column the trade reads once."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(drift, 0.008, HEDGE_DAYS)
    path = SPOT * np.exp(np.cumsum(np.insert(steps, 0, 0.0)))
    idx = pd.bdate_range("2020-01-01", periods=HEDGE_DAYS + 1)
    return pd.DataFrame({"spy": path, "vix": VIX}, index=idx)


def test_no_spread_reproduces_the_mid_market_trade_exactly():
    """The default has to be a no-op, or every number predating it moves."""
    w = window()
    mid = trade(w)
    explicit = trade(w, option_spread_vol_pts=0.0)

    assert mid["pnl"] == pytest.approx(explicit["pnl"])
    assert mid["spread_cost"] == pytest.approx(0.0)


def test_selling_at_the_bid_costs_exactly_the_premium_it_gives_up():
    """P&L falls by the reported spread cost and by nothing else.

    This is what pins "sold at the bid, hedged at the mid". If the spread
    leaked into the hedge, the two would differ by the hedge's P&L as well
    and this would not hold to a cent.
    """
    w = window()
    mid = trade(w, option_spread_vol_pts=0.0)
    charged = trade(w, option_spread_vol_pts=0.25)

    assert (mid["pnl"] - charged["pnl"]
            == pytest.approx(charged["spread_cost"], rel=1e-9))


def test_the_hedge_is_run_on_the_mid_not_on_the_bid():
    """Everything the hedge produces is identical; only the premium moves.

    A desk marks and hedges on its mid even though it crossed the spread to
    get the position on, and the two volatilities were already separate
    arguments to delta_hedge. If someone later wires the bid into the hedge
    too, gamma P&L and hedging costs will move and this will catch it.
    """
    w = window()
    mid = trade(w, option_spread_vol_pts=0.0)
    charged = trade(w, option_spread_vol_pts=0.50)

    assert charged["gamma_pnl"] == pytest.approx(mid["gamma_pnl"])
    assert charged["costs"] == pytest.approx(mid["costs"])
    assert charged["realized"] == pytest.approx(mid["realized"])
    assert charged["implied"] == pytest.approx(mid["implied"])
    assert charged["sold_at"] < mid["sold_at"]


def test_the_spread_cost_is_the_vega_arithmetic_in_the_docstring():
    """0.10 vol points should cost about vega/100 per $100 of underlying.

    hedging_study justifies its default by saying one-month ATM vega is about
    $0.67 a vol point on a $580 underlying. That claim is checkable, so it is
    checked - a parameter defended by arithmetic nobody verified is a
    parameter defended by nothing.

    Tolerance is 2%: vega is the derivative at the mid and the cost is a
    finite difference across the half-spread, so they agree to second order
    rather than exactly.
    """
    w = window()
    pts = 0.10
    charged = trade(w, option_spread_vol_pts=pts)

    scale = 100.0 / SPOT
    # black_scholes.vega is already quoted per one percentage point of vol,
    # which is the same unit option_spread_vol_pts is in - so no /100 here.
    expected = vega(SPOT, SPOT, T_YEARS, RATE, VIX / 100.0) * pts

    assert charged["spread_cost"] == pytest.approx(expected * scale, rel=0.02)


@pytest.mark.parametrize("seed", [0, 1, 2, 3])
def test_a_wider_spread_never_helps(seed):
    """Monotone in the spread, on paths that went up, down and sideways."""
    w = window(seed=seed)
    pnls = [trade(w, option_spread_vol_pts=p)["pnl"]
            for p in (0.0, 0.10, 0.25, 1.00)]

    assert pnls == sorted(pnls, reverse=True)
