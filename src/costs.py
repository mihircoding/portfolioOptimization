"""Transaction costs, drift, and a no-trade band.

Verify with:  pytest tests/test_costs.py

The walk-forward in run_optimization.py and the study in factor_study.py both
score a portfolio by multiplying each day's returns by a fixed weight vector.
That is a portfolio rebalanced back to target every day, for free, and its
turnover is measured from one target to the next as if the book had not moved
in between. This module does the bookkeeping properly:

  - between rebalances the holdings drift with prices (buy and hold),
  - at a rebalance the trade is from the DRIFTED weights to the new ones,
  - every dollar traded pays `cost_bps`.

Conventions:
  - traded notional  = sum |w_new - w_old|        (buys plus sells)
  - one-way turnover = traded notional / 2       (what the tables report)
  - cost             = cost_bps / 1e4 * traded notional

So a rebalance with 20% one-way turnover at 10 bps costs 2 * 0.20 * 10 = 4 bps
of NAV. cost_bps is an all-in, per-side number: half the bid-ask spread plus
commission plus a guess at market impact. The default below is roughly right
for a small book trading liquid US large caps; liquid ETFs like SPY and AGG
cost less, and anything with a real size or a thin name costs more, which is
why the studies report 5, 10 and 25 bps rather than trusting one number.
"""

import numpy as np

DEFAULT_COST_BPS = 10.0
COST_LEVELS_BPS = (5, 10, 25)   # what the studies report, cheap to expensive


def traded_notional(w_from: np.ndarray, w_to: np.ndarray) -> float:
    """Buys plus sells, as a fraction of NAV."""
    return float(np.abs(np.asarray(w_to) - np.asarray(w_from)).sum())


def one_way_turnover(w_from: np.ndarray, w_to: np.ndarray) -> float:
    """Half the traded notional: the share of the book replaced."""
    return traded_notional(w_from, w_to) / 2


def trading_cost(w_from: np.ndarray, w_to: np.ndarray,
                 cost_bps: float = DEFAULT_COST_BPS) -> float:
    """Linear cost of moving from w_from to w_to, as a fraction of NAV."""
    return cost_bps / 1e4 * traded_notional(w_from, w_to)


def drift(w: np.ndarray, returns: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Buy-and-hold `w` through a block of daily returns (T x n).

    Returns (daily portfolio returns, weights at the end of the block). Each
    position compounds at its own asset's return, so winners grow as a share of
    the book - which is exactly the drift a rebalance then has to pay to undo.
    Works for long-short books too: a short position's value is negative and
    compounds the same way.
    """
    growth = np.cumprod(1 + returns, axis=0)             # T x n, per asset
    value = np.vstack([w, w * growth])                   # T+1 x n, per position
    nav = value.sum(axis=1)
    daily = nav[1:] / nav[:-1] - 1
    return daily, value[-1] / nav[-1]


def band_rebalance(w_current: np.ndarray, w_target: np.ndarray,
                   band: float) -> np.ndarray:
    """Trade toward the target only where a weight has drifted past `band`.

    An asset within `band` of its target is left alone. One outside it is
    traded back to the edge of the band, not all the way to the target. That
    is the cheap part of the trade skipped: most of what a rebalance buys is
    small corrections that cost as much per dollar as the big ones.

    Trading each asset to its own band edge would leave the book not fully
    invested, so the targets get one common shift tau, found by bisection, that
    makes the trades net to zero. The result is exactly the solution of

        minimize  1/2 ||w - w_target||^2  +  band * ||w - w_current||_1
        subject to  sum(w) = 1

    i.e. an L1 turnover penalty around the current holding, with the target as
    the anchor. Two properties fall out and are tested: every new weight lies
    between its current and target value (the band never overshoots), and so
    a long-only book stays long-only. band = 0 returns the target exactly.
    """
    w_current = np.asarray(w_current, dtype=float)
    w_target = np.asarray(w_target, dtype=float)
    if band <= 0:
        return w_target.copy()

    desired = w_target - w_current

    def trades(tau: float) -> np.ndarray:
        x = desired - tau
        return np.sign(x) * np.maximum(np.abs(x) - band, 0.0)

    # sum(trades(tau)) is continuous and non-increasing in tau, and changes
    # sign on [-band, band] (both books sum to 1, so desired sums to 0).
    lo, hi = -band, band
    for _ in range(100):
        mid = (lo + hi) / 2
        if trades(mid).sum() > 0:
            lo = mid
        else:
            hi = mid
    w = w_current + trades((lo + hi) / 2)
    return w / w.sum()   # absorbs the ~1e-16 bisection residual


def hold(periods: list[tuple[np.ndarray, np.ndarray]],
         band: float = 0.0) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Run a sequence of (target weights, daily returns for the period).

    At the start of each period the book trades from its drifted weights
    toward the target (all the way, or to the band edge), then holds through
    the period. Costs are not charged here - see `net_returns` - because the
    trades do not depend on the cost level, so one run serves every cost.

    Returns (gross daily returns for the whole run, traded notional at each
    rebalance, index of the first day after each rebalance). The initial
    purchase is not counted: every strategy pays it once, and it says nothing
    about rebalancing.
    """
    daily, traded, starts = [], [], []
    w, day = None, 0
    for target, block in periods:
        if w is None:
            new = np.asarray(target, dtype=float)
        else:
            new = band_rebalance(w, target, band)
            traded.append(traded_notional(w, new))
            starts.append(day)
        r, w = drift(new, block)
        daily.append(r)
        day += len(block)
    return np.concatenate(daily), np.asarray(traded), starts


def net_returns(gross: np.ndarray, traded: np.ndarray, period_starts: list[int],
                cost_bps: float = DEFAULT_COST_BPS) -> np.ndarray:
    """Charge each rebalance's cost against the first day of its period.

    `period_starts[k]` is the index in `gross` of the first day held after
    rebalance k, as returned by `hold`. The cost is taken from
    NAV before that day's return, pro rata across positions, which leaves the
    weights - and so every later gross return - unchanged. With cost_bps = 0
    this returns `gross` exactly.
    """
    net = np.array(gross, dtype=float, copy=True)
    for start, notional in zip(period_starts, traded):
        net[start] = (1 + net[start]) * (1 - cost_bps / 1e4 * notional) - 1
    return net
