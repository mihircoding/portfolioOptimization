import numpy as np
import pytest
from scipy.optimize import minimize

from src.costs import (band_rebalance, drift, hold, net_returns, one_way_turnover,
                       trading_cost)


@pytest.fixture
def periods():
    """Six quarters of synthetic daily returns on four assets, with a new
    random long-only target each quarter."""
    rng = np.random.default_rng(7)
    out = []
    for _ in range(6):
        target = rng.dirichlet(np.ones(4))
        block = rng.normal(0.0004, 0.012, size=(63, 4))
        out.append((target, block))
    return out


class TestCostArithmetic:
    def test_docstring_example(self):
        # 20% one-way turnover at 10 bps costs 2 * 0.20 * 10 = 4 bps
        w0 = np.array([0.5, 0.5, 0.0])
        w1 = np.array([0.3, 0.5, 0.2])
        assert one_way_turnover(w0, w1) == pytest.approx(0.20)
        assert trading_cost(w0, w1, cost_bps=10) == pytest.approx(4e-4)

    def test_no_trade_costs_nothing(self):
        w = np.array([0.2, 0.3, 0.5])
        assert trading_cost(w, w, cost_bps=25) == 0.0

    def test_cost_is_linear_in_size_and_rate(self):
        w0 = np.array([0.25, 0.25, 0.25, 0.25])
        step = np.array([0.1, -0.1, 0.05, -0.05])
        small = trading_cost(w0, w0 + step, 10)
        assert trading_cost(w0, w0 + 2 * step, 10) == pytest.approx(2 * small)
        assert trading_cost(w0, w0 + step, 20) == pytest.approx(2 * small)

    def test_cost_monotone_in_turnover(self):
        w0 = np.full(5, 0.2)
        target = np.array([0.6, 0.1, 0.1, 0.1, 0.1])
        costs = [trading_cost(w0, w0 + t * (target - w0), 10)
                 for t in np.linspace(0, 1, 11)]
        assert np.all(np.diff(costs) > 0)


class TestDrift:
    def test_matches_position_by_position_compounding(self):
        rng = np.random.default_rng(1)
        w = np.array([0.5, 0.3, 0.2])
        r = rng.normal(0.0005, 0.01, size=(40, 3))
        daily, w_end = drift(w, r)
        values = w * np.prod(1 + r, axis=0)
        assert np.prod(1 + daily) == pytest.approx(values.sum())
        np.testing.assert_allclose(w_end, values / values.sum())

    def test_winner_becomes_overweight(self):
        r = np.zeros((10, 2))
        r[:, 0] = 0.01
        _, w_end = drift(np.array([0.5, 0.5]), r)
        assert w_end[0] > 0.5 and w_end.sum() == pytest.approx(1.0)

    def test_long_short_book(self):
        # a short leg compounds too: if the shorted asset rises, the book loses
        w = np.array([1.5, -0.5])
        r = np.array([[0.0, 0.10]])
        daily, _ = drift(w, r)
        assert daily[0] == pytest.approx(-0.05)


class TestBandRebalance:
    def test_zero_band_is_the_target(self):
        cur = np.array([0.4, 0.35, 0.25])
        tgt = np.array([0.2, 0.3, 0.5])
        np.testing.assert_array_equal(band_rebalance(cur, tgt, 0.0), tgt)

    def test_trades_back_to_band_edge(self):
        w = band_rebalance(np.array([0.5, 0.5]), np.array([0.6, 0.4]), 0.02)
        np.testing.assert_allclose(w, [0.58, 0.42])

    def test_asset_inside_band_is_left_alone(self):
        cur = np.array([0.30, 0.30, 0.40])
        tgt = np.array([0.40, 0.20, 0.40])
        w = band_rebalance(cur, tgt, 0.03)
        assert w[2] == pytest.approx(0.40)      # already on target: untouched
        np.testing.assert_allclose(w, [0.37, 0.23, 0.40])

    def test_huge_band_means_no_trade(self):
        cur = np.array([0.1, 0.6, 0.3])
        tgt = np.array([0.5, 0.2, 0.3])
        np.testing.assert_allclose(band_rebalance(cur, tgt, 10.0), cur)

    def test_never_overshoots_and_stays_long_only(self):
        rng = np.random.default_rng(3)
        for _ in range(200):
            cur, tgt = rng.dirichlet(np.ones(8)), rng.dirichlet(np.ones(8))
            w = band_rebalance(cur, tgt, rng.uniform(0.001, 0.1))
            assert w.sum() == pytest.approx(1.0)
            lo, hi = np.minimum(cur, tgt), np.maximum(cur, tgt)
            assert np.all(w >= lo - 1e-12) and np.all(w <= hi + 1e-12)
            assert np.all(w >= -1e-12)

    def test_turnover_falls_as_band_widens(self):
        rng = np.random.default_rng(5)
        cur, tgt = rng.dirichlet(np.ones(10)), rng.dirichlet(np.ones(10))
        turns = [one_way_turnover(cur, band_rebalance(cur, tgt, b))
                 for b in (0, 0.005, 0.01, 0.02, 0.05, 0.2)]
        assert np.all(np.diff(turns) <= 1e-12)
        assert turns[-1] == pytest.approx(0.0)

    def test_solves_the_stated_l1_problem(self):
        # min 1/2 ||w - tgt||^2 + band * ||w - cur||_1  s.t. sum(w) = 1,
        # solved independently with the trade split into buys and sells
        rng = np.random.default_rng(11)
        n, band = 6, 0.03
        cur = rng.dirichlet(np.ones(n))
        tgt = rng.normal(1 / n, 0.1, n)
        tgt /= tgt.sum()                     # long-short target is fine too

        def objective(x):
            buy, sell = x[:n], x[n:]
            w = cur + buy - sell
            return 0.5 * np.sum((w - tgt) ** 2) + band * np.sum(buy + sell)

        res = minimize(objective, np.zeros(2 * n), method="SLSQP",
                       bounds=[(0, None)] * (2 * n),
                       constraints=[{"type": "eq",
                                     "fun": lambda x: np.sum(x[:n] - x[n:])}],
                       options={"ftol": 1e-14, "maxiter": 1000})
        w_ref = cur + res.x[:n] - res.x[n:]
        np.testing.assert_allclose(band_rebalance(cur, tgt, band), w_ref, atol=1e-5)


class TestHoldAndNet:
    def test_zero_cost_net_equals_gross(self, periods):
        gross, traded, starts = hold(periods)
        np.testing.assert_array_equal(net_returns(gross, traded, starts, 0.0), gross)

    def test_zero_band_is_plain_rebalancing(self, periods):
        # band=0 must be exactly "trade to target every period"
        gross, traded, _ = hold(periods, band=0.0)
        expected, notional, w = [], [], None
        for target, block in periods:
            if w is not None:
                notional.append(np.abs(target - w).sum())
            r, w = drift(target, block)
            expected.append(r)
        np.testing.assert_allclose(gross, np.concatenate(expected))
        np.testing.assert_allclose(traded, notional)

    def test_huge_band_never_trades_after_the_first_buy(self, periods):
        gross, traded, _ = hold(periods, band=10.0)
        assert np.all(traded < 1e-12)
        buy_and_hold, _ = drift(periods[0][0], np.vstack([b for _, b in periods]))
        np.testing.assert_allclose(gross, buy_and_hold)

    def test_initial_purchase_not_charged(self, periods):
        _, traded, starts = hold(periods)
        assert len(traded) == len(periods) - 1
        assert starts[0] == len(periods[0][1])

    def test_net_growth_is_gross_times_cost_factors(self, periods):
        gross, traded, starts = hold(periods)
        net = net_returns(gross, traded, starts, 25)
        expected = np.prod(1 + gross) * np.prod(1 - 25e-4 * traded)
        assert np.prod(1 + net) == pytest.approx(expected)

    def test_net_falls_as_cost_rises(self, periods):
        gross, traded, starts = hold(periods)
        growth = [np.prod(1 + net_returns(gross, traded, starts, c))
                  for c in (0, 5, 10, 25, 50)]
        assert np.all(np.diff(growth) < 0)

    def test_band_reduces_traded_notional(self, periods):
        _, full, _ = hold(periods, band=0.0)
        _, banded, _ = hold(periods, band=0.03)
        assert banded.sum() < full.sum()
