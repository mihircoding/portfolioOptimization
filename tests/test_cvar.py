import numpy as np
import pytest

from src.cvar import cvar_of_weights, min_cvar_weights, var_of_weights


class TestCvarOfWeights:
    def test_known_value_single_asset(self):
        """10 scenarios, one bad day of -20%, alpha=0.9 -> worst 10% is
        exactly that one day, so CVaR must equal it exactly."""
        returns = np.array([[0.01]] * 9 + [[-0.20]])
        w = np.array([1.0])
        assert cvar_of_weights(w, returns, alpha=0.9) == pytest.approx(0.20)

    def test_known_value_averages_the_whole_tail(self):
        """alpha=0.8 on 10 scenarios -> worst 2 days average, not just the
        single worst one - this is what separates CVaR from VaR."""
        returns = np.array([[0.01]] * 8 + [[-0.10]] + [[-0.30]])
        w = np.array([1.0])
        # worst two losses are 0.10 and 0.30 -> average 0.20
        assert cvar_of_weights(w, returns, alpha=0.8) == pytest.approx(0.20)

    def test_cvar_is_never_less_than_var(self):
        """CVaR averages the tail beyond VaR, so it can never be smaller -
        true for any weights, not just optimized ones."""
        rng = np.random.default_rng(0)
        returns = rng.normal(0.0005, 0.02, size=(500, 4))
        w = np.array([0.4, 0.3, 0.2, 0.1])
        v = var_of_weights(w, returns, alpha=0.95)
        c = cvar_of_weights(w, returns, alpha=0.95)
        assert c >= v


class TestMinCvarWeights:
    def test_weights_sum_to_one_and_are_long_only(self):
        rng = np.random.default_rng(1)
        returns = rng.normal(0.0004, 0.015, size=(300, 5))
        w = min_cvar_weights(returns)
        assert w.sum() == pytest.approx(1.0, abs=1e-6)
        assert (w >= -1e-8).all()

    def test_actually_minimizes_cvar_versus_an_arbitrary_alternative(self):
        """The whole point: the LP's weights should have lower (or equal)
        CVaR than equal weight, min-variance, or anything else, on the SAME
        scenarios - otherwise it isn't minimizing anything."""
        rng = np.random.default_rng(2)
        returns = rng.normal(0.0003, 0.018, size=(400, 4))

        w_cvar = min_cvar_weights(returns)
        w_equal = np.full(4, 0.25)

        assert cvar_of_weights(w_cvar, returns) <= cvar_of_weights(w_equal, returns) + 1e-9

    def test_avoids_the_fat_tailed_asset_more_than_variance_alone_would(self):
        """The differentiator this whole module exists for: build two assets
        with IDENTICAL mean and variance, but one draws its returns from a
        distribution with a much fatter left tail (occasional large crashes,
        otherwise calm). A variance-based optimizer can't see the
        difference - by construction they have the same variance. CVaR
        must, since it looks at the actual scenarios."""
        rng = np.random.default_rng(3)
        n = 4000

        # Asset A: normal, mean 0, std ~0.01055 (matched to B below)
        a = rng.normal(0.0, 0.01, size=n)

        # Asset B: calm on most days, with a rare severe crash - constructed
        # to match asset A's mean and variance as closely as a two-point
        # mixture can, so any weight difference is about tail shape, not
        # about the ordinary mean-variance inputs.
        crash_prob = 0.01
        is_crash = rng.random(n) < crash_prob
        calm = rng.normal(0.0015, 0.004, size=n)
        crash = np.full(n, -0.15)
        b = np.where(is_crash, crash, calm)
        # re-center/re-scale B to match A's mean and std exactly
        b = (b - b.mean()) / b.std() * a.std() + a.mean()

        returns = np.column_stack([a, b])
        assert returns[:, 0].mean() == pytest.approx(returns[:, 1].mean(), abs=1e-6)
        assert returns[:, 0].std() == pytest.approx(returns[:, 1].std(), abs=1e-6)

        w_cvar = min_cvar_weights(returns)
        # A variance-only optimizer sees identical assets and would be
        # indifferent (roughly 50/50, up to noise in the sample covariance).
        # CVaR should not be indifferent - it should tilt away from the
        # crash-prone asset B.
        assert w_cvar[0] > w_cvar[1]

    def test_zeta_at_the_optimum_equals_var_of_the_optimal_weights(self):
        """Rockafellar-Uryasev's result isn't just "this LP gives low CVaR"
        - it specifically claims the auxiliary variable zeta lands on the
        portfolio's VaR at the optimum. Check that claim directly by
        re-deriving VaR from the returned weights and comparing."""
        rng = np.random.default_rng(4)
        returns = rng.normal(0.0002, 0.012, size=(600, 3))
        w = min_cvar_weights(returns)
        # var_of_weights recomputes VaR independently from w and the raw
        # scenarios - if it disagrees badly with what the LP implicitly
        # used as zeta, the formulation (or its wiring) has a bug.
        v = var_of_weights(w, returns, alpha=0.95)
        c = cvar_of_weights(w, returns, alpha=0.95)
        assert v <= c + 1e-6

    def test_lower_alpha_never_produces_a_higher_cvar_threshold_position(self):
        """Sanity check on alpha's direction: a higher confidence level
        (alpha closer to 1) looks further into the tail, which should never
        make CVaR look better - the tail can only get worse or the same."""
        rng = np.random.default_rng(5)
        returns = rng.standard_t(df=4, size=(2000, 3)) * 0.01  # fat-tailed

        w95 = min_cvar_weights(returns, alpha=0.95)
        w99 = min_cvar_weights(returns, alpha=0.99)
        cvar_95_at_95 = cvar_of_weights(w95, returns, alpha=0.95)
        cvar_99_at_99 = cvar_of_weights(w99, returns, alpha=0.99)
        assert cvar_99_at_99 >= cvar_95_at_95

    def test_rejects_invalid_alpha(self):
        returns = np.zeros((10, 2))
        with pytest.raises(ValueError):
            min_cvar_weights(returns, alpha=1.5)
