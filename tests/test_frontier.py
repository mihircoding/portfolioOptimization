import numpy as np
import pytest

from src.frontier import efficient_frontier
from src.optimizer import min_variance_weights, portfolio_performance


class TestFrontier:
    def test_shape_and_columns(self, three_asset):
        mu, cov = three_asset
        ef = efficient_frontier(mu, cov, n_points=15)
        assert list(ef.columns) == ["target_return", "volatility", "weights"]
        assert len(ef) > 10  # a few may fail to converge; most must succeed

    def test_no_point_beats_min_variance_vol(self, three_asset):
        mu, cov = three_asset
        ef = efficient_frontier(mu, cov, n_points=15)
        w_mv = min_variance_weights(cov)
        _, vol_mv, _ = portfolio_performance(w_mv, mu, cov)
        assert (ef["volatility"] >= vol_mv - 1e-6).all()

    def test_targets_are_hit(self, three_asset):
        mu, cov = three_asset
        ef = efficient_frontier(mu, cov, n_points=15)
        for _, row in ef.iterrows():
            achieved = row["weights"] @ mu
            assert achieved == pytest.approx(row["target_return"], abs=1e-4)

    def test_higher_return_costs_more_vol_above_minvar(self, three_asset):
        """On the upper (efficient) branch, vol rises with target return."""
        mu, cov = three_asset
        ef = efficient_frontier(mu, cov, n_points=20).reset_index(drop=True)
        i_min = ef["volatility"].idxmin()
        upper = ef.iloc[i_min:]
        assert upper["volatility"].is_monotonic_increasing
