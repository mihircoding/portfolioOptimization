"""
Shared heatmap plotting helper.

Plotly (instead of a static matplotlib image) is used for the heatmaps
for one reason: hovering. When you move the mouse over a cell, plotly
pops a tooltip with the exact spot / vol / value for that cell, which is
what makes the grid actually explorable. Matplotlib is still used for
the strategy payoff diagrams where hovering matters less.
"""

import numpy as np
import plotly.graph_objects as go


def price_grid(pricer, spot_range, vol_range, K, T, r):
    """
    Build the 2D matrix behind a heatmap: rows are volatilities, columns
    are spot prices, each cell is the Black-Scholes price at that combo.
    `pricer` is black_scholes.call_price or black_scholes.put_price.
    """
    grid = np.zeros((len(vol_range), len(spot_range)))
    for i, sigma in enumerate(vol_range):
        grid[i, :] = pricer(spot_range, K, T, r, sigma)
    return grid


def heatmap_figure(grid, spot_range, vol_range, title,
                   pnl_mode=False, value_label="Price"):
    """
    Render a grid as an annotated heatmap.

    pnl_mode=False -> plain price map, Viridis colors (dark = cheap,
                      bright = expensive), like the classic version.
    pnl_mode=True  -> profit/loss map, RdYlGn colors centered on zero so
                      losses are red and profits are green at a glance.
    """
    x_labels = [f"{s:.2f}" for s in spot_range]
    y_labels = [f"{v:.2f}" for v in vol_range]
    cell_text = [[f"{val:.2f}" for val in row] for row in grid]

    if pnl_mode:
        # symmetric range around 0 so the yellow midpoint of RdYlGn
        # lands exactly on break-even
        bound = max(abs(grid.min()), abs(grid.max()), 1e-9)
        colorscale, zmin, zmax = "RdYlGn", -bound, bound
    else:
        colorscale, zmin, zmax = "Viridis", grid.min(), grid.max()

    fig = go.Figure(go.Heatmap(
        z=grid,
        x=x_labels,
        y=y_labels,
        colorscale=colorscale,
        zmin=zmin,
        zmax=zmax,
        text=cell_text,
        texttemplate="%{text}",
        textfont={"size": 11},
        xgap=1.5,
        ygap=1.5,
        hovertemplate=(
            "Spot: %{x}<br>Volatility: %{y}<br>"
            + value_label + ": %{z:.2f}<extra></extra>"
        ),
        hoverlabel={"bgcolor": "white", "font_size": 14},
        colorbar={"title": value_label},
    ))
    fig.update_layout(
        title=title,
        xaxis_title="Spot Price",
        yaxis_title="Volatility",
        height=520,
        margin={"l": 60, "r": 20, "t": 50, "b": 60},
    )
    return fig
