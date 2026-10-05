"""
Page 1: Black-Scholes pricer + interactive heatmaps.

Layout mirrors the classic options-heatmap project:
  - sidebar holds every model input
  - top of the page shows the current call/put fair value (green/red)
  - below that, price heatmaps over a spot x volatility grid
  - below those, PnL heatmaps (model value minus what you paid)
  - greeks table at the bottom
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import black_scholes as bs
import heatmaps

st.set_page_config(page_title="Options Pricer", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar - all model inputs. The help= text on each widget is the little
# (?) blurb explaining what the parameter does.
# ---------------------------------------------------------------------------
st.sidebar.header("Model Inputs")

S = st.sidebar.number_input(
    "Spot Price ($)", min_value=0.01, value=100.0, step=1.0,
    help="Current market price of the underlying stock.",
)
K = st.sidebar.number_input(
    "Strike Price ($)", min_value=0.01, value=100.0, step=1.0,
    help="Price at which the option lets you buy (call) or sell (put) "
         "the stock at expiry.",
)
T_days = st.sidebar.number_input(
    "Time to Maturity (days)", min_value=1, value=365, step=1,
    help="Days until the option expires. Converted to years (days/365) "
         "inside the model. More time = more expensive option.",
)
r = st.sidebar.number_input(
    "Risk-Free Interest Rate", min_value=0.0, max_value=0.25,
    value=0.05, step=0.005, format="%.3f",
    help="Annualized rate on a 'safe' asset like T-bills, as a decimal "
         "(0.05 = 5%). Higher rates make calls dearer and puts cheaper.",
)
sigma = st.sidebar.slider(
    "Volatility (sigma)", min_value=0.01, max_value=1.5, value=0.20, step=0.01,
    help="Annualized standard deviation of the stock's returns "
         "(0.20 = 20%). The single biggest driver of option value.",
)

st.sidebar.divider()
st.sidebar.header("Heatmap Parameters")

min_spot = st.sidebar.number_input(
    "Min Spot Price", min_value=0.01, value=round(S * 0.8, 2),
    help="Left edge of the heatmap's spot-price axis.",
)
max_spot = st.sidebar.number_input(
    "Max Spot Price", min_value=0.02, value=round(S * 1.2, 2),
    help="Right edge of the heatmap's spot-price axis.",
)
min_vol, max_vol = st.sidebar.slider(
    "Volatility Range for Heatmap", 0.01, 1.5, (0.10, 0.60), step=0.01,
    help="Bottom and top of the heatmap's volatility axis.",
)
grid_n = st.sidebar.slider(
    "Grid Size", 5, 15, 10,
    help="Number of rows/columns in each heatmap.",
)

st.sidebar.divider()
st.sidebar.header("Your Trade (for PnL)")

call_paid = st.sidebar.number_input(
    "Call Purchase Price ($)", min_value=0.0, value=10.0, step=0.5,
    help="What you actually paid for the call. The PnL heatmap shows "
         "model value minus this number.",
)
put_paid = st.sidebar.number_input(
    "Put Purchase Price ($)", min_value=0.0, value=10.0, step=0.5,
    help="What you actually paid for the put.",
)

# ---------------------------------------------------------------------------
# Current fair value under the inputs above
# ---------------------------------------------------------------------------
T = T_days / 365.0
call_val = float(bs.call_price(S, K, T, r, sigma))
put_val = float(bs.put_price(S, K, T, r, sigma))

st.title("Options Price - Interactive Heatmap")
st.info(
    "Explore how option prices fluctuate with varying spot prices and "
    "volatility levels, while holding strike price, rate and expiry "
    "constant. Hover any cell for exact values."
)

col1, col2 = st.columns(2)
col1.markdown(
    f"<div style='background:#98e698;padding:14px;border-radius:8px;"
    f"text-align:center;font-size:26px;font-weight:bold;color:#111'>"
    f"CALL: ${call_val:.2f}</div>", unsafe_allow_html=True,
)
col2.markdown(
    f"<div style='background:#ffb3b3;padding:14px;border-radius:8px;"
    f"text-align:center;font-size:26px;font-weight:bold;color:#111'>"
    f"PUT: ${put_val:.2f}</div>", unsafe_allow_html=True,
)

with st.expander("What do the heatmap parameters mean?"):
    st.markdown("""
- **Spot price (x-axis)** - the stock price. Each column asks *"what would
  the option be worth if the stock were trading here?"*
- **Volatility (y-axis)** - how wildly the stock moves. Each row asks
  *"what if the market expected this much movement?"* More volatility means
  more chance of finishing deep in the money, so BOTH calls and puts get
  more expensive as you go down the rows.
- **Price heatmaps** - raw Black-Scholes fair value. Dark = cheap,
  bright = expensive.
- **PnL heatmaps** - fair value **minus what you paid** (set in the
  sidebar). Green cells = combinations of spot/vol where you'd be up money,
  red = down. Yellow is roughly break-even.
""")

# ---------------------------------------------------------------------------
# Build the grids once, reuse for price + PnL maps
# ---------------------------------------------------------------------------
spot_range = np.linspace(min_spot, max_spot, grid_n)
vol_range = np.linspace(min_vol, max_vol, grid_n)

call_grid = heatmaps.price_grid(bs.call_price, spot_range, vol_range, K, T, r)
put_grid = heatmaps.price_grid(bs.put_price, spot_range, vol_range, K, T, r)

st.subheader("Fair Value")
c1, c2 = st.columns(2)
with c1:
    st.plotly_chart(
        heatmaps.heatmap_figure(call_grid, spot_range, vol_range, "CALL"),
        width='stretch',
    )
with c2:
    st.plotly_chart(
        heatmaps.heatmap_figure(put_grid, spot_range, vol_range, "PUT"),
        width='stretch',
    )

st.subheader("PnL (model value - your purchase price)")
p1, p2 = st.columns(2)
with p1:
    st.plotly_chart(
        heatmaps.heatmap_figure(
            call_grid - call_paid, spot_range, vol_range,
            f"CALL PnL (paid ${call_paid:.2f})",
            pnl_mode=True, value_label="PnL",
        ),
        width='stretch',
    )
with p2:
    st.plotly_chart(
        heatmaps.heatmap_figure(
            put_grid - put_paid, spot_range, vol_range,
            f"PUT PnL (paid ${put_paid:.2f})",
            pnl_mode=True, value_label="PnL",
        ),
        width='stretch',
    )

# ---------------------------------------------------------------------------
# Greeks for the current inputs
# ---------------------------------------------------------------------------
st.subheader("Greeks at Current Inputs")
g1, g2 = st.columns(2)
with g1:
    st.markdown("**Call**")
    st.table(
        {k: [round(v, 4)] for k, v in
         bs.all_greeks("call", S, K, T, r, sigma).items()}
    )
with g2:
    st.markdown("**Put**")
    st.table(
        {k: [round(v, 4)] for k, v in
         bs.all_greeks("put", S, K, T, r, sigma).items()}
    )

st.caption(
    "Delta: $ change per $1 move in the stock. Gamma: change in delta per "
    "$1 move. Vega: $ change per 1 point of volatility. Theta: $ change "
    "per day that passes. Rho: $ change per 1 point of interest rate."
)

# ---------------------------------------------------------------------------
# Greeks across spot - the table above is one point; these curves show the
# whole shape, which is where the intuition lives.
# ---------------------------------------------------------------------------
st.subheader("How the Greeks change as the stock moves")

spot_axis = np.linspace(min_spot, max_spot, 200)
fig, axes = plt.subplots(2, 2, figsize=(11, 7))

axes[0, 0].plot(spot_axis, bs.delta("call", spot_axis, K, T, r, sigma),
                color="green", label="call")
axes[0, 0].plot(spot_axis, bs.delta("put", spot_axis, K, T, r, sigma),
                color="red", label="put")
axes[0, 0].set_title("Delta")
axes[0, 0].legend()

axes[0, 1].plot(spot_axis, bs.gamma(spot_axis, K, T, r, sigma),
                color="purple")
axes[0, 1].set_title("Gamma (same for call & put)")

axes[1, 0].plot(spot_axis, bs.vega(spot_axis, K, T, r, sigma),
                color="steelblue")
axes[1, 0].set_title("Vega (same for call & put)")

axes[1, 1].plot(spot_axis, bs.theta("call", spot_axis, K, T, r, sigma),
                color="green", label="call")
axes[1, 1].plot(spot_axis, bs.theta("put", spot_axis, K, T, r, sigma),
                color="red", label="put")
axes[1, 1].set_title("Theta ($ per day)")
axes[1, 1].legend()

for ax in axes.flat:
    ax.axvline(K, color="gray", linestyle="--", linewidth=0.8)
    ax.grid(alpha=0.3)
    ax.set_xlabel("Spot price")
fig.tight_layout()
st.pyplot(fig)
plt.close(fig)

st.caption(
    "Dashed line = strike. Notice how gamma and vega both peak at the "
    "money: that's where the next dollar of stock movement matters most, "
    "and where theta decay is deepest - at-the-money options are the most "
    "'alive' and pay for it in daily time decay."
)
