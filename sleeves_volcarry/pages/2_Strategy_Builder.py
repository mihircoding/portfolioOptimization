"""
Page 2: Strategy builder.

Pick one of ten classic multi-leg strategies and see:
  - the legs and what the position costs (debit) or pays you (credit)
  - the payoff diagram at expiry, green where you profit, red where you lose
  - a PnL heatmap over spot x volatility at a chosen check-in date
    (because before expiry, volatility still moves your position's value)
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import black_scholes as bs
import heatmaps
import strategies as strat

st.set_page_config(page_title="Strategy Builder", layout="wide")

# ---------------------------------------------------------------------------
# Sidebar inputs
# ---------------------------------------------------------------------------
st.sidebar.header("Market Inputs")

S = st.sidebar.number_input(
    "Spot Price ($)", min_value=0.01, value=100.0, step=1.0,
    help="Current stock price.",
)
K = st.sidebar.number_input(
    "Central Strike ($)", min_value=0.01, value=100.0, step=1.0,
    help="The main strike the strategy is built around. Wing strikes are "
         "placed 10% and 20% away automatically.",
)
T_days = st.sidebar.number_input(
    "Days to Expiry", min_value=1, value=90, step=1,
    help="Lifetime of every option leg.",
)
r = st.sidebar.number_input(
    "Risk-Free Rate", min_value=0.0, max_value=0.25, value=0.05,
    step=0.005, format="%.3f",
    help="Annualized risk-free rate as a decimal.",
)
sigma = st.sidebar.slider(
    "Volatility", 0.01, 1.5, 0.25, 0.01,
    help="Volatility used to price the legs when you OPEN the trade.",
)

st.sidebar.divider()
name = st.sidebar.selectbox("Strategy", strat.STRATEGY_NAMES)

check_days = st.sidebar.slider(
    "Days until you check the PnL heatmap", 0, int(T_days), int(T_days) // 2,
    help="The heatmap marks the position to model at this point in its "
         "life. At 0 days the trade just opened; at expiry only the spot "
         "price matters and volatility stops having any effect.",
)

# ---------------------------------------------------------------------------
# Build the position
# ---------------------------------------------------------------------------
T = T_days / 365.0
legs, description = strat.build_strategy(name, S, K)
cost = strat.strategy_cost(legs, S, T, r, sigma)

st.title(name)
st.info(description)

# Legs table + entry cost
rows = []
for leg in legs:
    if leg["kind"] == "stock":
        rows.append(["Stock", "-", leg["qty"], f"{S:.2f}"])
    else:
        unit = float(bs.price(leg["kind"], S, leg["strike"], T, r, sigma))
        rows.append([leg["kind"].capitalize(), f"{leg['strike']:.2f}",
                     leg["qty"], f"{unit:.2f}"])
legs_df = pd.DataFrame(rows, columns=["Type", "Strike", "Qty", "Unit Price"])

c1, c2 = st.columns([2, 1])
c1.table(legs_df)
if cost >= 0:
    c2.metric("Net Debit (you pay)", f"${cost:.2f}")
else:
    c2.metric("Net Credit (you receive)", f"${-cost:.2f}")
c2.caption("Positive qty = long (bought), negative = short (sold). "
           "Legs are priced with Black-Scholes at today's inputs.")

# ---------------------------------------------------------------------------
# Payoff at expiry - green above zero, red below
# ---------------------------------------------------------------------------
st.subheader("Profit / Loss at Expiry")

spots = np.linspace(S * 0.5, S * 1.5, 400)
pnl = strat.strategy_payoff(legs, spots) - cost

fig, ax = plt.subplots(figsize=(9, 4.5))
ax.plot(spots, pnl, color="black", linewidth=1.5)
ax.fill_between(spots, pnl, 0, where=pnl >= 0, color="green", alpha=0.35)
ax.fill_between(spots, pnl, 0, where=pnl < 0, color="red", alpha=0.35)
ax.axhline(0, color="gray", linewidth=0.8)
ax.axvline(S, color="blue", linestyle="--", linewidth=0.9, label=f"Spot {S:.0f}")
ax.set_xlabel("Stock Price at Expiry")
ax.set_ylabel("Profit / Loss ($)")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)

be_crossings = spots[np.where(np.diff(np.sign(pnl)) != 0)[0]]
m1, m2, m3 = st.columns(3)
m1.metric("Max Profit (in range shown)", f"${pnl.max():.2f}")
m2.metric("Max Loss (in range shown)", f"${pnl.min():.2f}")
m3.metric("Break-even(s)",
          ", ".join(f"{b:.2f}" for b in be_crossings) if len(be_crossings) else "none in range")

# ---------------------------------------------------------------------------
# PnL heatmap over spot x volatility, marked to model before expiry
# ---------------------------------------------------------------------------
st.subheader(f"PnL Heatmap - {check_days} days after opening")

T_left = max(T_days - check_days, 0) / 365.0
spot_range = np.linspace(S * 0.8, S * 1.2, 10)
vol_range = np.linspace(max(sigma - 0.15, 0.05), sigma + 0.25, 10)

grid = np.zeros((len(vol_range), len(spot_range)))
for i, v in enumerate(vol_range):
    grid[i, :] = strat.strategy_value(legs, spot_range, T_left, r, v) - cost

st.plotly_chart(
    heatmaps.heatmap_figure(
        grid, spot_range, vol_range,
        f"{name} PnL (marked to model, {max(T_days - check_days, 0)} days left)",
        pnl_mode=True, value_label="PnL",
    ),
    width='stretch',
)
st.caption(
    "Each cell: if the stock were at this spot and the market expected this "
    "volatility on the check-in date, the position would be worth this much "
    "more (green) or less (red) than you paid. Slide 'days until you check' "
    "to expiry and volatility stops mattering - only intrinsic value is left."
)
