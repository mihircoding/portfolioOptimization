"""
Page 4: Monte Carlo vs. Black-Scholes.

Prices the same option two independent ways:
  - the closed-form Black-Scholes formula (instant, exact)
  - brute-force simulation of thousands of random price paths

Watching the simulation estimate converge onto the formula is the most
convincing demonstration that the formula is correct - they share only
the ASSUMPTION (geometric Brownian motion), not the method.
"""

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

import black_scholes as bs
import monte_carlo as mc

st.set_page_config(page_title="Monte Carlo", layout="wide")

st.sidebar.header("Inputs")
S = st.sidebar.number_input("Spot Price ($)", 0.01, value=100.0, step=1.0,
                            help="Current stock price.")
K = st.sidebar.number_input("Strike Price ($)", 0.01, value=100.0, step=1.0,
                            help="Option strike.")
T_days = st.sidebar.number_input("Days to Expiry", 1, value=365, step=1)
r = st.sidebar.number_input("Risk-Free Rate", 0.0, 0.25, 0.05, 0.005,
                            format="%.3f")
sigma = st.sidebar.slider("Volatility", 0.01, 1.5, 0.20, 0.01)
option_type = st.sidebar.radio("Option Type", ["call", "put"], horizontal=True)
n_paths = st.sidebar.select_slider(
    "Simulated Paths", options=[1_000, 5_000, 10_000, 50_000, 100_000,
                                500_000, 1_000_000], value=100_000,
    help="More paths = tighter estimate. Error shrinks like 1/sqrt(n): "
         "100x more paths buys one extra digit of accuracy.")
seed = st.sidebar.number_input("Random Seed", 0, 9999, 0,
                               help="Same seed = same random draws. Change "
                                    "it to resample.")

T = T_days / 365.0

st.title("Monte Carlo vs. Black-Scholes")
st.info(
    "Black-Scholes assumes the stock follows a random walk (geometric "
    "Brownian motion). Instead of trusting the calculus, this page "
    "simulates that random walk directly: draw thousands of possible "
    "futures, average the option payoff over them, discount to today. "
    "If the formula is right, the two prices must agree - and they do."
)

# ---------------------------------------------------------------------------
# The two prices side by side
# ---------------------------------------------------------------------------
bs_val = float(bs.price(option_type, S, K, T, r, sigma))
mc_val, mc_err = mc.mc_price(option_type, S, K, T, r, sigma, n_paths, seed)

c1, c2, c3 = st.columns(3)
c1.metric("Black-Scholes (formula)", f"${bs_val:.4f}")
c2.metric(f"Monte Carlo ({n_paths:,} paths)", f"${mc_val:.4f}",
          delta=f"{mc_val - bs_val:+.4f} vs formula")
c3.metric("MC standard error", f"±${mc_err:.4f}",
          help="The simulation's own uncertainty estimate. The formula "
               "price should land within ~2 of these 95% of the time.")

within = abs(mc_val - bs_val) <= 2 * mc_err
st.caption(
    f"Difference is {abs(mc_val - bs_val) / mc_err:.2f} standard errors - "
    + ("consistent with the formula, as expected."
       if within else
       "slightly outside 2 SE; try another seed - about 5% of runs do this "
       "by pure chance.")
)

# ---------------------------------------------------------------------------
# Sample paths
# ---------------------------------------------------------------------------
st.subheader("What the model thinks the future looks like")
times, paths = mc.sample_paths(S, T, r, sigma, n_paths=60, n_steps=150,
                               seed=seed)
fig, ax = plt.subplots(figsize=(9, 4))
ax.plot(times * 365, paths.T, linewidth=0.7, alpha=0.5)
ax.axhline(K, color="red", linestyle="--", linewidth=1.2, label=f"Strike {K:.0f}")
ax.set_xlabel("Days from now")
ax.set_ylabel("Simulated stock price")
ax.legend()
ax.grid(alpha=0.3)
st.pyplot(fig)
plt.close(fig)
st.caption(
    "60 of the simulated futures. Every path starts at today's spot; the "
    "spread widens with time because uncertainty accumulates - that "
    "widening IS volatility, and it's why longer-dated options cost more."
)

# ---------------------------------------------------------------------------
# Terminal distribution + convergence
# ---------------------------------------------------------------------------
left, right = st.columns(2)

with left:
    st.subheader("Where the paths end up")
    s_t = mc.terminal_prices(S, T, r, sigma, min(n_paths, 100_000), seed)
    fig, ax = plt.subplots(figsize=(6, 4))
    itm = (s_t > K) if option_type == "call" else (s_t < K)
    ax.hist(s_t[itm], bins=60, color="green", alpha=0.6, label="in the money")
    ax.hist(s_t[~itm], bins=60, color="red", alpha=0.6, label="worthless")
    ax.axvline(K, color="black", linestyle="--", linewidth=1.2)
    ax.set_xlabel("Stock price at expiry")
    ax.set_ylabel("Number of paths")
    ax.legend()
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        f"{itm.mean():.1%} of paths finish in the money (green). The "
        "distribution is lognormal - skewed right, never below zero."
    )

with right:
    st.subheader("Convergence to the formula")
    ns, estimates = mc.convergence_curve(option_type, S, K, T, r, sigma,
                                         max(n_paths, 10_000), seed)
    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, estimates, linewidth=1.2, label="MC estimate")
    ax.axhline(bs_val, color="green", linestyle="--", linewidth=1.2,
               label=f"Black-Scholes {bs_val:.4f}")
    ax.set_xscale("log")
    ax.set_xlabel("Number of simulated paths (log scale)")
    ax.set_ylabel("Estimated option price")
    ax.legend()
    ax.grid(alpha=0.3)
    st.pyplot(fig)
    plt.close(fig)
    st.caption(
        "The running average of the simulation homing in on the "
        "closed-form price as more paths are added."
    )

# ---------------------------------------------------------------------------
# Greeks by simulation, two different ways
# ---------------------------------------------------------------------------
st.subheader("Greeks by simulation")
st.markdown(
    "The price above came from averaging a payoff. These Greeks come from "
    "differentiating that same averaging process - **not** from the "
    "closed-form Black-Scholes derivatives, even though they're shown "
    "alongside them for comparison."
)

greeks_mc = mc.mc_greeks(option_type, S, K, T, r, sigma, n_paths, seed)
bs_delta = float(bs.delta(option_type, S, K, T, r, sigma))
bs_vega = float(bs.vega(S, K, T, r, sigma))
bs_gamma = float(bs.gamma(S, K, T, r, sigma))

g1, g2, g3 = st.columns(3)
for col, label, bs_value, (mc_value, mc_se) in zip(
    (g1, g2, g3), ("Delta", "Vega", "Gamma"),
    (bs_delta, bs_vega, bs_gamma),
    (greeks_mc["Delta"], greeks_mc["Vega"], greeks_mc["Gamma"]),
):
    col.metric(f"{label} (Black-Scholes)", f"{bs_value:.4f}")
    within_4se = abs(mc_value - bs_value) <= 4 * mc_se
    col.metric(f"{label} (Monte Carlo)", f"{mc_value:.4f} ± {mc_se:.4f}",
              delta=f"{(abs(mc_value - bs_value) / mc_se):.2f} SE from formula"
              if mc_se > 0 else None,
              delta_color="normal" if within_4se else "inverse")

st.caption(
    "Delta and Vega use **pathwise differentiation**: the simulated path "
    "S_T is a smooth function of spot and volatility, so the derivative "
    "can be pushed inside the expectation and estimated directly from the "
    "same paths used for the price. Gamma can't use that trick - it needs "
    "the derivative of the payoff's SLOPE, and a call or put's slope jumps "
    "at the strike, so pathwise differentiation would need the derivative "
    "of a discontinuity. Gamma instead uses the **likelihood-ratio "
    "(score function) method**: differentiate the probability density of "
    "S_T instead of the payoff, which stays smooth even when the payoff "
    "doesn't. All three are genuinely independent estimates of the same "
    "numbers the Black-Scholes formulas give in closed form - see "
    "`monte_carlo.py` for the derivations."
)
