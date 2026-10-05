"""
Home page / entry point for the site.

Run with:  streamlit run streamlit_app.py

Streamlit turns every file in pages/ into a page in the left sidebar
automatically (the number prefix controls the order). This file is just
the landing page that explains what lives where.
"""

import streamlit as st

import black_scholes as bs

st.set_page_config(page_title="Options Pricer", layout="wide")

st.title("Options Pricer")
st.write(
    "A Black-Scholes options pricing tool built from scratch in Python. "
    "Use the sidebar on the left to move between pages."
)

c1, c2 = st.columns(2)
with c1:
    st.subheader("1. Options Pricer")
    st.write(
        "Price a call and a put, explore how their value changes across "
        "spot price and volatility with interactive heatmaps, see your "
        "PnL in green/red, and read off the five Greeks - both at a point "
        "and as curves across the whole spot range."
    )
    st.subheader("2. Strategy Builder")
    st.write(
        "Ten classic strategies - covered call, married put, spreads, "
        "straddle, strangle, butterfly, iron condor, iron butterfly. "
        "Payoff diagrams, break-evens and a PnL heatmap through time."
    )
with c2:
    st.subheader("3. Market Comparison")
    st.write(
        "Pick any S&P 500 stock, pull its live option chain from Yahoo "
        "Finance, and compare real market prices against this project's "
        "own dividend-adjusted Black-Scholes model, strike by strike - "
        "including the volatility smile backed out of real quotes."
    )
    st.subheader("4. Monte Carlo")
    st.write(
        "Price the same option by brute-force simulation of thousands of "
        "random price paths and watch the estimate converge onto the "
        "Black-Scholes closed form - proof the formula does what it claims."
    )

st.divider()

# A tiny live demo so the landing page isn't just text: price one option
# with the model right here.
st.subheader("Quick demo")
st.write("The model in one line - a 1-year at-the-money option on a $100 "
         "stock at 20% volatility and a 5% risk-free rate:")

call = float(bs.call_price(100, 100, 1.0, 0.05, 0.20))
put = float(bs.put_price(100, 100, 1.0, 0.05, 0.20))
d1, d2 = st.columns(2)
d1.metric("Call fair value", f"${call:.4f}")
d2.metric("Put fair value", f"${put:.4f}")

st.caption(
    "Model: Black-Scholes, implemented by hand in black_scholes.py - "
    "the only library math used is the normal distribution from SciPy. "
    "Educational project, not investment advice."
)
