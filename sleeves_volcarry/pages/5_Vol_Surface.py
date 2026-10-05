"""
Page 5: The volatility surface the market is actually quoting.

Page 3 puts our model price next to the market price and says the gap is a
volatility smile. This page measures it. It pulls a live chain, recovers the
forward from put-call parity, inverts every usable out-of-the-money quote
with this project's own solver, and plots what comes out - by strike, by
maturity, and as the two numbers a desk would quote it with.

Nothing here reads Yahoo's own impliedVolatility column. Every number on
the page is solved from a bid-ask mid by black_scholes.implied_vol.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from yfinance.exceptions import YFRateLimitError

import market_data as md
import vol_surface as vsurf

st.set_page_config(page_title="Vol Surface", layout="wide")
st.title("The Volatility Surface")

st.info(
    "Black-Scholes assumes one volatility for every strike and every "
    "expiry. The market disagrees, and this page is the disagreement "
    "measured rather than asserted. Quotes are bid-ask mids, the forward "
    "comes from put-call parity rather than from a dividend estimate, and "
    "only out-of-the-money options are used - they are the liquid ones and "
    "their price is almost entirely a statement about volatility."
)


@st.cache_data(ttl=24 * 3600)
def _tickers():
    return md.sp500_tickers()


@st.cache_data(ttl=3600, show_spinner=False)
def _surface(ticker, n_expiries):
    return vsurf.build_surface(ticker, max_expiries=n_expiries)


try:
    tickers = _tickers()
except Exception:
    tickers = md.FALLBACK_TICKERS

default = tickers.index("SPY") if "SPY" in tickers else 0
ticker = st.sidebar.selectbox("Underlying", tickers, index=default)
n_expiries = st.sidebar.slider("Expiries to sample", 3, 12, 8,
                               help="Spread evenly across the available term, "
                                    "not just the nearest few - otherwise every "
                                    "smile is next week's.")

with st.spinner(f"Pulling {ticker} chains and inverting every quote..."):
    try:
        surface = _surface(ticker, n_expiries)
    except YFRateLimitError:
        st.error("Yahoo is rate-limiting this app right now. Try again in a "
                 "few minutes - results are cached for an hour once one "
                 "request gets through.")
        st.stop()

if surface.empty:
    st.warning(f"No usable quotes for {ticker}. Thinly traded names often have "
               "one-sided or crossed markets on every strike, which the quote "
               "filter drops on purpose.")
    st.stop()

terms = vsurf.term_structure(surface)

c1, c2, c3, c4 = st.columns(4)
front = terms.iloc[0]
c1.metric("Front ATM vol", f"{front['atm']:.1%}")
c2.metric("25-delta risk reversal", f"{front['risk_reversal']:+.1%}",
          help="Put IV minus call IV. Positive means downside protection is "
               "dearer than upside - the usual state of an equity index.")
c3.metric("25-delta butterfly", f"{front['butterfly']:+.1%}",
          help="Wings minus the at-the-money level: how much more the market "
               "charges for tails than a lognormal would.")
c4.metric("Quotes inverted", f"{len(surface):,}")

st.subheader("The smile, by expiry")
fig = go.Figure()
for expiry, smile in surface.groupby("expiry", sort=False):
    s = smile.sort_values("moneyness")
    fig.add_trace(go.Scatter(x=s["moneyness"], y=s["iv"], mode="lines+markers",
                             name=f"{expiry}  ({s['T'].iloc[0] * 365:.0f}d)",
                             marker=dict(size=4)))
fig.add_vline(x=0, line_dash="dot", line_color="#888",
              annotation_text="forward", annotation_position="top")
fig.update_layout(height=460, xaxis_title="log(strike / forward)",
                  yaxis_title="implied volatility", yaxis_tickformat=".0%",
                  legend=dict(orientation="h", y=-0.2))
st.plotly_chart(fig, use_container_width=True)
st.caption(
    "Plotted against log-moneyness rather than strike so expiries are "
    "comparable: a $10 move means something different a week out than it "
    "does in a year. The downward slope is the equity skew - crash "
    "protection is bid, and that is a fact about demand, not about the "
    "distribution of returns."
)

left, right = st.columns([1.15, 1])

with left:
    st.subheader("Term structure")
    show = terms[["expiry", "T", "forward", "atm", "put_25d", "call_25d",
                  "risk_reversal", "butterfly", "n_quotes"]].copy()
    st.dataframe(
        show.style.format({"T": "{:.3f}", "forward": "{:.2f}", "atm": "{:.2%}",
                           "put_25d": "{:.2%}", "call_25d": "{:.2%}",
                           "risk_reversal": "{:+.2%}", "butterfly": "{:+.2%}"}),
        use_container_width=True, hide_index=True)
    st.caption(
        "Forwards come from the parity fit, one per expiry. They should rise "
        "with maturity by roughly the financing rate net of dividends - if "
        "they don't, the chain is stale and everything above it is suspect."
    )

with right:
    st.subheader("No-arbitrage check")
    violations = vsurf.calendar_arbitrage(surface)
    if violations.empty:
        st.success("Total variance is non-decreasing in maturity at every "
                   "moneyness tested. No calendar arbitrage.")
    else:
        st.warning(f"{len(violations)} calendar violation(s): total variance "
                   "falls as maturity rises, which would be free money in a "
                   "calendar spread. Usually a stale quote or a dividend the "
                   "parity fit missed rather than a real opportunity.")
        st.dataframe(violations, use_container_width=True, hide_index=True)
    st.caption(
        "Reported rather than silently smoothed away. A surface presented "
        "without this check is a surface nobody verified."
    )

st.subheader("What one flat volatility costs")
front_expiry = terms.iloc[0]["expiry"]
front_smile = surface[surface["expiry"] == front_expiry]
flat = float(front_smile["iv"].median())
errors = vsurf.flat_vol_error(front_smile, flat)

fig2 = go.Figure()
fig2.add_trace(go.Bar(x=errors["moneyness"], y=errors["error"],
                      marker_color=np.where(errors["error"] > 0, "#ef553b", "#00cc96"),
                      name="model - market"))
fig2.update_layout(height=340, xaxis_title="log(strike / forward)",
                   yaxis_title="model price - market price ($)",
                   showlegend=False)
st.plotly_chart(fig2, use_container_width=True)
st.caption(
    f"The {front_expiry} chain priced with a single {flat:.1%} volatility - "
    "the median of its own smile, so this is the flat model's best case. "
    "Red is the model charging too much, green too little. The shape is the "
    "smile: a flat vol underprices the downside strikes people actually buy "
    "and overprices the middle, which is why the errors are not noise around "
    "zero but a curve."
)
