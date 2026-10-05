"""
Page 3: Compare the model against the real market.

Pick any S&P 500 stock, pick an expiration, and the page:
  1. pulls the live spot price and option chain from Yahoo Finance
  2. estimates volatility from the last year of daily returns
  3. prices every near-the-money strike with OUR Black-Scholes code
  4. puts model price and market price side by side, colored green when
     the market quote is above our model (option looks "rich") and red
     when below (looks "cheap")

The differences you see are real and educational: the market prices in a
volatility SMILE (different implied vol per strike), while our model uses
one flat historical vol for everything.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from yfinance.exceptions import YFRateLimitError

import black_scholes as bs
import market_data as md

st.set_page_config(page_title="Market Comparison", layout="wide")
st.title("Model vs. Market - S&P 500 Options")

st.info(
    "Live quotes from Yahoo Finance, cached for up to an hour to stay under "
    "Yahoo's rate limits. Model prices come from this project's own "
    "Black-Scholes code, fed with volatility estimated from the last year "
    "of daily returns. Where the two disagree, the market is telling you "
    "its volatility expectation differs from history."
)


@st.cache_data(ttl=24 * 3600)
def _tickers():
    return md.sp500_tickers()


# Hour-long caches, shared across every visitor of the deployed app: one
# successful Yahoo fetch serves everyone until it expires, which is what
# keeps the shared cloud IP under the rate limit most of the time.
@st.cache_data(ttl=3600)
def _spot_and_vol(ticker):
    spot, hist = md.get_spot_and_history(ticker)
    return (spot, md.historical_volatility(hist),
            md.dividend_yield(hist, spot))


@st.cache_data(ttl=24 * 3600)
def _expirations(ticker):
    return md.get_expirations(ticker)


@st.cache_data(ttl=3600)
def _chain(ticker, expiry):
    return md.get_option_chain(ticker, expiry)


def _fetch(what, cached_fn, *args):
    """Run one of the cached fetchers; stop the page with a readable message
    instead of a traceback when Yahoo says no."""
    try:
        return cached_fn(*args)
    except YFRateLimitError:
        st.warning(
            "Yahoo Finance is rate-limiting this app's shared cloud IP at "
            "the moment (a known limitation of free hosting - every "
            "yfinance app on Streamlit Cloud shares the same addresses). "
            "Already-viewed tickers stay cached for an hour; for a fresh "
            "one, retry in a minute or two."
        )
        st.stop()
    except Exception as exc:
        st.error(f"Could not fetch {what}: {exc}")
        st.stop()


tickers = _tickers()
default_idx = tickers.index("AAPL") if "AAPL" in tickers else 0

c1, c2, c3 = st.columns(3)
ticker = c1.selectbox("S&P 500 stock", tickers, index=default_idx,
                      help="All ~500 index members, scraped from Wikipedia.")

spot, hist_vol, div_yield = _fetch(f"price history for {ticker}",
                                   _spot_and_vol, ticker)

expirations = _fetch(f"option expirations for {ticker}", _expirations, ticker)
if not expirations:
    st.error(f"{ticker} has no listed options.")
    st.stop()

expiry = c2.selectbox("Expiration", expirations,
                      help="Listed option expiration dates for this stock.")
r = c3.number_input("Risk-Free Rate", 0.0, 0.25, 0.05, 0.005, format="%.3f",
                    help="Used by the model. 3-month T-bill yield is a "
                         "reasonable choice.")

T = md.years_to_expiry(expiry)

m1, m2, m3, m4 = st.columns(4)
m1.metric(f"{ticker} Spot", f"${spot:.2f}")
m2.metric("Historical Vol (1y)", f"{hist_vol:.1%}")
m3.metric("Dividend Yield (1y)", f"{div_yield:.2%}",
          help="Trailing 12 months of dividends / current price. Fed into "
               "the model as q - dividends drag the forward price down, "
               "making calls cheaper and puts dearer.")
m4.metric("Time to Expiry", f"{T * 365:.0f} days")

calls, puts = _fetch(f"the {expiry} option chain for {ticker}",
                     _chain, ticker, expiry)

n_strikes = st.slider("Strikes around the money to show", 4, 20, 10)


def build_comparison(chain_df, option_type):
    """
    Take Yahoo's chain for one side (calls or puts), keep the strikes
    closest to the current spot, and add our model's outputs per row.
    """
    df = chain_df[["strike", "lastPrice", "bid", "ask",
                   "impliedVolatility", "volume"]].copy()
    df = df.iloc[(df["strike"] - spot).abs().sort_values().index[:n_strikes]]
    df = df.sort_values("strike").reset_index(drop=True)

    df["model"] = [
        float(bs.price(option_type, spot, k, T, r, hist_vol, div_yield))
        for k in df["strike"]
    ]
    df["diff"] = df["lastPrice"] - df["model"]
    # back out the implied vol from the market's last price with our own
    # bisection solver, to compare against Yahoo's reported IV
    df["our IV"] = [
        bs.implied_vol(option_type, p, spot, k, T, r, div_yield)
        for p, k in zip(df["lastPrice"], df["strike"])
    ]
    df["delta"] = [
        float(bs.delta(option_type, spot, k, T, r, hist_vol, div_yield))
        for k in df["strike"]
    ]
    return df


def style_diff(df):
    """Green = market above model (rich), red = market below (cheap)."""
    def color(v):
        if pd.isna(v):
            return ""
        return ("background-color:#c6efce;color:#111" if v > 0
                else "background-color:#ffc7ce;color:#111")
    return (
        df.style
        .map(color, subset=["diff"])
        .format({
            "strike": "{:.2f}", "lastPrice": "{:.2f}", "bid": "{:.2f}",
            "ask": "{:.2f}", "model": "{:.2f}", "diff": "{:+.2f}",
            "impliedVolatility": "{:.1%}", "our IV": "{:.1%}",
            "volume": "{:.0f}", "delta": "{:.3f}",
        }, na_rep="-")
    )


calls_cmp = build_comparison(calls, "call")
puts_cmp = build_comparison(puts, "put")

tab_c, tab_p = st.tabs(["Calls", "Puts"])
with tab_c:
    st.dataframe(style_diff(calls_cmp), width='stretch', hide_index=True)
with tab_p:
    st.dataframe(style_diff(puts_cmp), width='stretch', hide_index=True)

# ---------------------------------------------------------------------------
# Volatility smile - the market's IV per strike vs our one flat number.
# This chart is the single best picture of WHY model and market disagree.
# ---------------------------------------------------------------------------
st.subheader("Volatility Smile")

fig = go.Figure()
for cmp_df, label, color in [(calls_cmp, "Calls", "#2ca02c"),
                             (puts_cmp, "Puts", "#d62728")]:
    pts = cmp_df.dropna(subset=["our IV"])
    fig.add_trace(go.Scatter(
        x=pts["strike"], y=pts["our IV"], mode="lines+markers",
        name=f"{label} implied vol", line={"color": color},
        hovertemplate="Strike %{x}<br>IV %{y:.1%}<extra></extra>",
    ))
fig.add_hline(y=hist_vol, line_dash="dash", line_color="gray",
              annotation_text=f"historical vol {hist_vol:.1%}")
fig.add_vline(x=spot, line_dash="dot", line_color="steelblue",
              annotation_text=f"spot {spot:.0f}")
fig.update_layout(
    xaxis_title="Strike", yaxis_title="Implied Volatility",
    yaxis_tickformat=".0%", height=420,
    margin={"l": 60, "r": 20, "t": 20, "b": 60},
)
st.plotly_chart(fig, width='stretch')
st.caption(
    "Each point is the volatility our bisection solver backs out of one "
    "market price. A flat line at the gray dash is what plain Black-Scholes "
    "assumes; the curve you actually see - higher IV away from the money, "
    "especially on the downside - is the famous volatility smile/skew. "
    "Gaps mean the last traded price was too stale to invert."
)

with st.expander("How to read this table"):
    st.markdown("""
- **lastPrice / bid / ask** - straight from the market. `lastPrice` can be
  stale for illiquid strikes (check `volume`).
- **model** - our Black-Scholes price using 1-year historical volatility.
- **diff** - market minus model. **Green**: the market is paying MORE than
  history justifies (it expects more movement than the past year showed).
  **Red**: paying less.
- **impliedVolatility** - Yahoo's reported IV.
- **our IV** - implied vol we back out ourselves by inverting Black-Scholes
  with a bisection solver on `lastPrice`. It should land near Yahoo's
  number; `-` means the last price was too stale to invert.
- **delta** - model delta; ~0.50 is at-the-money.

Look at IV across strikes: it is usually NOT flat. Downside strikes carry
higher IV (the "volatility smile/skew") because crash protection is in
demand - one thing plain Black-Scholes, with its single flat sigma,
cannot capture.
""")
