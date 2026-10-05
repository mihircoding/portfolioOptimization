"""
Live market data via yfinance (free Yahoo Finance wrapper).

Three jobs:
  1. get the list of S&P 500 tickers (scraped from Wikipedia, with a
     hardcoded fallback list in case the scrape fails / no internet)
  2. pull the current spot price and the option chain for a ticker
  3. estimate historical volatility from the last year of daily closes,
     which we feed into Black-Scholes as our "model" volatility
"""

import time

import numpy as np
import pandas as pd
import yfinance as yf
from yfinance.exceptions import YFRateLimitError

# Yahoo throttles shared cloud IPs hard (every free yfinance deployment on
# Streamlit Cloud exits from the same few addresses). Retry with growing
# pauses before giving up; combined with the hour-long st.cache_data on the
# pages, one successful fetch then serves every visitor for an hour.
_RETRY_DELAYS = (2, 5, 10)


def _with_retries(fetch):
    for delay in _RETRY_DELAYS:
        try:
            return fetch()
        except YFRateLimitError:
            time.sleep(delay)
    return fetch()  # final attempt: let the error reach the page's handler

# Fallback so the app still works if the Wikipedia scrape fails.
FALLBACK_TICKERS = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL", "META", "TSLA", "BRK-B",
    "JPM", "V", "UNH", "XOM", "JNJ", "WMT", "MA", "PG", "HD", "COST",
    "ORCL", "MRK", "CVX", "ABBV", "KO", "PEP", "BAC", "CRM", "AMD",
    "NFLX", "ADBE", "TMO", "MCD", "CSCO", "ABT", "INTC", "DIS", "WFC",
    "QCOM", "IBM", "GE", "CAT", "VZ", "AXP", "TXN", "AMGN", "PFE",
    "GS", "MS", "NKE", "UBER", "BA", "HON", "SBUX", "LOW", "PLTR",
    "T", "BLK", "SPGI", "C", "SCHW", "BMY",
]


def sp500_tickers():
    """
    Scrape the S&P 500 constituent table from Wikipedia. pandas can read
    HTML tables directly. Yahoo uses '-' where the index uses '.'
    (BRK.B -> BRK-B), so we normalize. Falls back to a hardcoded list of
    60 large caps if anything goes wrong.
    """
    try:
        # Wikipedia rejects requests without a browser-like User-Agent,
        # so fetch the HTML ourselves and hand the text to pandas.
        from io import StringIO
        from urllib.request import Request, urlopen

        req = Request(
            "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies",
            headers={"User-Agent": "Mozilla/5.0"},
        )
        with urlopen(req, timeout=15) as resp:
            html = resp.read().decode("utf-8")
        tables = pd.read_html(StringIO(html))
        symbols = tables[0]["Symbol"].str.replace(".", "-", regex=False)
        return sorted(symbols.tolist())
    except Exception:
        return sorted(FALLBACK_TICKERS)


def get_spot_and_history(ticker, period="1y"):
    """
    Returns (latest close price, dataframe of daily history).
    Raises ValueError if Yahoo returns nothing (bad ticker / no network).
    """
    hist = _with_retries(lambda: yf.Ticker(ticker).history(period=period))
    if hist.empty:
        raise ValueError(f"No price data returned for {ticker}")
    return float(hist["Close"].iloc[-1]), hist


def historical_volatility(hist, window=252):
    """
    Classic realized-vol estimate:
      1. daily log returns: ln(P_t / P_{t-1})
      2. standard deviation of those returns
      3. annualize by sqrt(252) (252 trading days per year)
    This is the sigma we plug into Black-Scholes on the market page.
    """
    closes = hist["Close"].tail(window + 1)
    log_returns = np.log(closes / closes.shift(1)).dropna()
    return float(log_returns.std() * np.sqrt(252))


def dividend_yield(hist, spot):
    """
    Trailing dividend yield: everything the stock paid out over the
    history window (1 year by default) divided by today's price. yfinance
    includes a 'Dividends' column in the history dataframe - zero on most
    days, the per-share amount on ex-dividend dates. This becomes the q
    in the dividend-adjusted Black-Scholes model.
    """
    if "Dividends" not in hist.columns:
        return 0.0
    return float(hist["Dividends"].sum() / spot)


def get_option_chain(ticker, expiry):
    """
    Returns (calls_df, puts_df) for one expiration date. Each dataframe
    has strike, lastPrice, bid, ask, impliedVolatility, volume, etc.
    """
    chain = _with_retries(lambda: yf.Ticker(ticker).option_chain(expiry))
    return chain.calls, chain.puts


def get_expirations(ticker):
    """List of available option expiration dates ('YYYY-MM-DD' strings)."""
    return list(_with_retries(lambda: yf.Ticker(ticker).options))


def years_to_expiry(expiry_str):
    """Convert an expiry date string to year-fraction T for Black-Scholes."""
    expiry = pd.Timestamp(expiry_str)
    now = pd.Timestamp.now()
    days = max((expiry - now).days, 0) + 0.5   # +0.5: expiry day itself counts
    return days / 365.0
