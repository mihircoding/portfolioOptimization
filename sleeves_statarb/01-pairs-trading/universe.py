"""S&P 100 tickers with GICS sectors.


"""

SP100: dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology",
    "ACN": "Information Technology",
    "ADBE": "Information Technology",
    "AMD": "Information Technology",
    "AVGO": "Information Technology",
    "CRM": "Information Technology",
    "CSCO": "Information Technology",
    "IBM": "Information Technology",
    "INTC": "Information Technology",
    "INTU": "Information Technology",
    "MSFT": "Information Technology",
    "MU": "Information Technology",
    "NVDA": "Information Technology",
    "ORCL": "Information Technology",
    "QCOM": "Information Technology",
    "TXN": "Information Technology",
    # Financials
    "AIG": "Financials",
    "AXP": "Financials",
    "BAC": "Financials",
    "BK": "Financials",
    "BLK": "Financials",
    "BRK-B": "Financials",
    "C": "Financials",
    "COF": "Financials",
    "GS": "Financials",
    "JPM": "Financials",
    "MA": "Financials",
    "MET": "Financials",
    "MS": "Financials",
    "PYPL": "Financials",
    "SCHW": "Financials",
    "SPGI": "Financials",
    "USB": "Financials",
    "V": "Financials",
    "WFC": "Financials",
    # Health Care
    "ABBV": "Health Care",
    "ABT": "Health Care",
    "AMGN": "Health Care",
    "BMY": "Health Care",
    "CVS": "Health Care",
    "DHR": "Health Care",
    "GILD": "Health Care",
    "JNJ": "Health Care",
    "LLY": "Health Care",
    "MDT": "Health Care",
    "MRK": "Health Care",
    "PFE": "Health Care",
    "TMO": "Health Care",
    "UNH": "Health Care",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary",
    "HD": "Consumer Discretionary",
    "LOW": "Consumer Discretionary",
    "MCD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary",
    "SBUX": "Consumer Discretionary",
    "TGT": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    # Consumer Staples
    "CL": "Consumer Staples",
    "COST": "Consumer Staples",
    "KHC": "Consumer Staples",
    "KO": "Consumer Staples",
    "MDLZ": "Consumer Staples",
    "MO": "Consumer Staples",
    "PEP": "Consumer Staples",
    "PG": "Consumer Staples",
    "PM": "Consumer Staples",
    "WMT": "Consumer Staples",
    # Communication Services
    "CHTR": "Communication Services",
    "CMCSA": "Communication Services",
    "DIS": "Communication Services",
    "GOOGL": "Communication Services",
    "META": "Communication Services",
    "NFLX": "Communication Services",
    "T": "Communication Services",
    "TMUS": "Communication Services",
    "VZ": "Communication Services",
    # Industrials
    "BA": "Industrials",
    "CAT": "Industrials",
    "DE": "Industrials",
    "EMR": "Industrials",
    "FDX": "Industrials",
    "GD": "Industrials",
    "GE": "Industrials",
    "HON": "Industrials",
    "LMT": "Industrials",
    "MMM": "Industrials",
    "RTX": "Industrials",
    "UNP": "Industrials",
    "UPS": "Industrials",
    # Energy
    "COP": "Energy",
    "CVX": "Energy",
    "XOM": "Energy",
    # Utilities
    "DUK": "Utilities",
    "NEE": "Utilities",
    "SO": "Utilities",
    # Materials
    "DOW": "Materials",
    "LIN": "Materials",
    # Real Estate
    "AMT": "Real Estate",
    "SPG": "Real Estate",
}

TICKERS: list[str] = sorted(SP100)


def sector(ticker: str) -> str:
    return SP100.get(ticker, "Unknown")


# rough stand-in for "is there an economic reason these two move together"
def same_sector(a: str, b: str) -> bool:
    return sector(a) == sector(b) and sector(a) != "Unknown"


if __name__ == "__main__":
    from collections import Counter

    counts = Counter(SP100.values())
    print(f"{len(TICKERS)} tickers across {len(counts)} sectors")
    print(f"{len(TICKERS) * (len(TICKERS) - 1) // 2} pairs to test")
    for name, n in counts.most_common():
        print(f"  {n:>3}  {name}")
