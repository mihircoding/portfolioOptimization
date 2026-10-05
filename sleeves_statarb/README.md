# Pairs Trading

**[Live site &rarr;](https://mihircoding.github.io/pairsTrading/)** — the full scan, every surviving pair, and a browser-side backtest you can re-run with your own parameters.

A statistical arbitrage study on the S&P 100: screen every pair for cointegration on a
formation window, then trade the survivors out of sample and see whether the screen actually
predicted anything.

Short version — it didn't. 4,950 pairs tested, 930 passed at 5%, one survived Bonferroni,
and the mean out-of-sample Sharpe across the survivors was indistinguishable from zero.
The write-up leads with that rather than the top of the leaderboard.

Trading all 930 survivors as one equal-weight book returns −2.14% at a Sharpe of −0.19, and
the formation p-value's rank correlation with out-of-sample Sharpe is −0.05 over those 930
pairs. The screen doesn't rank — which is a stronger statement than any single pair can make.

And the control nobody runs: `control.py` backtests the 4,020 pairs the screen **rejected**,
under identical rules. They came out marginally ahead — mean Sharpe +0.009 against −0.009 for
the survivors, 45.5% profitable against 41.8% — with a permutation test at p = 0.21, so the two
groups are the same group. Across all 4,950 pairs the rank correlation between formation p-value
and out-of-sample Sharpe is −0.001.

The obvious objection is that everything is estimated once and frozen for five years, which is
not how a pairs book is run. So `walkforward.py` re-scans all 4,950 pairs every quarter on
trailing data, re-estimates each hedge ratio, and trades whatever currently passes — 99,000
cointegration tests. Refitting the beta, re-running the screen, tightening it to 1%, and sizing
the z-score window by each pair's own half-life all land between −1.27% and +0.08% over five
years. And the reason is in the scan itself: **0 of 4,950 pairs pass the cointegration test in
all 20 quarters**, and the median pair that ever passes, passes in 4 of 20. The screen doesn't
rank and it doesn't repeat.

## Layout

| Path | What it is |
|---|---|
| [`01-pairs-trading/src/pairs.ipynb`](01-pairs-trading/src/pairs.ipynb) | The pipeline, built up step by step on a 12-ticker universe |
| [`01-pairs-trading/scan.py`](01-pairs-trading/scan.py) | The full S&P 100 scan — writes everything to `results/` |
| [`01-pairs-trading/portfolio.py`](01-pairs-trading/portfolio.py) | Trades all 930 survivors as a book, and asks whether the p-value ranks |
| [`01-pairs-trading/walkforward.py`](01-pairs-trading/walkforward.py) | Re-scans and re-estimates every quarter, and asks whether the screen repeats |
| [`01-pairs-trading/control.py`](01-pairs-trading/control.py) | Trades the 4,020 pairs the screen rejected, as a control group |
| [`01-pairs-trading/app.py`](01-pairs-trading/app.py) | Streamlit explorer over those results |
| [`01-pairs-trading/README.md`](01-pairs-trading/README.md) | The theory: cointegration vs correlation, Engle-Granger, the pitfalls |
| [`docs/`](docs/) | Static version of the explorer for GitHub Pages; `portfolio.py` and `walkforward.py` write its data files |

## Running it

```
cd 01-pairs-trading
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
pytest
python scan.py          # ~65s, builds results/
python portfolio.py     # the book of all 930 survivors, needs results/
python walkforward.py   # quarterly re-scan, 99,000 tests, slow; --cached reuses the scan
streamlit run app.py
```

## Caveats

Survivorship bias: the universe is the S&P 100 as it stands today, so the backtest asks how
today's winners behaved on their way to winning. Fixing it needs point-in-time constituents
(CRSP, Compustat), which free data doesn't provide.

Prices come from Yahoo Finance — fine for this, useless for anything live. None of this is
investment advice.
