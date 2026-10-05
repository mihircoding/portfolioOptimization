# Results

Three write-ups, in three files, because they answer three different questions.

- **[notes/statarb.md](notes/statarb.md)** — do cointegrated equity pairs mean
  revert out of sample? Screen 4,950 pairs of the S&P 100, trade the 930 that
  pass, and measure what the screen predicted. Includes the control group, the
  screen's own size distortion, and the survivorship bias priced rather than
  admitted.
- **[notes/volcarry.md](notes/volcarry.md)** — what is an option worth, and
  what does the hedge that justifies the price actually earn? Black-Scholes
  from scratch, an arbitrage-free SSVI surface, and a delta-hedging study that
  sells a one-month ATM SPY call every month from 2007.
- **[notes/allocator.md](notes/allocator.md)** — mean-variance, shrinkage, risk
  parity, CVaR, a factor risk model and hierarchical risk parity, each tested
  out of sample. Equal weighting beats all of them.
- **This file** — the question that needs all three, and could not be asked
  while they were separate repositories.

Interview notes: [notes/interview-statarb.md](notes/interview-statarb.md),
[notes/interview-volcarry.md](notes/interview-volcarry.md),
[notes/interview-allocator.md](notes/interview-allocator.md).

---

## The optimizer finally gets something worth optimizing

Reproduce with: `python multi_strategy.py`

The standing result of notes/allocator.md is that equal weighting beats
mean-variance out of sample, at every estimation window tested, on five ETFs
and on fifty US large caps. That result has always had an obvious rejoinder:
of course it does, because those are **assets**, and a risk allocator exists to
combine **strategies**. When everything in the book is 85% correlated with the
market there is no risk to diversify away, so no weighting scheme can earn
anything by trying.

This repository now holds two strategies, and they arrived from the two
projects that merged into it.

| sleeve | what it is | ann. return | ann. vol | Sharpe | worst month |
|---|---|---|---|---|---|
| `statarb` | 930 cointegrated equity pairs, long one leg, short the other | −0.67% | 1.53% | **−0.44** | −1.17% |
| `volcarry` | sell a one-month ATM SPY call monthly, delta-hedge daily | +4.59% | 1.66% | **2.77** | −0.78% |

46 overlapping months, 2021-02 to 2024-11. Correlation between them: **+0.029**.
Diversification ratio **1.39**, against a ceiling of 1.41 for two sleeves. By
the textbook this is as good as a two-strategy book gets.

### 1. For the first time here, the optimizer wins

Walk-forward: estimate on a trailing 24 months, hold for one, repeat. 22
out-of-sample months.

| scheme | ann. return | ann. vol | Sharpe | weight on `statarb` |
|---|---|---|---|---|
| **max Sharpe** | 3.51% | 1.33% | **2.64** | **0%** |
| risk parity | 1.49% | 0.95% | 1.58 | 43% |
| inverse vol | 1.49% | 0.95% | 1.58 | 43% |
| equal weight | 1.39% | 0.89% | 1.57 | 50% |
| HRP | 1.60% | 1.03% | 1.56 | 37% |
| min variance | 1.59% | 1.02% | 1.55 | 37% |

Mean-variance beats equal weighting by more than a full unit of Sharpe, and
every risk-based method lands in a dead heat with 1/N. That is the opposite of
every table in notes/allocator.md.

### 2. It does not win for the reason the diversification argument predicts

The reason is not that it found a clever combination of two good things. It is
that **one of the sleeves does not make money, and only mean-variance can see
it.**

Risk parity, inverse volatility, HRP and min variance never read an expected
return — the whole appeal of that family is that it does not need one, because
expected returns are the hardest thing to estimate. `allocators()` gives all six
schemes the same `(mu, cov)` signature so that the four which throw `mu` away
do so visibly, and `test_the_risk_based_schemes_ignore_expected_returns_entirely`
pins it: multiply every expected return by ten and those four weights do not
move at all.

Then the trap closes. The sleeve that earns nothing is also the **quieter** one —
1.53% annualized volatility against 1.66%:

```
statarb      1.53% vol,  -0.67% return      <- lower risk AND lower return
volcarry     1.66% vol,  +4.59% return
```

So every return-agnostic method allocates *toward* it. Inverse volatility and
risk parity put 43% of the book into a strategy with a negative Sharpe; HRP and
min variance put 37%; equal weighting puts 50% by definition. Mean-variance puts
zero.

This is a real blind spot and not a contrived one. The pitch for risk parity is
that it avoids the estimation error in expected returns. The cost of that, which
nobody puts on the slide, is that it has no mechanism whatsoever for declining
to fund something broken — and a broken strategy usually looks *calm*, because
a strategy with no edge has no edge to be volatile about. Hierarchical risk
parity does not help: notes/allocator.md already found that its tree-clustering
step is worth about 1% of the volatility it predicts, and clustering two sleeves
is clustering nothing.

### 3. The test that says which regime you are in, before you choose

The two findings are not in conflict, and the thing that reconciles them is one
number. Mean-variance beats equal weighting when the difference in expected
return is **large relative to the error on estimating it**, and loses when it is
not, because then it is allocating on noise.

```
return gap        +5.27% a year  (+0.4391% a month)
standard error     0.0961% a month, on 46 months
t-statistic           4.57
```

A t-statistic of 4.57 on the return difference. The sleeves are not similar
assets whose means cannot be told apart; one of them is reliably better, and 46
months is enough to know it. In notes/allocator.md's five-ETF problem the same
statistic is under 1 — which is why the optimizer loses there and wins here.

That is the useful version of the result, and it is computable **before**
picking an allocator rather than after. "Optimizers don't work" and "optimizers
work" are both wrong. The question is whether your assets differ by more than
your sample can resolve, and `return_gap_tstat()` answers it in one line.

### 4. What the overlap window leaves out, which is the check that matters

`statarb` only exists from 2021, so the two sleeves overlap over 2021-2024 —
and 2021-2024 contains no 2008 and no March 2020. Short volatility is a strategy
whose entire risk lives in months like those. The window was chosen by the other
sleeve's start date, which is a selection effect even though nobody selected it.

| `volcarry` over | months | ann. return | ann. vol | Sharpe | worst month |
|---|---|---|---|---|---|
| its full history | 215 | 4.87% | 2.22% | 2.20 | −2.68% |
| the overlap only | 46 | 4.59% | 1.66% | **2.77** | −0.78% |

Worst months excluded: 2020-03 (−2.68%), 2008-09 (−1.96%), 2008-11 (−1.47%).
The window inflates the sleeve's Sharpe by **+0.57**.

The important part is *where* the inflation is. The annualized return is almost
unchanged — 4.59% against 4.87% — and all of the difference is in volatility,
1.66% against 2.22%. So the quantity mean-variance actually acted on, the
**return gap**, is not a window artifact; what the window flattered is the risk.
The allocation conclusion survives. The 2.64 Sharpe does not, and the honest
number for the combined book is lower.

### 5. And it holds at every estimation window

A single lookback is a single draw, so the same move notes/allocator.md makes
for its own headline:

| lookback | OOS months | equal weight | inverse vol | risk parity | HRP | min variance | max Sharpe |
|---|---|---|---|---|---|---|---|
| 12m | 34 | 1.21 | 1.23 | 1.23 | 1.22 | 1.09 | **2.06** |
| 18m | 28 | 1.43 | 1.55 | 1.55 | 1.62 | 1.64 | **2.36** |
| 24m | 22 | 1.57 | 1.58 | 1.58 | 1.56 | 1.55 | **2.64** |
| 30m | 16 | 1.75 | 1.73 | 1.73 | 1.68 | 1.69 | **1.88** |

Max Sharpe wins at all four. The margin collapses at 30 months, where only 16
out-of-sample months are left and the comparison is close to meaningless — which
is the right place for it to collapse.

### What this does not establish

- **Two sleeves is not a portfolio.** The interesting version of this study has
  eight or ten weakly correlated sleeves, where the covariance matters as much
  as the means. With two, "allocation" is one number.
- **22 out-of-sample months.** Everything in section 1 rests on fewer than two
  years. Section 5 is the only reason to take it at all seriously, and it is a
  weak reason.
- **The sleeves are not cost-matched.** `statarb` charges its own transaction
  costs per notes/statarb.md; `volcarry` charges hedging costs in basis points
  but no bid-ask on the option it sells, which for a one-month ATM SPY option
  is not negligible. The return gap is therefore overstated by an unmeasured
  amount, and the amount is small relative to 5.27% a year but not zero.
- **Nothing here is levered to a common risk target.** Both sleeves run at
  under 2% volatility, so the combined book earns 3.5% a year at 1.3% vol. A
  real book would lever that, and leverage changes which risks matter.
