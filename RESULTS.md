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
| `volcarry` | sell a one-month ATM SPY call monthly, delta-hedge daily | +4.46% | 1.66% | **2.68** | −0.79% |

46 overlapping months, 2021-02 to 2024-11. Correlation between them: **+0.029**.
Diversification ratio **1.39**, against a ceiling of 1.41 for two sleeves. By
the textbook this is as good as a two-strategy book gets.

### 1. For the first time here, the optimizer wins

Walk-forward: estimate on a trailing 24 months, hold for one, repeat. 22
out-of-sample months.

| scheme | ann. return | ann. vol | Sharpe | weight on `statarb` |
|---|---|---|---|---|
| **max Sharpe** | 3.37% | 1.33% | **2.54** | **0%** |
| risk parity | 1.41% | 0.95% | 1.50 | 43% |
| inverse vol | 1.41% | 0.95% | 1.50 | 43% |
| equal weight | 1.32% | 0.89% | 1.49 | 50% |
| HRP | 1.51% | 1.03% | 1.47 | 37% |
| min variance | 1.50% | 1.02% | 1.47 | 37% |

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
return gap        +5.13% a year  (+0.4276% a month)
standard error     0.0961% a month, on 46 months
t-statistic           4.45
```

A t-statistic of 4.45 on the return difference. The sleeves are not similar
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
| its full history | 215 | 4.73% | 2.22% | 2.13 | −2.69% |
| the overlap only | 46 | 4.46% | 1.66% | **2.68** | −0.79% |

Worst months excluded: 2020-03 (−2.69%), 2008-09 (−1.97%), 2008-11 (−1.48%).
The window inflates the sleeve's Sharpe by **+0.55**.

The important part is *where* the inflation is. The annualized return is almost
unchanged — 4.46% against 4.73% — and all of the difference is in volatility,
1.66% against 2.22%. So the quantity mean-variance actually acted on, the
**return gap**, is not a window artifact; what the window flattered is the risk.
The allocation conclusion survives. The 2.54 Sharpe does not, and the honest
number for the combined book is lower.

### 5. And it holds at every estimation window

A single lookback is a single draw, so the same move notes/allocator.md makes
for its own headline:

| lookback | OOS months | equal weight | inverse vol | risk parity | HRP | min variance | max Sharpe |
|---|---|---|---|---|---|---|---|
| 12m | 34 | 1.14 | 1.16 | 1.16 | 1.14 | 1.01 | **1.93** |
| 18m | 28 | 1.35 | 1.46 | 1.46 | 1.53 | 1.55 | **2.25** |
| 24m | 22 | 1.49 | 1.50 | 1.50 | 1.47 | 1.47 | **2.54** |
| 30m | 16 | 1.66 | 1.63 | 1.63 | 1.58 | 1.59 | **1.78** |

Max Sharpe wins at all four. The margin collapses at 30 months, where only 16
out-of-sample months are left and the comparison is close to meaningless — which
is the right place for it to collapse.

### 6. The cost the two sleeves were not matched on

Reproduce with: `python multi_strategy.py` (section 7)

Every version of this study until now carried a caveat that it could not
price: `statarb` pays bid-ask on every share it trades, while `volcarry`
charged its delta-hedging in basis points and then **sold the option itself at
the mid**. Nobody sells at the mid. So the gap between the sleeves was
measured with one of them paying its spreads and the other not, and the honest
sentence in that section said it was overstated "by an unmeasured amount". This
measures it — and every number in sections 1 to 5 above is now quoted with the
charge applied, so the 0.00 row below is what those sections used to say.

`option_spread_vol_pts` charges half the option's bid-ask on the monthly sale,
quoted in **vol points** because that is how an options market's width is
quoted and compared. The option is sold at the bid and hedged at the mid, which
is what a desk actually does — it crosses the spread to get the position on and
then marks and hedges on its own mid — and `delta_hedge` already took those two
volatilities as separate arguments, so it is one line.

The default is 0.10 vol points, and the arithmetic behind it is checkable
rather than asserted: one-month ATM vega is about `S·√(T/2π)`, which at SPY 580
and 21 trading days is **$0.67 per vol point** (the test suite verifies that
number against the pricer). SPY's front-month ATM market is a penny or two
wide, so a cent of half-spread is roughly 0.02 vol points. 0.10 is therefore
deliberately two to five times a calm day's quoted width, because this sleeve
trades every month from 2007 and option markets in October 2008 and March 2020
were not a penny wide.

The useful question is not what the spread really was — nobody has an
eighteen-year history of SPY option quotes to settle it — but **how large it
would have to be to change the answer.** That converts an unmeasured hole into
a bounded one, which is the most a study can do with a number it cannot
observe:

| half-spread (vol pts) | volcarry return | volcarry Sharpe | gap t-stat | max Sharpe | 1/N | weight on statarb: MV | IV |
|---|---|---|---|---|---|---|---|
| 0.00 (mid) | 4.59% | 2.77 | 4.57 | **2.64** | 1.57 | 0% | 43% |
| **0.10 (shipped)** | **4.46%** | **2.68** | **4.45** | **2.54** | 1.49 | 0% | 43% |
| 0.25 | 4.25% | 2.56 | 4.27 | **2.38** | 1.38 | 0% | 43% |
| 0.50 | 3.90% | 2.35 | 3.97 | **2.12** | 1.18 | 0% | 43% |
| 1.00 | 3.21% | 1.93 | 3.37 | **1.60** | 0.79 | 0% | 43% |
| 2.00 | 1.83% | 1.10 | 2.17 | −0.01 | 0.01 | 23% | 43% |
| 4.00 | −0.93% | −0.56 | −0.22 | −0.16 | −1.55 | 91% | 43% |
| 8.00 | −6.46% | −3.89 | −5.01 | −0.48 | −4.66 | 100% | 43% |

**At the realistic charge the caveat was worth 0.14% of annual return and 0.08
of Sharpe.** The return gap goes from 5.27% to 5.13%, its t-statistic from 4.57
to 4.45, the combined book's out-of-sample Sharpe from 2.64 to 2.54, and the
ranking of all six schemes is unchanged. Mean-variance still beats equal
weighting by more than a unit of Sharpe. Since the correction is small and
points the right way, the charge is now the default rather than a sensitivity,
which is why the tables above moved too.

The gap stays significant out to a half-spread of **2.00 vol points — about
twenty times SPY's front-month market** — and mean-variance keeps zero weight
on the stat-arb sleeve all the way to that point. So the objection is real,
correctly stated, and bounded at roughly 20x. That is a better answer than
either pretending the cost away or quietly removing the caveat.

Two things in that table are worth more than the headline.

**The correlation and the diversification ratio do not move at all** — 0.029
and 1.393 at every row. A monthly half-spread is very nearly a constant drag,
so it shifts a mean and leaves the covariance alone. That is the mechanical
reason a cost correction cannot reorder the risk-based allocators against each
other: it only moves the input that mean-variance is the one to read. There is
a test on each half of that.

**And the last column is section 2's finding, harder.** Inverse volatility
holds 43% of the book in the stat-arb sleeve at *every* spread — including the
rows where the other sleeve has stopped making money altogether, and the row
where it loses 6.5% a year. It is not reacting slowly to the deterioration; it
is not reacting. Section 2 established that by multiplying expected returns by
ten and watching the weights not move, which is a synthetic test. This is the
same blind spot shown on a change that actually happened to the book, and it
is a sharper way to say it: a method that never reads a return cannot notice
that something it funds has stopped working, however far it falls.

### What this does not establish

- **Two sleeves is not a portfolio.** The interesting version of this study has
  eight or ten weakly correlated sleeves, where the covariance matters as much
  as the means. With two, "allocation" is one number.
- **22 out-of-sample months.** Everything in section 1 rests on fewer than two
  years. Section 5 is the only reason to take it at all seriously, and it is a
  weak reason.
- ~~**The sleeves are not cost-matched.**~~ Measured in section 6.
  `option_spread_vol_pts` charges the option's half-spread on the monthly sale,
  and at a realistic 0.10 vol points it costs the sleeve 0.14% a year and 0.08
  of Sharpe. The gap stays significant out to twenty times SPY's quoted width.
  What is still not matched: the spread is a constant, and a real one widens in
  exactly the months this sleeve is most exposed. The sweep brackets that
  rather than modelling it.
- **Nothing here is levered to a common risk target.** Both sleeves run at
  under 2% volatility, so the combined book earns 3.5% a year at 1.3% vol. A
  real book would lever that, and leverage changes which risks matter.
