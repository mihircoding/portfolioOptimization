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
| `volcarry` | sell a one-month ATM SPY call monthly, delta-hedge daily | +4.44% | 1.66% | **2.68** | −0.79% |

46 overlapping months, 2021-02 to 2024-11. Correlation between them: **+0.029**.
Diversification ratio **1.39**, against a ceiling of 1.41 for two sleeves. By
the textbook this is as good as a two-strategy book gets.

### 1. For the first time here, the optimizer wins

Walk-forward: estimate on a trailing 24 months, hold for one, repeat. 22
out-of-sample months.

| scheme | ann. return | ann. vol | Sharpe | weight on `statarb` |
|---|---|---|---|---|
| **max Sharpe** | 3.39% | 1.33% | **2.55** | **0%** |
| risk parity | 1.42% | 0.94% | 1.51 | 43% |
| inverse vol | 1.42% | 0.94% | 1.51 | 43% |
| equal weight | 1.33% | 0.89% | 1.50 | 50% |
| HRP | 1.52% | 1.02% | 1.49 | 37% |
| min variance | 1.51% | 1.02% | 1.48 | 37% |

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
return gap        +5.12% a year  (+0.4263% a month)
standard error     0.0960% a month, on 46 months
t-statistic           4.44
```

A t-statistic of 4.44 on the return difference. The sleeves are not similar
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
| its full history | 215 | 4.71% | 2.21% | 2.13 | −2.70% |
| the overlap only | 46 | 4.44% | 1.66% | **2.68** | −0.79% |

Worst months excluded: 2020-03 (−2.70%), 2008-09 (−1.98%), 2008-11 (−1.51%).
The window inflates the sleeve's Sharpe by **+0.55**.

The important part is *where* the inflation is. The annualized return is almost
unchanged — 4.44% against 4.71% — and all of the difference is in volatility,
1.66% against 2.21%. So the quantity mean-variance actually acted on, the
**return gap**, is not a window artifact; what the window flattered is the risk.
The allocation conclusion survives. The 2.55 Sharpe does not, and the honest
number for the combined book is lower.

### 5. And it holds at every estimation window

A single lookback is a single draw, so the same move notes/allocator.md makes
for its own headline:

| lookback | OOS months | equal weight | inverse vol | risk parity | HRP | min variance | max Sharpe |
|---|---|---|---|---|---|---|---|
| 12m | 34 | 1.14 | 1.16 | 1.16 | 1.14 | 1.01 | **1.93** |
| 18m | 28 | 1.35 | 1.46 | 1.46 | 1.53 | 1.56 | **2.25** |
| 24m | 22 | 1.50 | 1.51 | 1.51 | 1.49 | 1.48 | **2.55** |
| 30m | 16 | 1.67 | 1.65 | 1.65 | 1.60 | 1.61 | **1.79** |

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
which is why the tables above moved too. Section 7 then makes that default
proportional to the level of volatility, which moves them again by about a
hundredth of a Sharpe; the rows in this table are deliberately left on a
constant spread, because "how wide would it have to be" is a question about the
level and sweeping a proportional spread would answer two at once.

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

### 7. And the spread is not the same width every month

Reproduce with: `python multi_strategy.py` (section 8)

Section 6 bounded how **wide** the option spread is. It left the objection's
sharper half standing, in the words the caveat list used: *the spread is a
constant, and a real one widens in exactly the months this sleeve is most
exposed.* That is the uncomfortable version, because it says the cost and the
risk arrive together, and no sweep over constants can see it.

The mechanism is worth a sentence, because it is the reason a constant is wrong
rather than just obviously wrong. A maker's bid-ask covers the risk of being
wrong about fair value, and fair value for an option **is** a volatility. So
the width scales with the level of volatility — which is the same statement as
"spreads widen in a crisis", said in units a reader can check. One line in the
sleeve: charge `0.10 × VIX / 17.5` instead of `0.10`. That is 0.10 vol points
at VIX 17.5, 0.07 in 2017 and 0.39 in March 2020. 17.5 is VIX's long-run median
to the nearest half point, picked because it is round and because it makes the
proportional model agree with the constant one on a typical month instead of
repricing everything at once. It is a normalisation, not a fit; the ratio is
what does the work.

| spread charged | avg half-spread | volcarry return | Sharpe | gap t-stat | weight on statarb: MV |
|---|---|---|---|---|---|
| mid (none) | 0.0000 | 4.59% | 2.77 | 4.57 | 0% |
| constant | 0.1000 | 4.46% | 2.68 | 4.45 | 0% |
| **proportional to VIX** | **0.1144** | **4.44%** | **2.68** | **4.44** | **0%** |
| constant, level-matched | 0.1144 | 4.44% | 2.67 | 4.43 | 0% |

The fourth row is the control, and it is the row that makes the section an
answer rather than a demonstration. A cost that moves with the market has a
**level** and a **co-movement**, and conflating them is how a small effect gets
written up as a large one. So: a *constant* spread set to the proportional
series' own realized average — same average cost, none of the timing. It lands
on 4.44% and a t-stat of 4.43 against the proportional model's 4.44%, 4.44. **So
essentially all of the difference is the average width, and the timing is worth
about 0.01 of a t-statistic.**

Which raises the obvious question of why, and the answer is the one table here
that is better than the headline:

| months | avg VIX | premium sold | half-spread | cost / premium, proportional | cost / premium, constant |
|---|---|---|---|---|---|
| 72 | 13.0 | 1.49 | 0.074 | **0.57%** | 0.79% |
| 71 | 17.8 | 2.03 | 0.101 | **0.57%** | 0.57% |
| 72 | 29.3 | 3.35 | 0.167 | **0.57%** | 0.37% |

Per $100 of underlying, months sorted into terciles by the VIX they sold at.

The proportional column is **flat to four decimal places**, and that is the
whole result. An at-the-money option's premium is about `0.4·S·σ·√T` and its
vega is the same quantity without the σ, so the spread costs vega times the
width, the premium is vega times σ, and a width proportional to σ makes the
ratio σ-free. The sleeve pays 0.57% of what it sold in every regime there is.
There is a test on that identity rather than on the figure.

And the flip side, which is the part that reframes the objection: **the constant
spread is the regime-dependent one.** 0.79% of premium in the calm third,
0.37% in the loud third. It overcharges quiet months and undercharges exactly
the months the sleeve has the most premium at risk in — the opposite of what the
caveat worried about, and the reason the caveat was pointing at the right thing
for not quite the right reason.

**The contrast with the market-simulation repo is the part worth being able to
say out loud.** The same question — what happens when a cost moves with
volatility — was asked there of an order book's depth, and came out the other
way: 81% of the extra cost was the co-movement rather than the level, because a
moving-average crossover trades a fixed number of shares and meets a book that
thins when it happens to be trading. Here the exposure moves with the same
variable the cost does, and they cancel. One sentence covers both: **a cost's
co-movement with volatility bites you only to the extent your position size
does not co-move with it too.** That generalises further than either project —
it is why a fixed-notional trend follower and a short-vol book have opposite
sensitivities to the same liquidity fact.

What this still does not settle:

- **The exponent is 1, and not measured.** Spread proportional to vol is what
  the risk argument predicts and what the quoted-width literature reports; it
  is not fitted to an option-quote history, because this project does not have
  one. A harder exponent would make loud months worse than linearly, and the
  constant-vs-proportional control above suggests that would still mostly show
  up as a change in the average.
- **VIX is the proxy for the width, and also for the price.** The option is
  sold at VIX and the spread is a fraction of VIX, so one series carries both
  jobs. Real quoted widths are not a clean function of the index level —
  they gap on event days in a way a smooth ratio cannot produce.
- **Nothing here widens the hedge's spread.** `cost_bps` on the share hedge is
  still a constant, and equity spreads widen with volatility for the same
  reason option spreads do. That is the next version of this section, and it
  is the one place the two repositories' corrections would actually meet.

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
- ~~**And that spread is a constant, while a real one widens in exactly the
  months this sleeve is most exposed.**~~ Measured in section 7.
  `option_spread_vix_anchor` scales the width with VIX, and it turns out to cost
  0.02% a year beyond its own change in average width, because the premium sold
  scales with the same variable and the two cancel to four decimal places. The
  constant spread is the regime-dependent one. What is still not matched: the
  SHARE hedge's bid-ask is still a constant, and equity spreads widen for the
  same reason option spreads do.
- **Nothing here is levered to a common risk target.** Both sleeves run at
  under 2% volatility, so the combined book earns 3.5% a year at 1.3% vol. A
  real book would lever that, and leverage changes which risks matter.
