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

### 8. A third sleeve, so that "allocation" is more than one number

Reproduce with: `python -m sleeves.xsmom.momentum`, then `python multi_strategy.py`

Every section above has been criticised in its own caveats for the same thing:
two sleeves is not a portfolio. With two, every allocator chooses one number,
the covariance matrix has one off-diagonal entry, and the difference between
risk parity and inverse volatility is nearly definitional. Including HRP at all
was fair game — a clustering algorithm with two things to cluster is not
clustering.

`sleeves/xsmom/` is the cheapest third stream this repository could build.
Cross-sectional momentum, Jegadeesh-Titman 12-1: rank every name on its return
from twelve months ago to one month ago, long the top fifth, short the bottom
fifth, equal weight inside each leg, one dollar against one dollar, rebalanced
monthly. Costs are the same `src/costs.py` bookkeeping as everywhere else —
drifted book, 10 bps a side — and momentum turns over 53% of the book a month,
so that is a third of the gross return. The universe is the same 102 names the
pairs scan screens, which is deliberate and is the point of the section.

**On its own, over 2014-02 to 2025-12 (143 months), it earns almost nothing.**

| | ann. return | ann. vol | Sharpe | worst month | turnover / month | beta to the universe |
|---|---|---|---|---|---|---|
| plain, gross | +2.57% | 15.89% | 0.16 | −18.87% | 52.6% | −0.30 |
| **plain, net** | **+1.30%** | **15.89%** | **0.08** | **−18.95%** | 52.6% | −0.30 |
| sector-neutral, net | +1.12% | 11.10% | 0.10 | −9.08% | 53.9% | −0.11 |

That is the known fact about momentum in mega-caps and it is worth stating
rather than tuning around: the anomaly lives in small and mid caps, and on a
hundred of the largest names a twelve-month signal is one bet a year. By
calendar year the sleeve goes +2, +12, −13, +10, +14, −10, +13, −15, +10, −15,
+12, −4 percent. The sign flips nine times in eleven steps.

Its three worst months are the check that it is the real thing:

| month | xsmom | universe, that month | universe, prior 3 months |
|---|---|---|---|
| 2023-01 | −19.0% | +6.5% | +13.2% |
| 2020-11 | −13.8% | +14.3% | +2.6% |
| 2021-02 | −11.3% | +4.7% | +17.3% |

Every one is a rally. Momentum does not lose in a crash; it loses in the month
the crash reverses, because by then the short leg is full of whatever fell
hardest, which is whatever bounces hardest. Daniel and Moskowitz call it a
momentum crash, it is the one risk the strategy is famous for, and the sleeve
built here has the scar in the same place. November 2020 is the vaccine
announcement; January 2023 is the junk rally after 2022.

A word on the universe, because it cuts an unusual way here. These are the
S&P 100 constituents as of 2025, so every name survived to the end of the
sample — survivorship bias. For a long-only strategy that flatters the result.
For a momentum short leg it does the opposite: the leg is a bet that last
year's losers keep losing, and in a universe where every loser recovered enough
to still be in the index a decade later, that bet is fighting the sample. The
figure above is more likely an understatement than an overstatement. It is
still a caveat.

**On the overlap, 2021-02 to 2024-11, 46 months:**

| sleeve | ann. return | ann. vol | Sharpe | corr. w/ statarb | corr. w/ volcarry |
|---|---|---|---|---|---|
| `statarb` | −0.67% | 1.53% | −0.44 | | +0.029 |
| `volcarry` | +4.44% | 1.66% | 2.68 | +0.029 | |
| `xsmom` | −2.98% | 18.36% | −0.16 | **−0.461** | −0.146 |

**−0.46 against statarb is the number to notice.** The pairs book is long the
lagging leg of a pair and short the leading leg; momentum is long the leaders
and short the laggards. On one universe they are opposite sides of the same
trade — a bet on relative performance, with the sign flipped — and the
correlation says how far that goes: a fifth of the stat-arb sleeve's
variance is explainable as a short-momentum position (r² = 0.21 — this
paragraph first said "nearly half", which was the correlation read as a share;
section 9 has the regression). That is a fact about the pairs sleeve that
nothing in notes/statarb.md could have shown, because that file never had a
momentum series to regress against.

Diversification ratio, which is where the vol mismatch starts to matter:

| weights | two sleeves | three sleeves |
|---|---|---|
| equal | 1.39 | **1.23** |
| inverse volatility | | **2.21** |

Equal weight gets *worse* with the third sleeve. The sleeve is ten times as
volatile as the other two, so 1/N is not a third each — it is the momentum
sleeve with a small hedge attached. Size by risk instead and the ratio goes to
2.21, which is past the √3 = 1.73 that three *uncorrelated* sleeves could ever
reach. Only negative correlation gets a book over that line, and the −0.46 is
what gets it there.

**Allocated out of sample — the same 24-month walk-forward as section 1, 22
months traded, three columns:**

| scheme | ann. return | ann. vol | Sharpe | Sharpe, 2 sleeves | w statarb | w volcarry | w xsmom |
|---|---|---|---|---|---|---|---|
| max Sharpe | 3.24% | 1.29% | **2.52** | 2.55 | 2% | 98% | 0.3% |
| inverse vol | 1.63% | 1.02% | 1.60 | 1.51 | 42% | 55% | 3.4% |
| HRP | 1.54% | 1.02% | 1.50 | 1.49 | 37% | 63% | 0.3% |
| risk parity | 1.46% | 1.05% | 1.39 | 1.51 | 46% | 50% | 3.9% |
| min variance | 1.40% | 1.06% | 1.32 | 1.48 | 40% | 58% | 1.6% |
| equal weight | 3.20% | 4.64% | **0.69** | 1.50 | 33% | 33% | 33.3% |

Three things in that table.

**Equal weight loses, cleanly, for the first time in this repository.** 1.50
to 0.69, with the volatility going from 0.89% to 4.64%. Every earlier table
here said 1/N beats the optimizer, and every one of them was a table of things
at similar scale — five ETFs, fifty stocks, two sleeves under 2% vol. Hand it
one stream ten times the size of the others and it is not diversified, it is
concentrated. Inverse volatility, which is exactly 1/N after scaling, holds
3.4% in the sleeve and is back to 1.60. So the standing result survives, with
its missing condition now written down: equal weighting beats optimization
among things of comparable risk, and the moment they are not, "equal" has to
mean equal risk rather than equal dollars.

**HRP finally has something to cluster, and look at what it does with it.**
0.3% in momentum, against risk parity's 3.9%. The tree puts momentum in a
cluster of its own, which is right — it is the one sleeve negatively
correlated with the others. Then recursive bisection splits capital between
the two clusters by inverse cluster variance *alone*. The correlation that
built the tree is not used at the step that sizes the branches, so the hedge
the −0.46 offers is never taken. Risk parity reads the whole matrix and takes
it; HRP reads it, orders by it, and then forgets it. That is the method's
known blind spot (`src/hrp.py` says as much about the split rule), and this is
the first panel in the repository where it had anything to be blind to.

**Mean-variance puts 0.3% in it, for the same reason it puts 2% in statarb.**
The sleeve does not make money, and that is the one thing mean-variance can
see and the risk-based four cannot. A sleeve with no return cannot diversify a
return. It can only diversify risk, and the risk-based four all land on the
same answer for how much of it to take for that: a few percent. Which is why
the headline from section 1 does not move — 2.55 to 2.52 — and why, honestly,
the third sleeve changed the *allocation* result less than it changed what the
repository knows about its other two sleeves.

What this does not settle:

- **It is still not a portfolio.** Three sleeves where two of them earn
  nothing is a volatility sleeve with two hedges on. The interesting version
  of this study needs sleeves that *all* make money, so the allocator has to
  trade off returns against each other rather than just decline to fund the
  broken ones.
- **22 months, still.** The overlap is set by the stat-arb sleeve's start date
  and a third sleeve with 143 months of history does nothing to extend it.
- **The momentum sleeve is the textbook version and nothing more.** No
  volatility scaling, no skip of the crash months, no dynamic weighting of
  the two legs — all of which are documented fixes for exactly the crash
  pattern above. They were left out so the sleeve measures the anomaly and not
  the engineering on top of it.

### 9. The −0.46 in return units

Reproduce with: `python multi_strategy.py` (section 10 of the printout)

Section 8 found a −0.46 correlation between the stat-arb and momentum sleeves
and said nearly half of the stat-arb variance was a short-momentum position.
That sentence read a correlation as a variance share, and they are not the
same number. The share of one series' variance that another explains is the
correlation *squared*: −0.46 is 21%, a fifth and not a half. The sentence in
section 8 is corrected rather than deleted, and this section is the
regression that should have been there in the first place, so the claim has a
standard error attached to it.

Everything below is OLS on the 46-month overlap, monthly and non-overlapping,
classical standard errors. "Hedged" is the sleeve minus its beta times the
factor — what a desk that neutralized the exposure would have been left with.

**Stat-arb on momentum.**

| | beta | t | alpha / yr | t | R² | Sharpe | hedged Sharpe |
|---|---|---|---|---|---|---|---|
| statarb on xsmom | −0.038 | −3.4 | −0.79% | −1.1 | 0.21 | −0.44 | −0.58 |

The beta is four cents: a dollar of stat-arb carries four cents short of the
momentum book. That sounds like nothing until it is set against a sleeve
running at 1.5% volatility, where four cents of an 18%-vol factor is a fifth of
everything that moves. The t-statistic of −3.4 says the tilt is real on 46
months, which was the one part of section 8 that was right.

**Hedging it would have made the sleeve worse, not better.** Volatility drops
from 1.53% to 1.36%, which is what removing a fifth of the variance should do,
and the Sharpe goes from −0.44 to −0.58. The sleeve was earning money from the
tilt. Over 2021–24 momentum lost 3% a year on these names, and a book that was
short it collected about 0.11% a year for the privilege — small, but the
stat-arb sleeve's whole return over the window is −0.67% a year, so the tilt
was the least bad thing in it. Take it out and what is left is a cleaner
series with less return. That is the general lesson about hedging a factor
exposure: it only helps if the factor was pure noise to you. If you were being
paid for it, however accidentally, the hedge costs exactly what it was paying.

**Market beta of all three, on the equal-weight universe.** Every sleeve is
market-neutral on paper — a dollar long against a dollar short, or an option
delta-hedged every day. This is the check that the paper is right. The factor
is the equal-weight average of the 102 names rather than SPY, because the
books cannot hold anything outside those names and a beta to an index with
400 other stocks in it would mix "is the book neutral" with "do these names
track the index".

| sleeve | beta | t | alpha / yr | t | R² | Sharpe | market-hedged |
|---|---|---|---|---|---|---|---|
| statarb | +0.029 | +2.1 | −1.14% | −1.4 | 0.09 | −0.44 | −0.78 |
| volcarry | +0.034 | +2.3 | +3.90% | +4.6 | 0.11 | +2.68 | +2.49 |
| xsmom | **−0.473** | −3.0 | +4.59% | +0.5 | 0.17 | −0.16 | +0.27 |

Two of the three are what the paper says. The volatility sleeve's beta of
0.03 is the short call being short the market on the way down whatever the
delta said at the open — and its alpha is 3.9% a year at a t of 4.6, which is
the variance risk premium surviving a market factor. The stat-arb beta of 0.03
is small and, as the next table shows, is not really its own.

**The momentum book is dollar-neutral and half a dollar short the market.**
A beta of −0.47 with a t of −3.0, on a book built to be a dollar long against
a dollar short. This is the known thing about momentum and it is worth being
able to explain: the signal ranks on trailing returns, so after a drawdown the
short leg fills with the names that fell hardest, which are the high-beta
names, and the long leg fills with whatever held up, which are the low-beta
names. Dollar-neutral then means beta-negative, and the size of it moves with
the cycle (Grundy and Martin 2001 is the reference). On 2021–24, a window the
market spent mostly going up, that is where the sleeve's −3% a year went:
hedge the market out and the Sharpe goes from −0.16 to +0.27. Which is also
the mechanical content of section 8's "momentum crash" table — the three
worst months were all rallies, and a book half a dollar short the market
loses in a rally by construction.

**Both factors at once.**

| sleeve | β market | t | β xsmom | t | alpha / yr | t | R² |
|---|---|---|---|---|---|---|---|
| statarb | +0.013 | +0.9 | −0.034 | −2.7 | −0.98% | −1.3 | 0.23 |
| volcarry | +0.033 | +2.1 | −0.001 | −0.1 | +3.90% | +4.6 | 0.11 |

The stat-arb sleeve's market beta goes from 0.029 (t 2.1) to 0.013 (t 0.9)
once momentum is in the regression. Its market exposure was never its own: it
was short a book that was short the market, which nets to a small long. The
volatility sleeve has no momentum exposure at all (t −0.1) and keeps its market
beta, so the two sleeves' small positive betas have different causes even
though they are the same size — one is a hedge-timing residual, the other is
a factor exposure once removed.

What this changes in the earlier sections: the correlation in section 8 was
real and the inference drawn from it was overstated by a factor of two and a
half. The allocator tables are unaffected — they read the covariance matrix,
and the covariance matrix was always right about the size of the comovement.
What was wrong was the sentence describing it. And the "what this does not
settle" list in section 8 gets one more entry, because it is now measurable:
the momentum sleeve as built has a market beta of −0.47, and a desk would
beta-hedge it before allocating to it, which would change the panel the
allocators see. Section 10 runs that.

### 10. Beta-hedge the momentum sleeve before it goes in

Reproduce with: `python multi_strategy.py` (section 11 of the printout)

Section 9 ended on a measurable question. The momentum book as built carries
a market beta of −0.47, a desk would hedge that before allocating to the
sleeve, and the hedge changes the panel every allocator above has been
looking at. This section runs it.

**The hedge has to be out of sample, and that rules out the −0.47.** It is a
full-window number: fitted on the same 46 months it would then be used to
hedge, which is the look-ahead the walk-forward in section 1 was built to
avoid. What a desk has at the start of a month is the beta the trailing
window said last month. So the hedge ratio here is a 36-month OLS beta on
the sleeve's own prior history, re-estimated every month, shifted one month
so nothing it uses is a return it is about to be applied to. The momentum
sleeve runs from 2014, so the first full window closes in early 2017 and a
ratio exists for every month of the overlap. `trailing_beta` is the
function, and the test that matters is the one that changes the last month
of the data and checks no earlier beta moves.

| | |
|---|---|
| full-window beta, section 9 | −0.47 (t −3.0) |
| trailing 36-month beta, mean over the overlap | −0.40 |
| range | −0.24 to −0.67 |
| at the start of the window / at the end | −0.24 / −0.51 |
| months with a negative hedge ratio | 46 of 46 |

The ratio a desk could have run was never the one the regression reports.
It starts at half the full-window figure and ends above it, and the only
thing stable about it is the sign — negative in every one of the 46 months,
which is the Grundy–Martin mechanism from section 9 showing up as a fact
about the path rather than an average. Hedging with −0.47 throughout would
have been a backtest hedging with a number it did not have yet.

**What the hedge does to the sleeve.**

| | ret / yr | vol | Sharpe | worst month |
|---|---|---|---|---|
| own history, 2017-02 to 2025-12, raw | +1.85% | 16.96% | +0.11 | −18.95% |
| same, hedged | **+6.64%** | 16.73% | **+0.40** | −16.63% |
| overlap, 46 months, raw | −2.98% | 18.36% | −0.16 | −18.95% |
| overlap, hedged | **+3.85%** | 17.52% | **+0.22** | −16.63% |

The return moves and the volatility does not, which is what hedging a factor
that *lost* money over the window should do: the sleeve was short a market
that went up, the short was costing it about 7% a year, and taking the short
out gives that back without taking much variance with it (the market
explained 17% of the sleeve's variance, so the vol can only fall by about
that much in the limit). On the overlap the sleeve goes from a losing
strategy to one with a Sharpe of 0.22, which is roughly what the literature
says mega-cap momentum earns net of costs — nothing to run a book on, but
not the negative number the allocators have been shown since section 8.

**The hedge is a trailing indicator, and the worst months show it.** The
beta measured on the hedged sleeve after the fact is −0.09, not zero. A
36-month window lags a beta that moves with the cycle, and it lags most
exactly when it matters: the ratio going into the sleeve's three worst
months was −0.36 (January 2023), −0.05 (November 2020) and −0.24 (February
2021). The November 2020 hedge was a twentieth of the full-window figure,
because the window ending October 2020 was mostly a bull market in which
the book's short leg was not yet full of high-beta names. The beta the
sleeve has in a crash reversal is built *during* the drawdown that precedes
it, and a trailing window cannot see a drawdown that has just started. The
worst month improves from −18.95% to −16.63%, which is the hedge helping
with a ratio a quarter of the size it should have been.

**What it does to the correlation matrix.** Stat-arb against momentum goes
from −0.46 to −0.39 once momentum is hedged, and volatility-carry against
momentum from −0.15 to −0.08. A sixth of the comovement between the pairs
book and the momentum book was the two of them sharing a market exposure
with opposite signs, not the relative-performance bet the write-up
describes. The rest of it — the −0.39 — is that bet, and it survives.

**And what it does to the allocators, which is the question.** Same 22
months, same 24-month window, the only change is the third column of the
panel.

| scheme | Sharpe, raw | hedged | vol, raw | hedged | weight on xsmom, raw | hedged |
|---|---|---|---|---|---|---|
| equal weight | +0.69 | **+1.30** | 4.64% | 5.05% | 0.33 | 0.33 |
| inverse vol | +1.60 | +1.90 | 1.02% | 1.06% | 0.03 | 0.04 |
| risk parity | +1.39 | +1.72 | 1.05% | 1.10% | 0.04 | 0.04 |
| HRP | +1.50 | +1.54 | 1.02% | 1.02% | 0.00 | 0.00 |
| min variance | +1.32 | +1.45 | 1.06% | 1.05% | 0.02 | 0.01 |
| max Sharpe | +2.52 | +2.58 | 1.29% | 1.27% | 0.00 | 0.01 |

Equal weight gains 0.6 of Sharpe and nobody else gains more than 0.3, and
the reason is the weight column: equal weight is the only scheme holding a
third of the momentum sleeve, so it is the only scheme whose P&L the hedge
can move much. The risk-based four hold 0–4% of it, because a 17%-vol
stream next to two 1.5%-vol streams gets a 17-to-1 haircut whatever its
return is, and their improvement is that small position earning +4% instead
of −3%. Max Sharpe does not move at all — 0.3% of the book before, 0.6%
after — and that is the finding. Mean-variance was already ignoring the
momentum sleeve, and it goes on ignoring it once hedged, because a
0.22-Sharpe sleeve does not earn a seat next to a 2.7-Sharpe sleeve at any
weight the covariance can justify. Hedging turned the sleeve from "loses
money" into "earns a little", and a little is not enough.

So the section 1 story is intact. The optimizer wins here because it reads
returns and the risk-based schemes do not, and the hedge, which changed the
returns of the sleeve the optimizer was already not holding, changed the
risk-based schemes more than it changed the optimizer. Equal weight is the
allocator the hedge rescued, from 0.69 to 1.30, and it is still last among
the six.

Two things this does not charge for. A desk short-hedging with index futures
pays financing on the short and a roll every quarter, which the hedged series
does not; and the hedge instrument here is the equal-weight average of the
102 names, which is not tradable — a real hedge would be an index future
whose beta to this basket is not one, so the residual beta of −0.09 would be
larger in practice.

### 11. Lever every sleeve to the same risk, and 1/N wins

Reproduce with: `python multi_strategy.py` (section 12 of the printout)

The last unticked item on the list below. The sleeves run at 1.6%, 1.5% and
17% volatility, and section 8 showed what that does to the one allocator that
does not read a covariance: equal weight was a momentum book with two small
positions attached, and it lost by 0.8 of Sharpe for that reason alone. Every
multi-strategy desk sizes its sleeves to a risk budget before any allocator
sees them. So this does that, with the same discipline as the hedge in
section 10: each sleeve is scaled month by month to a **10% annualized vol
target** using a trailing estimate, shifted one month, so nothing in the
scale factor has seen the return it multiplies. `trailing_vol` and
`vol_targeted` are the functions; the test that matters changes the last
month of the data and checks no earlier leverage moves.

The estimate is a 36-month window on the two sleeves that only exist
monthly, which both have history back before the overlap. The pairs book
does not — it starts in January 2021, and a 36-month window on its monthly
returns would close in 2024 and leave nothing to allocate. It has 1,250 daily
returns, though, and 63 of them pin a vol down better than 36 monthly ones
do, so `statarb_trailing_vol` reads the sleeve's daily equity curve and
stamps each month with the estimate at the close of the month before. That
window fills in April 2021, so the panel is three months shorter than
section 10's at the start: 43 months, 19 of them traded. The momentum sleeve
goes in already beta-hedged, because that is where section 10 left it.

**What the rule asks for, and what it does to each sleeve on its own.**

| sleeve | vol, raw | scaled | Sharpe, raw | scaled | worst month, raw | scaled | leverage, mean | range |
|---|---|---|---|---|---|---|---|---|
| statarb | 1.57% | 8.77% | −0.53 | −0.56 | −1.17% | −7.06% | 5.7× | 3.2× to **16.2×** |
| volcarry | 1.50% | 7.91% | **+2.56** | **+2.26** | −0.79% | −6.36% | 4.9× | 3.5× to 8.0× |
| xsmom | 17.00% | 9.54% | +0.43 | +0.40 | −16.63% | −9.67% | 0.6× | 0.5× to 0.6× |

Two things in that table before the allocators see it.

The first is that the rule asks the pairs book for 16× in its quietest month.
Nobody runs a pairs book at 16×, and the realized vol lands at 8.8% rather
than 10% because the months the rule levers hardest are not the months that
turn out loud. The leverage going into the sleeve's worst scaled month
(October 2021) was 7.7×, against 4.7× going into its worst raw month: the
scaled worst month is not the raw one, it is the one the rule walked into
with the most leverage on.

The second is that **vol-targeting costs the best sleeve 0.3 of Sharpe**, on
the overlap (2.56 to 2.26) and on its own history since 2010 (2.45 to 2.14).
Volatility-managed portfolios are supposed to help — the Moreira–Muir result
is that scaling the market by inverse trailing variance raises its Sharpe,
because vol clusters and return does not. A short-volatility sleeve is the
case where that logic runs backwards. Its trailing vol is highest right after
a crash, which is when the premium it sells is richest, and lowest after
three quiet years, which is when the premium is thinnest and the next spike
is nearest. The rule de-levers into the good months and levers into the bad
ones, and the worst month in the sample says so. The July 2024 trade sold a
call at a VIX of 12 on the first of the month and hedged it through a VIX of
18 three weeks later, and the ratio going into it was **8.0×, the maximum the
rule reached in 43 months**, because the 36-month window ending June 2024
held nothing but quiet. The next trade, sold at 18.6 on August 1 — the
session before the August 5 spike — broke even: that premium was priced for
what came. The July one was not, and the rule had the sleeve at its largest
exactly there. A vol forecast that is most confident after the quietest
three years is most confident when it is about to be most wrong.

**And what it does to the allocators.** Same 19 months, same 24-month
estimation window, raw sleeves against levered. The third column caps every
sleeve at 5× — a prime broker has a view on 16× — and the cap binds in 56% of
the pairs book's months and 47% of the vol sleeve's.

| scheme | Sharpe, raw | levered | capped 5× | vol, raw | levered | weight on statarb / volcarry / xsmom, levered |
|---|---|---|---|---|---|---|
| equal weight | +1.78 | **+2.51** | **+2.67** | 4.73% | 3.86% | 0.33 / 0.33 / 0.33 |
| inverse vol | +2.51 | +2.22 | +2.46 | 0.86% | 4.56% | 0.29 / 0.46 / 0.24 |
| risk parity | +2.46 | +2.23 | +2.43 | 0.85% | 4.31% | 0.31 / 0.43 / 0.26 |
| HRP | +1.61 | +1.92 | +2.19 | 0.95% | 5.46% | 0.25 / 0.59 / 0.16 |
| min variance | +1.75 | +1.98 | +2.20 | 0.91% | 5.00% | 0.28 / 0.53 / 0.19 |
| max Sharpe | +2.26 | +1.94 | +2.26 | 1.26% | 8.18% | 0.04 / **0.91** / 0.05 |

Equal weight goes from fourth to first and max Sharpe from third to fifth.
Three things to say about that, in order of how much they explain.

*The best sleeve was damaged by the scaling, and max Sharpe holds 91% of
it.* Section 10 guessed the optimizer would stop concentrating in the vol
sleeve once the others were at a comparable scale. It did not: 97% before,
91% after, because mean-variance reads returns and the pairs sleeve still
loses money at any scale. So max Sharpe's Sharpe is the vol sleeve's Sharpe,
and the vol sleeve's Sharpe is what the trailing window did to it. Cap the
leverage at 5× and the sleeve is back to 2.55 and max Sharpe is back to 2.26
— exactly its raw number — and still fourth of six.

*At equal risk, a third of a losing sleeve that hedges another is worth
holding.* Equal weight's levered volatility is 3.86%, *lower* than its raw
4.73%, from three sleeves that each run near 9%. Three uncorrelated sleeves
at that scale would combine to 5.2%; the −0.41 between the pairs book and
the momentum book takes it to 3.9%. That is the diversification section 8
measured and no allocator except 1/N has been collecting, because the
risk-based schemes size by the diagonal and the optimizer will not hold a
sleeve with a negative mean whatever it hedges. The pairs sleeve lost 4.9% a
year levered; it earned its place in the equal-weight book anyway, and the
capped equal-weight worst month is −1.43% against max Sharpe's −3.91% at
twice the vol.

*The risk-based four became equal weight.* Inverse vol and risk parity hold
29/46/24 and 31/43/26, which is 1/N with a tilt toward the one sleeve whose
scaled vol came in under target. They lost Sharpe against their own raw
columns — 2.51 to 2.22, 2.46 to 2.23 — for the same reason max Sharpe did:
the raw versions held 40–56% of a 1.5%-vol stream, which at a common scale is
a 46% position in the sleeve the rule damaged. A scheme that sizes by inverse
vol on top of sleeves already sized by inverse vol is doing the same thing
twice, and the second pass can only add the estimation noise of a second
window.

**Three months moved the table before any of this happened.** The raw column
here is not section 10's hedged column, and the difference is instructive:
dropping February–April 2023 from the traded months — the pairs book's daily
window filling — moved max Sharpe from 2.58 to 2.26 and equal weight from
1.30 to 1.78, with no change to any sleeve or any allocator. A ranking that
reorders on three months is what the "22 out-of-sample months" caveat below
looks like in a table, and it applies to this section as much as to the
ones above it. The direction of the result — 1/N first at a common scale,
last at a mismatched one — is the thing to take from it, not the second
decimal.

What this charges for and does not. The leverage is free: no financing on
the 5× to 16× the rule asks for, no margin on the short legs, no borrow. A
pairs book levered 6× pays for it, and a vol sleeve levered 5× is a margin
call in the month it is levered into. The cap column is the one to read as a
desk would, and even it is unfinanced.

### 12. Size the vol sleeve on the vol it sells, and the trailing rule's loss comes back — in one month

Reproduce with: `python multi_strategy.py` (section 13 of the printout)

Section 11 ended on a diagnosis: a trailing estimate is the wrong sizing
signal for a short-volatility sleeve, because it is lowest after the quiet
years that precede a spike and highest after the crash that makes the
premium rich. The obvious test is to size on the vol the sleeve is actually
selling. The call goes on at the first session of the month at a known
implied vol, so that number is the one thing about month *t* a desk has in
hand on day one — no window, no shift, no argument about lookahead.
`volcarry_implied` reads it off the trade log and `implied_vol_forecast`
turns it into a forecast of the sleeve's own vol:

    var_t = β_{t−1} · implied_t²

with β the ratio of the sleeve's summed squared returns to its summed squared
implieds over every month before *t* — an expanding regression through the
origin, so the ratio at *t* has not seen month *t*, and the only thing about
month *t* in the forecast is the implied. The sleeve's P&L is vega times the
gap between the vol sold and the vol that happened, so its scale should move
with the level of vol, and the ratio is what says by how much. It needs 36
months to fit, which puts its first forecast in January 2010, where the
36-month trailing rule's is, so the two run on the same 179 months.

**On the sleeve's own history, 2010–2024.** Both rules target 10%, both
uncapped; the capped column is the 5× cap of section 11. The last three
columns are the leverage each rule had on going into the sleeve's three
worst raw months.

| rule | Sharpe | capped 5× | vol | leverage, mean | range | corr(leverage, \|return\|) | Mar 2020 | Aug 2011 | Feb 2018 |
|---|---|---|---|---|---|---|---|---|---|
| none | +2.45 | | 1.87% | | | | | | |
| trailing 36-month | +2.14 | +2.37 | 10.81% | 5.5× | 2.9× to 10.4× | −0.15 | **6.0×** | 3.1× | 6.4× |
| implied, expanding β | **+2.66** | **+2.67** | 7.72% | 5.0× | 1.4× to 8.8× | **−0.41** | **2.5×** | 3.5× | 6.2× |

The prediction was that sizing on implied would put the Sharpe back above
2.5 at a constant risk. It does: 2.66, against 2.14 on trailing and 2.45
unsized, so the rule a desk would run beats the rule section 11 ran by half
a point of Sharpe over fifteen years, and beats no rule at all. The
correlation column is the mechanism. A sizing rule exists to be small in
the big months, so that number should be negative, and the implied rule's
is nearly three times the trailing rule's: it anticipates the big months
and the trailing rule remembers them.

Then the last three columns say where the half point came from, and it is
one month. The March 2020 trade sold a call on March 2 at an implied of 33
— the week of February 24 had already happened — and lost 2.7% raw, the
worst month in the sample. The trailing window ending February 2020 held
three quiet years and had the sleeve at 6.0×; the implied rule read the 33
and had it at 2.5×. That is the difference between a −16.2% month and a
−7.0% one, and it is most of the difference between 2.14 and 2.66. Take
2020 out and the two rules are 2.52 and 2.91, with the unsized sleeve at
2.85 between them: the implied rule still wins, by less, and the trailing
rule still loses to not sizing at all.

The other two columns are the limit of the method. August 2011 was sold at
24 after the debt-ceiling week, and both rules had it at 3×. February 2018
was sold at 13 on the first of the month and the VIX tripled on the fifth;
both rules had it at 6×, because an implied of 13 is a quiet market and
nothing about the morning of February 1 said otherwise. **The implied rule
knows about a crash that has started. It does not know about one that has
not**, and neither does anything else that is known on day one. The same
holds for July 2024, section 11's worst month on the overlap: sold at 12,
the implied rule had it at 6.9× against the trailing rule's 8.0×. Less, not
little.

One more number in that table to be honest about: the implied rule's
realized vol is 7.7%, not 10%. The ratio is a ratio of two sums of squares,
so it is dominated by the months that have the most of them, and 2008 sits
in every expanding window from 2010 on. In a quiet year the forecast runs
hot and the sleeve is undersized by a quarter. The rule got the Sharpe
right and the level wrong, and the level is what the allocators see.

**On the overlap, and what the allocators make of it.** The 19 traded
months are all after April 2021, and nothing in them is March 2020.

| | raw | trailing | implied | trailing, capped | implied, capped |
|---|---|---|---|---|---|
| volcarry Sharpe, overlap | +2.56 | +2.26 | +2.42 | +2.55 | +2.55 |
| volcarry vol, overlap | 1.50% | 7.91% | 6.53% | | |

| scheme | Sharpe: trailing | implied | trailing, capped 5× | implied, capped 5× | weight on volcarry, implied |
|---|---|---|---|---|---|
| equal weight | +2.51 | +2.51 | **+2.67** | +2.60 | 0.33 |
| inverse vol | +2.22 | +2.26 | +2.46 | +2.41 | 0.49 |
| risk parity | +2.23 | +2.25 | +2.43 | +2.37 | 0.46 |
| HRP | +1.92 | +1.97 | +2.19 | +2.12 | 0.65 |
| min variance | +1.98 | +2.02 | +2.20 | +2.14 | 0.59 |
| max Sharpe | +1.94 | +1.99 | +2.26 | +2.23 | 0.92 |

Uncapped, every allocator gains 0.04 or 0.05 from the better sizing rule,
which is the sleeve's 2.26 going to 2.42 passed through whatever weight it
holds. Capped, every allocator *loses* 0.03 to 0.07, and the reason is the
level, not the Sharpe: with the cap on, the sleeve's Sharpe is 2.55 under
either rule, but the implied rule runs it at 6.5% vol where the trailing
rule ran it at 7.9%, so a third of it is a smaller position in the best
sleeve and equal weight collects less of it. On these 19 months a 5× cap on
a trailing estimate already delivered everything implied sizing delivers,
because the months that separate the two rules are not in the sample. The
ranking is untouched: 1/N first, max Sharpe at 92% in the sleeve.

What to take from this, in order. The rule a desk would run is the implied
one; it is worth half a point of Sharpe on the sleeve over fifteen years and
all of it is in the one month the overlap does not contain. The rule's
failure mode is the one every sizing rule has — it cannot see a spike from a
quiet market — and it shares that with the cap, which is why the cap did as
well. And the thing the implied rule gets wrong, the level, is a one-number
calibration problem (a β that forgets 2008 would fix it and would also have
been at 6× into March 2020), which is the kind of trade-off a vol desk
argues about and this study now has a table for. The leverage is still
free: no financing, no margin, no borrow, on either rule.

### What this does not establish

- ~~**Two sleeves is not a portfolio.**~~ Partly answered in section 8. A
  third sleeve — cross-sectional momentum on the same universe — gives the
  covariance something to do, and what it mostly does is show that the pairs
  sleeve carries a short-momentum position (correlation −0.46; section 9
  puts it at a fifth of the variance, not the half this bullet first said) and that 1/N
  loses the moment the sleeves are not at comparable scale. Three sleeves where
  two earn nothing is still not the eight-or-ten-sleeve book this study wants.
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
- **The hedge in section 10 is free and the instrument is not tradable.**
  No financing on the short, no roll, and the factor is the equal-weight
  universe rather than a future on it. Both make the hedged sleeve look a
  little better than a desk would find it.
- ~~**Nothing here is levered to a common risk target.**~~ Done in section
  11, and it reorders the table: at 10% vol each, equal weight goes from
  fourth to first and max Sharpe from third to fifth, mostly because a trailing vol estimate
  de-levers a short-vol sleeve after the crash that makes it rich and levers
  it up after the quiet years that precede the next one — 8× going into the
  worst month in the sample. Section 12 sizes that sleeve on the implied vol
  it sells instead, which is known on day one, and gets the half point of
  Sharpe back on the sleeve's own history — in March 2020, which the overlap
  does not contain, so the allocators barely move. What is still not charged:
  the leverage is free. No
  financing, no margin, no borrow on 5× to 16×, and the pairs book's 16× is
  not a number a prime broker would allow, which is why the capped column
  exists.
