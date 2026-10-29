---
title: Alpha to Allocation
emoji: 📈
colorFrom: blue
colorTo: gray
sdk: streamlit
app_file: streamlit_app.py
pinned: false
---

# Alpha to Allocation

**[Live site &rarr;](https://mihircoding.github.io/portfolioOptimization/)** —
the efficient frontier, the cointegration scan, the volatility surface and the
cross-sleeve allocation, charted from this repository's own output. The
interactive pricer runs in the browser.

![tests](https://github.com/mihircoding/portfolioOptimization/actions/workflows/ci.yml/badge.svg)

Three strategies that produce a return stream, and the machinery that decides
how much of each to hold.

```
sleeves/statarb/     930 cointegrated equity pairs, long one leg and short the other
sleeves/volcarry/    sell a one-month ATM SPY call every month, delta-hedge it daily
sleeves/xsmom/       12-1 momentum on the same 102 names, top fifth against bottom fifth
src/                 the allocator: mean-variance, shrinkage, risk parity, CVaR,
                     a factor risk model, Black-Litterman, hierarchical risk parity
multi_strategy.py    the allocator, pointed at the sleeves
```

They were three separate projects and the third one had a hole in it. Every
result in notes/allocator.md tests the optimizer on **assets** — five ETFs,
then fifty US large caps — and finds that equal weighting beats mean-variance
out of sample at every estimation window tried. Which is a real result, and it
has always had an obvious rejoinder: a risk allocator exists to combine
**strategies**, and when everything in the book is 85% correlated with the
market there is no risk left to diversify away.

`multi_strategy.py` removes the rejoinder. The two sleeves have a correlation of
**+0.03** and a diversification ratio of **1.39** against a theoretical ceiling
of 1.41 — about as clean a two-strategy book as exists. And for the first time
in this repository the optimizer wins, by more than a full unit of Sharpe.

It does not win for the reason the textbook predicts. It wins because one of
the two sleeves **does not make money**, mean-variance can see that and risk
parity, inverse volatility, HRP and min variance cannot — they never read an
expected return, by design, and the sleeve that earns nothing is also the
*quieter* one, so all four allocate 37–43% of the book toward it. A strategy
with no edge has no edge to be volatile about, which makes "calm" a terrible
proxy for "safe."

The third sleeve was added because two is not a portfolio, and the first thing
it showed was about the other two. Cross-sectional momentum on the pairs
universe has a correlation of **−0.46** with the pairs sleeve — long the
leaders and short the laggards is the pairs trade with the sign flipped, and
a fifth of the stat-arb book's variance is a short-momentum position it never
knew it had (beta −0.04, t −3.4; the regression is section 9 of RESULTS.md,
along with the finding that the dollar-neutral momentum book is half a dollar
short the market). The second thing: equal weighting, which had won every table
in this repository, loses by 0.8 of Sharpe the moment one sleeve is ten times
the scale of the others. 1/N only ever meant equal *risk*; the earlier tables
just never contained anything that could tell the difference.

Hedging that market beta out of the momentum sleeve — with a trailing
36-month ratio, since the −0.47 was not known until the window was over —
turns it from a losing sleeve into a 0.22-Sharpe one, and the optimizer still
does not want it (section 10). The hedge rescued equal weight, from 0.69 to
1.30, and moved max Sharpe by 0.06, because mean-variance was already not
holding the sleeve and a little return is not enough to change that.

Then lever all three to the same 10% vol, on a trailing estimate shifted a
month, and the table reorders: equal weight goes from fourth to first and max
Sharpe from third to fifth (section 11). Part of that is 1/N finally
collecting the −0.41 hedge between the pairs book and the momentum book at
equal risk. The bigger part is that a trailing vol estimate is the wrong
sizing rule for a short-volatility sleeve — it de-levers after the crash
that makes the premium rich and levers up after the quiet years before the
next one, and it had the vol sleeve at 8×, its maximum in the sample, going
into the worst month. Max Sharpe held 91% of that sleeve. Cap the leverage at
5× and every allocator improves, which is a cap doing the vol forecast's job
better than the forecast.

So size that sleeve on the vol it actually sells — the implied vol of the
call, known the morning it goes on — and the half point of Sharpe comes back
on the sleeve's own history: 2.66 against 2.14 on trailing (section 12).
Nearly all of it is March 2020, where the implied rule read a 33 and had the
sleeve at 2.5× while the trailing window, full of quiet years, had it at 6×.
The rule still cannot see a spike from a quiet market — February 2018 was
sold at 13 and both rules were at 6× — and the 19 months the allocators are
tested on contain no 2020, so there a 5× cap on the trailing estimate already
delivered everything implied sizing does.

Then pay for the leverage (section 13). The pairs book at 6× was supposed to
be the sleeve that hurt; it is 2.6× gross on a book that is half in cash,
and its borrow is small change. The vol sleeve is the one with a cash leg —
its hedge is a margin loan at Fed funds — and on the 19 traded months, at
a 5.2% rate, the loan costs 8.8% a year and takes the sleeve from 2.24 to
0.90. Equal weight goes from 2.67 to 1.42 and still wins, now by half a
point over the best single sleeve rather than a tenth, because a third of
the book is in the two sleeves that cost almost nothing to run. The rate
alone, with no broker terms at all, is most of the bill.

Max Sharpe held 92% of that sleeve on net returns, estimating on a window
that was half zero-rate months, so the fix the textbook suggests is to
re-charge the window at the rate known on day one before the optimizer
sees it (section 14). It is worth a hundredth: 0.92 to 0.91 on the sleeve,
0.93 to 0.94 on the book. The sleeve after the loan is still the only thing
in the window that made money — the pairs book and the momentum book lost
over those two years at −0.92 and −0.35 — and no price of funding changes
which sleeve earned. The window has to be charged 10%, twice the 2023 peak,
before max Sharpe holds less than half, and at 12% it finally beats equal
weight, by being told the loan costs more than twice what it did. That is
not a funding forecast; it is a haircut on the winning sleeve's mean, which
is what shrinkage is.

So shrink the mean on purpose (section 15): the window's mean pulled toward
Jorion's minimum-variance return, or toward the return under which 1/N is
the answer, before the optimizer sees it. Toward min variance there is
nothing — the strength the Bayes-Stein estimator picks from the data on its
own scores 0.86, below the unshrunk 0.93. Toward 1/N there is a strength,
0.96, at which max Sharpe reads 1.62 against equal weight's 1.42, holding
20/55/25 — the first mean-variance book here to beat 1/N on levered
sleeves without being told something false. But 0.96 sits at the far end
of a dial whose scale nothing sets in advance, and it was read off the
answer. And the 12% row was never shrinkage: its book was 0/37/63, a view
against one sleeve that happened to land in the sleeve about to earn.

[RESULTS.md](RESULTS.md) has the tables, the t-statistic that says in advance
which of those two regimes you are in, and the check that matters: the overlap
window contains no 2008 and no March 2020, which inflates the volatility sleeve's
Sharpe by 0.55 — all of it in the risk, almost none in the return the optimizer
actually acted on.

341 tests.

```bash
pip install -r requirements.txt
python -m pytest -q                 # 341 passed

# all of it together
python multi_strategy.py            # allocate across the sleeves, the headline result

# the allocator
python run_optimization.py          # frontier, walk-forward, Black-Litterman, costs
python factor_study.py              # 50 large caps, where a covariance has 1,275 parameters
python hrp_study.py                 # hierarchical risk parity
python hrp_split_study.py           # and the flaw in its published algorithm

# the stat-arb sleeve
python sleeves/statarb/scan.py            # screen 4,950 pairs, trade the survivors
python sleeves/statarb/control.py         # trade the 4,020 the screen REJECTED, as a control
python sleeves/statarb/screen_validity.py # the screen's 5% test rejects 15% of the time
python sleeves/statarb/survivorship.py    # the bias priced instead of admitted
python sleeves/statarb/walkforward.py     # refit the formation window, trade out of sample

# the volatility sleeve
python sleeves/volcarry/hedging_study.py  # sell and hedge an ATM call monthly since 2007
python sleeves/volcarry/ssvi_study.py     # one arbitrage-free surface, not six smiles
python sleeves/volcarry/arbitrage_study.py # butterfly and calendar checks on real quotes

# the momentum sleeve
python -m sleeves.xsmom.momentum          # 12-1 momentum on the S&P 100, plain and sector-neutral

# the interactive pricer
python -m streamlit run streamlit_app.py
```

![Efficient frontier and out-of-sample growth](frontier.png)

## Results

| | |
|---|---|
| [RESULTS.md](RESULTS.md) | allocating across the sleeves, the blind spot in risk parity, and the one in 1/N |
| [notes/allocator.md](notes/allocator.md) | mean-variance, shrinkage, risk parity, CVaR, factor model, HRP — all out of sample |
| [notes/statarb.md](notes/statarb.md) | the cointegration screen, its control group, its size distortion, its survivorship bias |
| [notes/volcarry.md](notes/volcarry.md) | Black-Scholes from scratch, the SSVI surface, and what the hedge actually earns |
| [notes/interview-allocator.md](notes/interview-allocator.md) | how to talk about the allocator |

| [notes/interview-volcarry.md](notes/interview-volcarry.md) | how to talk about the options side |

The short version of all of it: the cointegration screen predicts almost
nothing out of sample and the sections after that measure why; the variance risk
premium is real and is the only thing here that reliably makes money; and the
optimizer loses to equal weighting on assets and beats it on strategies, for a
reason that reduces to a single t-statistic.

---

## How the three parts work

### The allocator

Given `n` assets with expected returns `μ` and covariance `Σ`, a portfolio with
weights `w` summing to 1 has expected return `wᵀμ` and variance `wᵀΣw`. Markowitz
picks the `w` that minimises variance for a given return; sweeping the return
target traces the **efficient frontier**.

The problem is that `μ` and `Σ` have to be estimated, and the optimizer treats
estimates as facts. With fifty assets a covariance matrix has 1,275 free
parameters and maybe a year of daily data to fit them with, so the optimizer
reliably finds the portfolio whose risk was *underestimated* by the sample — it
is an error-maximiser as much as a risk-minimiser. In notes/allocator.md the
sample covariance promises 9.2% risk and delivers 15.5%. Section 9 of that file
is what does and does not fix it, and the answer turns out to be a constraint
rather than a better estimator.

### The stat-arb sleeve

Two stocks are **cointegrated** if some linear combination of their prices is
stationary — the spread wanders but keeps coming back, so a wide spread is a
bet with a known direction. The Engle-Granger test regresses one on the other
and runs an ADF test on the residual.

Run it across every pair in the S&P 100 and 930 of 4,950 pairs pass at 5%. One
survives a Bonferroni correction. The mean out-of-sample Sharpe across the 930
is indistinguishable from zero, and the three sections after that one establish
why: a control group of the *rejected* pairs trades about as well, the screen's
nominal 5% test actually rejects 15% of the time, and the universe is today's
index members, so the left tail of the return distribution is missing entirely.

### The volatility sleeve

Black-Scholes prices an option as the cost of replicating it by continuously
trading the underlying. Nobody hedges continuously, so the interesting question
is what the discrete hedge misses — and the answer is that it is not noise, it
is a bet on variance:

```
P&L  =  ∫ ½ Γ S² (σ²_implied − σ²_realized) dt
```

Sell a one-month ATM SPY call every month from 2007 and delta-hedge it daily and
implied volatility averages 20.0% against 16.2% realized. The seller earns 0.405
per $100 a month at a Sharpe of 2.20, and loses 2.675 in March 2020. The gamma
decomposition reproduces the realized P&L with r = 0.99, which is what makes
"a delta-hedged option is a bet on variance" a measurement rather than a slogan.

That P&L series is what `multi_strategy.py` allocates, and the March 2020 number
is the one section 4 of RESULTS.md is about.
