# Results

All 73 tests pass (`python -m pytest -q`). Numbers below are `python run_optimization.py`,
except section 9, which is `python factor_study.py`.

Universe: SPY (US equity), EFA (intl equity), AGG (bonds), GLD (gold), VNQ (REITs).
Data: 2007-01-03 → 2024-12-30, 4,529 trading days. Risk-free rate assumed 0 throughout.

---

## 1. In sample — max-Sharpe wins, as it must

| Portfolio | Return | Vol | Sharpe | Weights |
|---|---|---|---|---|
| Equal weight | 7.78% | 14.15% | 0.55 | 20% each |
| Inverse vol | 6.03% | 8.83% | 0.68 | SPY 14 / EFA 12 / AGG 50 / GLD 16 / VNQ 9 |
| Min variance | 3.75% | 5.22% | 0.72 | SPY 7 / EFA 0 / **AGG 91** / GLD 2 / VNQ 0 |
| **Max Sharpe** | 5.83% | 6.50% | **0.90** | SPY 21 / EFA 0 / AGG 63 / GLD 16 / VNQ 0 |
| Equal risk contribution | 5.61% | 7.63% | 0.74 | SPY 10 / EFA 9 / AGG 57 / GLD 16 / VNQ 7 |

This table proves nothing. Max-Sharpe is *defined* as the in-sample argmax — reporting that it
won in sample is reporting that the optimizer ran.

Two things are worth noticing, though. Both optimized portfolios **zero out EFA and VNQ
entirely**. That's the corner-solution behavior: mean-variance doesn't hedge its bets, it picks.
And min-variance puts **91% in a single asset** — a portfolio built by a diversification
algorithm that is barely diversified, because minimizing variance with no return constraint just
finds the lowest-vol thing available.

## 2. Risk contributions — weights lie

| Portfolio | SPY | EFA | AGG | GLD | VNQ |
|---|---|---|---|---|---|
| Equal weight | 25% | 28% | **1%** | 9% | **37%** |
| Min variance | 7% | 0% | 91% | 2% | 0% |
| Max Sharpe | 43% | 0% | 32% | 25% | 0% |
| Equal risk contribution | 20% | 20% | 20% | 20% | 20% |

Equal weight holds 20% of its *capital* in bonds and takes **1% of its risk** from them. VNQ is
also 20% of capital and **37% of risk**. The portfolio that looks most balanced is, in risk
terms, a leveraged bet on real estate and equities with a rounding error in bonds.

This is the single most useful number in portfolio reporting, and it's why "asset X is 4% of the
book and 38% of the risk" is a sentence risk managers say constantly.

## 3. Shrinkage

α = 0.3, shrinking toward the diagonal:

| | Weights |
|---|---|
| Min-var on raw Σ | SPY 7% / EFA 0% / AGG 91% / GLD 2% / VNQ 0% |
| Min-var on shrunk Σ | SPY 6% / EFA 2% / AGG 88% / GLD 4% / VNQ 0% |
| L1 weight change | 0.0737 |

Condition number of Σ: **54.9 → 44.7**.

Shrinkage lifts the smallest eigenvalue. Those near-zero eigenvalue directions are the
worst-estimated parts of the covariance matrix and precisely where an optimizer piles in — a
spuriously low variance looks like free risk reduction from inside the objective function.

The weight change is modest here because five liquid ETFs is an *easy* estimation problem: 15
covariance parameters from 4,529 observations. At 50 assets you're estimating 1,275 parameters,
the sample matrix is closer to singular, and shrinkage stops being an improvement and becomes a
requirement.

## 4. Walk-forward — the result that matters

Estimate μ and Σ on a 3-year trailing window, hold for the next calendar year, roll. 15 annual
rebalances, 2010–2024. Fully out of sample.

| Portfolio | Ann. return | Ann. vol | Sharpe | Worst year | Turnover |
|---|---|---|---|---|---|
| Black-Litterman (momentum view) | 8.75% | 8.65% | **1.01** | −12.53% | **21.8%** |
| Black-Litterman (no views) | **10.78%** | 10.86% | 0.99 | −15.21% | 0.3% |
| **Equal weight** | 7.80% | 9.46% | 0.82 | −14.21% | **0.0%** |
| Inverse vol | 5.79% | 7.32% | 0.79 | −12.67% | 3.5% |
| Equal risk contribution | 5.42% | 7.08% | 0.77 | −12.09% | 4.2% |
| Max Sharpe | 6.65% | 8.89% | 0.75 | −11.70% | 16.2% |
| Min variance | 3.32% | 5.74% | 0.58 | −13.04% | 2.3% |

The two Black-Litterman rows are new and they top the table. **Do not believe them yet** —
section 8 takes them apart, and most of that margin turns out to belong to the starting point
rather than to the method. The rest of this section is about the five rows below them, whose
ranking is unchanged.

**Max Sharpe went from 1st in sample (0.90) to 4th out of sample (0.75). Equal weight went from
5th (0.55) to 1st (0.82).**

The portfolio that required no estimation, no optimizer, no covariance matrix, and no trading
beat the one that used all four. This is DeMiguel, Garlappi & Uppal (2009) reproduced on a
different universe and a later sample.

Turnover is the mechanism made visible. Max Sharpe rewrote **16.2% of the book every year**,
chasing an optimum that moved because the estimates moved, not because the world did. Equal
weight traded nothing. Note carefully that turnover is *not* the explanation for the performance
gap — at realistic ETF costs, 16% one-way turnover is a couple of basis points a year. The gap
is estimation error, and the turnover is a symptom of it, not the cause.

The two risk-based methods (inverse vol, ERC) land in the middle with far lower turnover than
max-Sharpe and much better returns than min-variance. That's the practical case for them: most
of the benefit of optimizing, without the input you can't estimate.

## 5. Trading the turnover away

Section 4 charges max-Sharpe with 16.2% one-way turnover a year and blames that turnover for
nothing in particular - the text there is explicit that at realistic ETF costs, 16% is a couple
of basis points and the real gap is estimation error. This asks the question directly instead of
asserting the answer: what happens to Sharpe if the optimizer is actually charged for trading?

Same walk-forward, same rebalance dates, but `max_sharpe_turnover_penalized()` adds
`penalty * sum((w - w_prev)^2)` to the objective - a quadratic cost for moving away from last
year's holding, in the same units as squared Sharpe. penalty=0 reproduces the section 4 row
exactly (16.2% turnover, Sharpe 0.75), which is the check that the penalty term is wired in
correctly and not just decorative.

| Penalty | Ann. return | Ann. vol | Sharpe | Turnover |
|---|---|---|---|---|
| 0 (plain max-Sharpe) | 6.65% | 8.89% | 0.75 | 16.2% |
| 5 | 5.24% | 6.94% | **0.76** | 5.9% |
| 15 | 4.75% | 7.05% | 0.67 | 3.8% |
| 40 | 4.37% | 7.31% | 0.60 | 2.8% |
| 100 | 4.05% | 7.32% | 0.55 | 1.2% |

A small penalty is close to free: turnover drops by two-thirds (16.2% → 5.9%) for a Sharpe that
is, if anything, marginally *better* (0.75 → 0.76) — well within the noise given 15 annual
observations, so read that as "no worse," not "improved." Past that point it is a real trade-off:
by penalty 100, turnover is down to 1.2% but Sharpe has fallen to 0.55, worse than min-variance.

So "penalize turnover in the objective" — item 4 on the old build-next list below — is not a
free lunch in general, but there is a cheap first step available: a light penalty kills most of
the unnecessary trading without giving up return. It still does not touch the actual problem,
which is that mu is poorly estimated; it just makes the optimizer trade less on a bad estimate
rather than fixing the estimate.

## 6. Is this an artifact of the 3-year window?

Out-of-sample Sharpe at four estimation windows:

| Portfolio | 2y | 3y | 5y | 7y |
|---|---|---|---|---|
| **Equal weight** | **0.87** | **0.82** | **0.75** | **0.69** |
| Inverse vol | 0.87 | 0.79 | 0.67 | 0.63 |
| Equal risk contribution | 0.84 | 0.77 | 0.62 | 0.62 |
| Max Sharpe | 0.79 | 0.75 | 0.55 | 0.62 |
| Min variance | 0.66 | 0.58 | 0.41 | 0.40 |

**Equal weight wins at every window. Max-Sharpe is fourth at three of four and third at one.**
The ranking is essentially stable, which is what turns one result into a finding — a conclusion
that flipped between a 3-year and a 5-year lookback would be a statement about the lookback.

More data did not help. Going from 2 to 7 years of estimation made *every* method worse. That's
the mean-return estimation problem in one row: a longer window buys precision about a parameter
that isn't stable, so you estimate a stale quantity more accurately.

---

## 8. Black-Litterman, and how much of it is the anchor

Section 4 has Black-Litterman first and second by out-of-sample Sharpe, ahead of equal weight,
at every estimation window tested:

| Portfolio | 2y | 3y | 5y | 7y |
|---|---|---|---|---|
| Black-Litterman (momentum) | 1.09 | 1.01 | 0.95 | 0.83 |
| Black-Litterman (no views) | 1.05 | 0.99 | 0.95 | 0.82 |
| Equal weight | 0.87 | 0.82 | 0.75 | 0.69 |
| Max Sharpe | 0.79 | 0.75 | 0.55 | 0.62 |

That looks like the answer to this whole document. It mostly isn't, for a reason worth being
blunt about.

**The anchor is contaminated.** Black-Litterman needs a market portfolio to reverse-optimize,
and the one used here is the five ETFs' net assets — SPY is 65% of the total. But SPY is 65%
*today*, and a large part of why is that US equities beat the other four over exactly the
2010–2024 window being tested. The equilibrium anchor was built, in part, out of the answer.
That is not a rounding error in a walk-forward; it is information from after the fact sitting
in the most important input.

So run it again from anchors that know nothing about the outcome:

| Anchor | Weights | BL, no views | BL, momentum |
|---|---|---|---|
| Fund AUM (today's) | 65/6/11/12/6 | 0.99 | 1.01 |
| Textbook global market | 40/15/30/5/10 | 0.90 | 0.92 |
| Equal weight anchor | 20/20/20/20/20 | 0.83 | 0.79 |

Equal weight over the same walk-forward is 0.82.

Three things fall out of that table.

**Most of the edge was the anchor.** Going from today's fund sizes to a plausible outcome-blind
global market allocation costs 0.09 of Sharpe. Going all the way to an equal anchor costs 0.16
and lands on top of equal weight — which is what should happen, since Black-Litterman with an
equal anchor and no views is equal weight with a covariance-shaped nudge. The method does not
manufacture an edge; it inherits whatever the anchor had.

**Some of it isn't.** The textbook-global anchor is an honest, publicly-stated allocation that
nobody chose with hindsight, and it still clears equal weight 0.90 to 0.82. That gap is small
and, on 15 annual observations, not significant on its own — but it points the same direction
as the rest of this document. Getting μ from the covariance matrix and observable holdings
instead of from an 18-year sample mean is the same move as dropping μ entirely (min variance),
except it keeps the return forecast instead of throwing it away, and it does better than both
max-Sharpe and min-variance at every window.

**The views did essentially nothing.** The momentum view adds 0.02 of Sharpe at the AUM anchor,
0.02 at the global anchor, and *subtracts* 0.04 at the equal anchor. It also costs 21.8%
one-way turnover a year — the highest number in section 4's table, more than max-Sharpe's 16.2%
— because the trailing-momentum leader changes and the book chases it. Paying the most
turnover in the study to buy 0.02 of Sharpe that flips sign under a different anchor is not a
strategy; it is noise with a transaction cost.

The honest summary is that Black-Litterman's contribution here is a better *prior*, not a
better *forecast*. Which is roughly what Black and Litterman said it was.

Two implementation notes, since both are places to get it wrong:

- **Risk aversion δ comes from an assumption, not the data.** δ = 0.40 / σ_market, i.e. a
  long-run market Sharpe of 0.4 divided by the trailing estimate of market volatility.
  Computing δ from the sample mean instead would smuggle the noisy input back in through the
  one door the method exists to close.
- **The posterior covariance is Σ + M, not Σ.** M is the uncertainty in μ_BL itself. Including
  it is the honest choice and it costs the textbook identity: with no views, the unconstrained
  answer becomes w_market / (1+τ) rather than w_market, and under a fully-invested constraint
  that missing 5% gets re-invested along the minimum-variance direction. `tests/
  test_black_litterman.py` pins down both versions so the difference is deliberate rather than
  discovered later.

## 7. CVaR: does targeting tail risk directly change anything here?

Every method above optimizes variance, which treats a surprise gain and a surprise loss of the
same size identically. `src/cvar.py` adds one that doesn't: `min_cvar_weights()` minimizes
CVaR (Conditional Value-at-Risk / Expected Shortfall) directly, via the Rockafellar-Uryasev
linear program - working from the actual historical daily scenarios rather than reducing them
to (mu, Sigma) first, so it can in principle tell apart two assets with identical mean and
variance but different tail shape. `tests/test_cvar.py` proves that difference is real on
synthetic data: a Gaussian asset and a fat-tailed, occasional-crash asset built to share the
same mean and variance are NOT treated the same by `min_cvar_weights` (it tilts away from the
crash-prone one), even though a variance-only optimizer is mathematically unable to see any
difference between them at all.

So does it change anything on this project's actual 5-ETF universe? Almost not at all:

| | In-sample vol | VaR 95% (daily) | CVaR 95% (daily) | Weights |
|---|---|---|---|---|
| Min variance | 5.22% | 0.45% | 0.74% | SPY 7 / EFA 0 / AGG 91 / GLD 2 / VNQ 0 |
| Min CVaR (95%) | 5.22% | 0.45% | 0.74% | SPY 7 / EFA 0 / AGG 90 / GLD 3 / VNQ 0 |

Same volatility to two decimal places, same VaR, same CVaR, a one-point shuffle between AGG and
GLD. Out of sample the story repeats - walk-forward Sharpe 0.60 for Min CVaR against min-variance's
0.58 (essentially noise, 15 annual observations), same -13% worst year, and the same ranking
against equal weight (4th of 6, same corner-solution problem: minimizing tail risk with no
return target still just finds the calmest asset and piles into it). The lookback-sensitivity
table tells the same story at every window from 2 to 7 years - Min CVaR and min-variance move
together throughout.

**Why the synthetic test finds a real difference and this data doesn't.** The test's synthetic
pair is constructed so mean and variance are IDENTICAL and only the tail shape differs - that's
the one case a variance-based optimizer is mathematically blind to by construction. AGG (the
asset both methods pile into here) doesn't need that kind of help to look good: it has both the
lowest variance AND unremarkable tail behavior for its variance, in this sample. When the
lowest-variance asset isn't ALSO tail-risky, minimizing variance and minimizing CVaR point the
same optimizer at the same corner. CVaR optimization would earn its keep on a universe where
that isn't true — a portfolio that includes something like a short-volatility strategy or a
emerging-market currency carry trade, where the calm, low-variance days are calm precisely
because the risk shows up rarely and severely instead of continuously. Five liquid, long-only
ETFs mostly don't manufacture that pattern.

**What this means for item 5 below.** Implementing CVaR optimization was worth doing - the
Rockafellar-Uryasev LP is a real, useful piece of machinery, and the synthetic test shows it does
exactly what it claims when the assets in front of it actually have different tail shapes. What
this section shows is equally worth stating plainly: on the specific dataset the rest of this
project uses, it does not produce a materially different portfolio than the min-variance solution
this project already had. A tool doing nothing new on a specific input is a finding about the
input, not a bug in the tool - but it's not the input a reader would guess without seeing the
number.

---

## 9. Fifty assets: where a covariance matrix starts to matter

Every number above this line comes from five ETFs, and the caveats section says plainly that five
assets is an easy problem. This section is the hard one: 50 US large caps, 2006-2024, quarterly
rebalance, everything out of sample. `python factor_study.py`.

The question is not which portfolio earned more. Returns over any one sample are mostly luck. The
question is whether the risk number each estimator printed at the rebalance turned out to be true,
which is measurable every quarter and is the number a risk manager is actually paid to get right.

### The counting problem

| Window | Observations per parameter | Noise edge λ₊ | Eigenvalues above it | Their share of variance |
|---|---|---|---|---|
| 126d | 4.9 | 2.66 | 3 | 43% |
| 252d | 9.9 | 2.09 | 3 | 36% |
| 504d | 19.8 | 1.73 | 4 | 43% |
| 1008d | 39.5 | 1.50 | 5 | 52% |

A 50×50 covariance matrix has 1,275 free parameters. λ₊ = (1 + √(N/T))² is the Marchenko-Pastur
edge: the largest eigenvalue a correlation matrix of this shape produces when the underlying data
has *no structure at all*. At a 126-day window it sits at 2.66, so an eigenvalue of 2 there means
nothing; at 1008 days the edge is 1.50 and the same eigenvalue is a real factor. Identical number,
opposite verdict, purely from how much data went into the matrix.

Three to five directions clear the band. The other 45-47 are noise, and a min-variance optimizer
is a machine for finding the smallest eigenvalue and betting on it.

### What each estimate promised, and what it delivered

Unconstrained minimum variance, quarterly, four estimation windows. `predicted` is the volatility
the estimate claimed at the rebalance; `realized` is what the portfolio then did.

The first row is the control and it is the reason the rest of the table can be read at all. It
holds 1/N every quarter and optimizes nothing, so its ratio is not estimation error — it is
volatility clustering, the fact that the quarter after a calm window is often not calm. Every
other row has to clear **1.12** before any of its miss can be blamed on the matrix.

| 252-day window | predicted | realized | ratio | turnover | gross short |
|---|---|---|---|---|---|
| Equal weight (control) | 16.79% | 18.87% | **1.12** | 0% | 0% |
| Sample | 9.22% | 15.48% | 1.68 | 94% | 104% |
| Shrunk (Ledoit-Wolf) | 9.40% | 15.02% | **1.60** | 76% | 83% |
| Factor (k from MP) | 8.27% | 16.25% | 1.97 | 54% | 68% |
| Diagonal only | 3.29% | 16.67% | 5.07 | 5% | 0% |

| Sample covariance, by window | predicted | realized | ratio | control ratio |
|---|---|---|---|---|
| 126d | 7.41% | 17.08% | 2.30 | 1.16 |
| 252d | 9.22% | 15.48% | 1.68 | 1.12 |
| 504d | 10.61% | 15.17% | 1.43 | 1.07 |
| 1008d | 11.50% | 13.74% | 1.20 | 0.89 |

**Two separate things are true and they point in opposite directions.** Minimum variance works:
15.5% realized against 18.9% for holding all fifty equally, a fifth less risk, exactly as
advertised. And the risk *number* is a fantasy: on a 126-day window the optimizer promised 7.4%
and delivered 17.1%. The portfolio was fine. The report was off by a factor of 2.3, and shrinking
the window — the obvious way to be "more responsive to current conditions" — is what makes it
worse.

The diagonal row is the reductio. Assume every correlation is zero and you predict 3.3% risk for a
portfolio that runs at 16.7%: a 5× understatement from an assumption that sounds merely
conservative.

### How many factors? The bias-variance tradeoff, measured

Same test, only k changes. k = 50 reproduces the sample covariance exactly, so this is a continuum
rather than a list of alternatives.

| k (1008-day window) | predicted | realized | ratio | turnover |
|---|---|---|---|---|
| 1 | 7.43% | 16.00% | 2.15 | 9% |
| 2 | 9.28% | 16.89% | 1.82 | 17% |
| 3 | 10.33% | 15.40% | 1.49 | 22% |
| 5 | 10.81% | 15.31% | 1.42 | 23% |
| 10 | 11.12% | **14.78%** | 1.33 | 29% |
| 20 | 10.96% | 14.98% | 1.37 | 40% |
| 50 (= sample) | 11.50% | **13.74%** | **1.20** | 25% |

Realized risk falls as k rises to 10, worsens at 20, and is lowest at 50 — the sample covariance
this section was supposed to improve on. **The factor model loses here, and the reason is in the
first table.** Marchenko-Pastur says keep 5 factors at this window; realized risk is minimized
around 10 and keeps improving to 50. The cutoff answers "which eigenvalues are distinguishable
from noise," and that is not the same question as "which eigenvalues are worth keeping." A
direction can be statistically indistinguishable from noise and still be a better estimate of the
truth than zero.

The other half of the answer: a factor model asserts that residual correlations are exactly zero.
Among 50 mega-caps, residuals within a sector are not zero — two money-center banks co-move for
reasons no small set of principal components captures — and setting those to zero tells the
optimizer that a pairs-like position in JPM and BAC is better diversified than it is. With 4,700
days of history on 50 liquid names, that lie costs more than the noise it removes. Reverse the
ratio — 500 names on a one-year window — and the arithmetic flips, which is why the industry's
risk models are factor models. This universe simply is not starved enough.

What the factor model does buy is honesty about leverage: at k = 5 it runs 86% gross short against
the sample's 80% for a fifth less turnover, and at short windows it roughly halves both. That is a
real result and it is not the one it was built for.

### The constraint beats all of it (Jagannathan-Ma)

Same test, same windows, shorting forbidden.

| 252-day window, long-only | predicted | realized | ratio | turnover |
|---|---|---|---|---|
| Equal weight (control) | 16.79% | 18.87% | 1.12 | 0% |
| Sample | 11.49% | 14.94% | 1.30 | 27% |
| Shrunk (Ledoit-Wolf) | 11.30% | 14.85% | 1.31 | 26% |
| Factor (k from MP) | 10.94% | 14.75% | 1.35 | 24% |

Forbidding shorts takes the sample covariance from 1.68 to 1.30 and its turnover from 94% to 27%,
and it lands realized risk *below* anything the unconstrained version achieved with any estimator.
Then all four estimators collapse into a 0.2 percentage point range: **once shorting is off the
table, the choice of covariance matrix stops mattering.**

This is Jagannathan and Ma (2003) and it is not a coincidence. A no-short constraint is
algebraically equivalent to subtracting the constraint's Lagrange multipliers from the covariance
matrix, and those multipliers bind precisely on the assets whose estimated covariances are most
extreme. The constraint *is* a shrinkage estimator. It is free, it needs no parameters, and on
this data it does more than any of the three estimators that were designed for the job.

The ordering to remember, worst first: no constraint and a bad matrix (2.30), no constraint and a
good matrix (1.60), a constraint and any matrix (1.30), no optimization at all (1.12).

### The control

The same code on this project's five ETFs, 252-day window: sample 1.29, shrunk 1.25, factor 1.33,
diagonal 1.41 — everything inside the sampling noise, with the control at 1.14. Fifteen
parameters from 252 observations leaves nothing for a better estimator to fix, which is the
caveat this project has been carrying since the first commit, now with a number attached.

---

## Honest caveats

I'd rather state these than have them found.

**The sample favors equal weight.** 2010–2024 was a strong equity decade with weak bond returns.
Equal weight carries the most equity risk of any method here, so it was structurally positioned
to win this particular sample. The finding "estimation error degrades optimization out of
sample" is robust and well documented; the specific finding "equal weight beats everything" is
partly this sample.

**Sharpe here is computed on annual returns, not daily** — 15 observations, so the standard error
on any of these Sharpe ratios is large. The gap between 0.82 and 0.75 is not statistically
significant on its own. What carries weight is the direction and its consistency across four
independent window choices, not any single comparison.

**Risk-free rate is 0 throughout.** Over 2010–2024 that flatters every portfolio equally, so it
doesn't affect the ranking, but the Sharpe levels are overstated.

**Turnover is measured, not charged.** Adding realistic costs would widen the gap slightly in
equal weight's favor and change nothing material.

**Five assets is an easy problem.** The estimation pathology gets dramatically worse with more
assets, which is where shrinkage, factor models and Black-Litterman earn their keep. Section 9
runs the same machinery on 50 stocks for exactly this reason, and the gap is not subtle: the
sample covariance understates realized risk by 2.3x there against 1.3x here. Everything in
sections 1-8 should be read as the well-behaved end of the problem.

**The Black-Litterman anchor uses today's fund sizes.** Stated in section 8 as well, because it
is the one genuinely forward-looking input anywhere in this project and it should not be
possible to read the walk-forward table without meeting it.

## What I'd build next

1. ~~Ledoit-Wolf optimal α rather than a hand-picked 0.3~~ — done, see
   `ledoit_wolf_alpha()` in `src/risk_parity.py`. On the full 2007-2024 daily
   history for this 5-asset universe, the analytic optimum comes out to
   **alpha ~ 0.005** — essentially no shrinkage — against the 0.3 used
   everywhere else in this writeup. That's not a contradiction, it's the
   formula doing its job: alpha* trades off sampling noise against target
   bias, and with ~4,500 daily observations for only 5 assets, the sample
   covariance barely has any noise left to correct for. The 0.3 used above
   is closer to what the formula would recommend on a much shorter window
   (a few hundred observations, or many more assets) — worth keeping in mind
   before copying 0.3 into a problem with a different N and T.
2. ~~Black-Litterman~~ — done, `src/black_litterman.py`, results in section 8. The short
   version: it beats everything in the walk-forward table, and roughly two thirds of that
   margin traces to an equilibrium anchor built from today's fund sizes rather than to the
   method. The momentum view attached to it buys 0.02 of Sharpe for the highest turnover in the
   study. Worth having; not worth believing at face value.
3. ~~Factor-model covariance~~ — done, `src/factor_model.py`, results in section 9. It works
   as designed — positive definite by construction, a third of the turnover, half the shorting —
   and it still does not beat the sample covariance on realized risk for this universe, because
   50 liquid mega-caps with 18 years of history are not parameter-starved enough for the
   tradeoff to pay. The finding that outranks it: forbidding short sales does more for realized
   risk than any of the three covariance estimators, and once you do, which matrix you used
   stops mattering.
4. ~~Turnover penalty in the objective~~ — done, see section 5. A quadratic penalty helps up to
   a point (turnover -63% for flat Sharpe) and then trades real return for lower turnover past
   that; a proper linear/transaction-cost penalty would need slack variables but is the more
   correct version of the same idea.
5. ~~CVaR optimization~~ — done, see section 7 and `src/cvar.py`. Real and provably different
   from variance-based optimization in general (the synthetic test in `tests/test_cvar.py`
   shows it clearly), but converges to essentially the same portfolio as min-variance on this
   project's actual 5-ETF universe — a finding about this dataset's tail shapes, not evidence
   the method doesn't work.
