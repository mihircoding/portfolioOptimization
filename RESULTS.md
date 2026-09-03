# Results

All 24 tests pass (`python -m pytest -q`). Numbers below are `python run_optimization.py`.

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
| **Equal weight** | **7.80%** | 9.46% | **0.82** | −14.21% | **0.0%** |
| Inverse vol | 5.79% | 7.32% | 0.79 | −12.67% | 3.5% |
| Equal risk contribution | 5.42% | 7.08% | 0.77 | −12.09% | 4.2% |
| Max Sharpe | 6.65% | 8.89% | 0.75 | −11.70% | **16.2%** |
| Min variance | 3.32% | 5.74% | 0.58 | −13.04% | 2.3% |

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
assets, which is where shrinkage, factor models, and Black-Litterman earn their keep — none of
which are implemented here.

## What I'd build next

1. **Ledoit-Wolf optimal α** rather than a hand-picked 0.3, and compare.
2. **Black-Litterman** — start from market-implied equilibrium returns and tilt with explicit
   views and confidences, instead of feeding raw historical means into an optimizer.
3. **Factor-model covariance** — estimate `Σ = BΩBᵀ + D` from a handful of factors instead of
   `N(N+1)/2` free parameters.
4. ~~Turnover penalty in the objective~~ — done, see section 5. A quadratic penalty helps up to
   a point (turnover -63% for flat Sharpe) and then trades real return for lower turnover past
   that; a proper linear/transaction-cost penalty would need slack variables but is the more
   correct version of the same idea.
5. **CVaR optimization** — because variance penalizes upside and downside identically, and
   nobody actually minds the upside.
