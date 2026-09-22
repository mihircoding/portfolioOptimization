# Interview notes — portfolio optimization

The material here is the most standardized in quant finance, which means reciting Markowitz gets
you nothing. What distinguishes an answer is knowing that the textbook method **fails out of
sample**, why it fails, and what practitioners do instead. This project has the numbers for all
three.

---

## The 60-second version

> I implemented mean-variance optimization — min-variance, max-Sharpe, the efficient frontier —
> plus covariance shrinkage and equal-risk-contribution weighting. Then I walk-forward tested
> them: estimate on a 3-year trailing window, hold a year, roll, 15 rebalances over 2010–2024.
> In sample max-Sharpe wins with 0.90, which is trivially true since it's the in-sample argmax.
> Out of sample it finished fourth of five at 0.75, behind equal weighting at 0.82 — which needs
> no estimates, no optimizer, and no trading, while max-Sharpe turned over 16% of the book a
> year. That ranking held at 2, 3, 5 and 7-year estimation windows. It's DeMiguel-Garlappi-Uppal
> reproduced on a different universe.

---

## The core theory

### The problem

```
expected return  =  wᵀμ
variance         =  wᵀΣw
```

**Be able to say why the second line is the entire subject.** Portfolio volatility is not the
weighted average of individual volatilities — it's lower whenever assets aren't perfectly
correlated. The difference *is* diversification. An asset isn't risky or safe on its own, only
relative to what it's held with; a volatile asset that's negatively correlated with your book
reduces portfolio risk.

### The efficient frontier

For each target return, the minimum-variance portfolio achieving it. The set of portfolios you
can't improve on. Below the curve is dominated; the upper branch is the only part worth holding.

With a risk-free asset, the **capital market line** runs from `rf` tangent to the frontier, and
everyone holds the same tangency portfolio levered up or down to taste — that's **two-fund
separation**, and it's the step from Markowitz to CAPM.

### Why min-variance is special

`min_variance_weights(cov)` doesn't take `mu`. Say this out loud in an interview, because it's
the whole point: **that portfolio needs no return forecasts.** And expected returns are the
input you estimate worst, so an optimizer that never receives them can't be wrecked by them.

---

## The failure, and how to explain it

### The instability

**Q: What's wrong with mean-variance optimization?**

It treats μ and Σ as known constants. They're noisy estimates, and the optimizer's response to
noise is pathological: it loads into whatever asset had the best estimation luck, because from
inside the objective a spuriously high mean is indistinguishable from a real edge. Michaud calls
it the **error-maximizing property**.

**The asymmetry is the key sentence:** you need decades of data to estimate a mean return to
useful precision; a covariance converges in months. The input the optimizer is most sensitive to
is the one you know least about. Give it a slightly different sample and the weights move
enormously.

**Then give the number, because that's what makes it your answer rather than a recital:**
in-sample Sharpe 0.90 for max-Sharpe against 0.55 for equal weight; out of sample it's 0.75
against 0.82. The ordering completely inverts.

**Q: So the optimizer is useless?**

No — it's correct given its inputs. The failure is in the inputs, not the mathematics. This
matters as a framing because the fixes all target the inputs: shrink the covariance, replace
historical means with equilibrium-plus-views, or stop using expected returns at all.

### The fixes, in order of how often they're used

**1. Covariance shrinkage.** `Σ_shrunk = (1−α)Σ_sample + α·target`, target usually the diagonal
or a constant-correlation matrix.

Why it works: bias-variance. An N-asset sample covariance estimates N(N+1)/2 parameters — 1,275
for 50 assets — from data that rarely supports it. The **extreme eigenvalues are the worst
estimated**, and those are exactly the directions an optimizer loads into. Shrinkage adds bias
and removes much more variance. Ledoit-Wolf derive the optimal α analytically.

Measured here: α = 0.3 cuts the condition number from 54.9 to 44.7.

**2. Black-Litterman.** Don't estimate μ from history at all. Start from the returns *implied*
by market-cap weights being optimal (reverse-optimize CAPM equilibrium), then tilt toward
explicit views with explicit confidences via Bayesian updating. Produces stable, sensible
weights that default to the market portfolio when you have no views — which is the right default.

**3. Constraints as implicit regularization.** Long-only bounds and position caps are usually
justified as mandate requirements, but Jagannathan & Ma (2003) showed they're mathematically
equivalent to a form of shrinkage. They improve out-of-sample performance *even when the true
optimal portfolio would short.* That's a strong point to have loaded.

**4. Resampled efficiency (Michaud).** Bootstrap the inputs, optimize each resample, average the
weights. Expensive, no clean theory, empirically decent.

**5. Turnover penalty.** Add a cost term to the objective. This is what actually makes optimized
portfolios usable in production.

**6. Drop expected returns entirely.** Min-variance, inverse-vol, ERC.

---

## Risk parity and risk contributions

### The decomposition

```
rc_i = w_i · (Σw)_i / (wᵀΣw)
```

**Exact, not approximate** — portfolio variance is homogeneous of degree 2 in w, so Euler's
theorem makes the parts sum to the whole. Being able to say *why* it's exact is a differentiator.

**Why it matters, with the number from this project:** equal weight puts 20% of capital in every
one of five ETFs and gets **1% of its risk from bonds and 37% from REITs.** Weights look
balanced; risk isn't. That gap is what ERC exists to close, and "asset X is 4% of the book and
38% of the risk" is a sentence risk managers say constantly.

### ERC

Solve for weights where every asset contributes equally. No closed form except when all
correlations are equal — where it reduces to inverse-vol, which is why inverse-vol is the right
starting point for the solver. The objective isn't convex, so the starting point matters.

**Q: Why is risk parity popular?**

- Needs no expected returns — the robustness argument again.
- Diversifies risk rather than capital, so a bond sleeve actually does something.
- Empirically stable weights, hence low turnover: 4.2% here against max-Sharpe's 16.2%.

**Q: What's the criticism?**

- It requires **leverage** to reach an equity-like return, because it's structurally bond-heavy.
  Leverage introduces financing cost, margin calls, and forced deleveraging in exactly the
  correlated selloffs you own bonds to survive.
- It implicitly assumes all assets have the same Sharpe ratio, which is an unstated return
  forecast — one that's less visible than the one mean-variance makes explicit.
- The famous version of the strategy had a very good run during a 40-year bond bull market.
  2022 was informative.

---

## Questions I'd expect

**"Walk me through mean-variance optimization."**
Estimate μ and Σ from historical returns, annualized. Then it's a constrained optimization:
minimize `wᵀΣw` subject to weights summing to 1, optionally with a target-return constraint or
long-only bounds. SLSQP handles it — smooth objective, nonlinear constraints, low dimension.
Sweeping the target return traces the efficient frontier. Maximizing `(wᵀμ − rf)/√(wᵀΣw)` gives
the tangency portfolio.

**"You have 500 stocks and 2 years of daily data. What goes wrong?"**
The sample covariance is **singular**. 500 assets need 125,250 parameters and you have roughly
500 observations, so the matrix has rank at most ~500 in a 500-dimensional space — the smallest
eigenvalues are zero or near-zero, and the optimizer treats those directions as free risk
reduction and levers into pure estimation noise. Fixes: a factor model (`Σ = BΩBᵀ + D`, a few
hundred parameters instead of 125,250), shrinkage, or PCA truncation. This is the question where
knowing the parameter count off the top of your head pays off.

**"Why is annualized covariance 252× daily but volatility only √252×?"**
Variance of a sum of i.i.d. terms adds, so variance scales linearly in time. Volatility is its
square root. It's the most common unit error in the field and worth stating cleanly.

**"Your optimizer returned all equal weights. What happened?"**
Almost certainly `result.success == False` and scipy handed back `x0`, which is the equal-weight
starting point. Always check the flag. Common causes: infeasible constraints (a target return
outside the achievable range under long-only), a singular covariance matrix, or a tolerance too
tight for the scaling. My code raises rather than returning silently.

**"Is the frontier convex?"**
Yes — the feasible set in (vol, return) space is convex, because a combination of two portfolios
has volatility at most the weighted average of theirs (equality only at correlation 1). That's
why every point between two attainable portfolios is attainable, and it's the same fact as
diversification.

**"How would you know if your optimization is any good?"**
Not by its in-sample Sharpe, which is guaranteed to look good and means nothing. Walk it
forward: estimate on a trailing window, hold, roll, and measure realized performance. Then check
the result survives a change of estimation window — I ran 2, 3, 5 and 7 years, and the ranking
was stable, which is what turned one result into a finding.

**"Variance as a risk measure — problems?"**
It's symmetric: upside deviation is penalized identically to downside, and nobody actually minds
the upside. It also assumes returns are adequately described by two moments, which fails for
anything with skew or fat tails — options, credit, anything with a payoff floor. Alternatives:
semi-variance, VaR (not sub-additive, so it can penalize diversification, which is a genuine
theoretical defect), and **CVaR/expected shortfall** — coherent, and linearly programmable,
which is why it's the practical choice.

**"Sharpe ratio — limitations?"**
Assumes normality, so it flatters strategies with negative skew and fat tails — short-vol
strategies look wonderful right up until they don't. It's also easy to inflate by lengthening the
measurement interval (annualizing monthly returns by √12 assumes no autocorrelation, and
illiquid assets are heavily autocorrelated). Alternatives: Sortino (downside deviation only),
Calmar (return over max drawdown), and the probabilistic Sharpe ratio, which puts a confidence
interval on it — which most reported Sharpe ratios badly need.

---

## Things to admit before they ask

- Historical means as the return forecast is the weakest possible choice. The walk-forward test
  is partly a demonstration of exactly that.
- Sample estimators only — no factor model, no Ledoit-Wolf optimal α, no Black-Litterman.
- Sharpe computed on 15 annual observations, so the standard errors are large and the gap
  between 0.82 and 0.75 isn't significant on its own. What carries weight is the *consistency*
  across four independent window choices, not any single comparison.
- 2010–2024 was a strong equity decade, and equal weight carries the most equity risk here, so
  the sample favored it. The general finding (estimation error degrades optimization out of
  sample) is robust and well documented; "equal weight beats everything" is partly this sample.
- Five liquid ETFs is an easy estimation problem. The pathology gets much worse with 50+ assets,
  which is where the techniques I *didn't* implement start mattering.
- Costs are a flat rate per dollar traded. They're charged now (RESULTS.md section 10), and on
  the five ETFs they change nothing: max-Sharpe's drag is 3 bps a year at 10 bps a side, so
  turnover there is a *symptom* of estimation error, not the cause of the underperformance. On
  50 stocks it's different — unconstrained min-variance loses 0.82% a year to costs — and a
  5-point no-trade band gets most of that back. No impact model, no borrow fee, no tax.
