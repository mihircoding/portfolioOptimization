# Code Walkthrough & Quant Interview Prep

This doc goes through the project file by file, explains the important lines
and the math behind them, and ends each section with the interview questions
that code prepares you to answer. If you can explain everything here without
looking, you can defend this project in an interview.

---

## 1. The big picture (say this first in an interview)

> "I implemented Black-Scholes from scratch — pricing, all five Greeks, and
> an implied-vol solver — then validated it three independent ways: against
> textbook values and put-call parity, against numerical derivatives for
> every Greek, and against a Monte Carlo simulator that shares only the
> model's assumption, not its math. Then I compared it to live S&P 500
> option chains and used the disagreement to demonstrate the volatility
> smile — i.e., I can show exactly where the model breaks and why."

The three ideas that everything below hangs on:

1. **No-arbitrage / risk-neutral pricing.** An option's price is the
   expected value of its payoff *under the risk-neutral measure*,
   discounted at the risk-free rate. Not the real-world expectation — the
   one where every asset drifts at `r`. This is THE core concept quant
   interviews test.
2. **Replication.** Black-Scholes works because you can continuously trade
   the stock and a bond to replicate the option's payoff. The option must
   cost what the replicating portfolio costs, or there's free money.
3. **Volatility is the only unknown.** Spot, strike, time, and rate are
   observable. The entire market for options is really a market for sigma —
   which is why implied vol, not price, is how traders quote options.

---

## 2. `black_scholes.py` — the model

### d1 and d2 (`d1_d2`)

```python
d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
d2 = d1 - sigma * np.sqrt(T)
```

This is the line to understand most deeply.

- Under the risk-neutral measure the stock follows geometric Brownian
  motion, so `ln(S_T)` is normally distributed with mean
  `ln(S) + (r - q - sigma^2/2)T` and standard deviation `sigma*sqrt(T)`.
- **d2** is the number of standard deviations by which the expected
  log-price exceeds the log-strike: `N(d2)` = risk-neutral probability the
  option finishes in the money, i.e. `P(S_T > K)`.
- **d1** is d2 shifted up by one full standard deviation (`sigma*sqrt(T)`).
  That shift comes from a change of measure (using the stock itself as
  numeraire): `N(d1)` is the probability of finishing in the money
  *weighted by how large S_T is when it happens*.
- The `-sigma^2/2` in the drift (hidden inside d2's derivation) is the
  **Itô correction**: `E[ln(S_T/S)] = (r - q - sigma^2/2)T` even though
  `E[S_T/S] = e^{(r-q)T}`, because the log of an average is more than the
  average of a log (Jensen's inequality). Interviewers love this.

The clipping lines:

```python
T = np.maximum(T, 1e-10)
sigma = np.maximum(sigma, 1e-10)
```

are not math — they prevent 0/0 at expiry or at zero vol. The limit they
approximate is correct: as T→0 the price converges to intrinsic value
(the test suite verifies this via `strategy_value at T=0 equals expiry
payoff`).

### The price formulas (`call_price`, `put_price`)

```python
return S * np.exp(-q * T) * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
```

Read it as two expectations:
- `K·e^{-rT}·N(d2)` — you pay K, but only in the worlds where you exercise
  (probability N(d2)), discounted to today.
- `S·e^{-qT}·N(d1)` — you receive the stock in those same worlds, but the
  "probability" must be payoff-weighted, hence N(d1) not N(d2).

The `e^{-qT}` on the spot is **Merton's dividend extension**: holding the
option instead of the stock means forfeiting the dividend stream, so the
effective spot you're optioning is the spot minus its future dividends.
Set q=0 and it's vanilla Black-Scholes.

**Put-call parity** (the first thing to reach for in any options question):

```
C - P = S·e^{-qT} - K·e^{-rT}
```

Proof: long call + short put = forward contract at strike K (you WILL buy
at K either way). No model needed — pure no-arbitrage. That's why the test
checks it to 1e-9: it must hold to machine precision regardless of any
model parameter. If an interviewer gives you a call price, you can always
get the put price without re-running any model.

### The Greeks — what each line says

**Delta** — `e^{-qT}·N(d1)` for calls, `e^{-qT}·(N(d1)-1)` for puts.
- It is NOT the probability of finishing ITM (that's N(d2)) — a classic
  interview trap. It's the replicating hedge ratio: how many shares you
  hold to be locally flat.
- Call delta lives in (0,1), put delta in (-1,0); call delta minus put
  delta = `e^{-qT}` (differentiate put-call parity — everything about
  Greeks pairs falls out of parity).

**Gamma** — `e^{-qT}·φ(d1) / (S·sigma·sqrt(T))`.
- Same for calls and puts (parity again: the S-term of parity is linear,
  so its second derivative is zero).
- φ is the normal *density*, peaked at d1=0 — gamma is largest at the
  money and explodes as T→0 (the sqrt(T) in the denominator): a pin-risk
  ATM option near expiry flips between delta 0 and 1 on tiny moves.
- Long options = long gamma = you profit from being wrong fast. You pay
  for it via theta.

**Vega** — `S·e^{-qT}·φ(d1)·sqrt(T) / 100`.
- Same for calls and puts. The `/100` is a *quoting convention* (per vol
  point, not per unit) — know the difference between a formula and a
  convention; interviewers probe sloppy units.
- Grows with sqrt(T): long-dated options are volatility instruments,
  short-dated ones are direction instruments.

**Theta** — the money line is the decay term:

```python
common = -(S * np.exp(-q * T) * norm.pdf(d1) * sigma) / (2.0 * np.sqrt(T))
```

- Note it contains `φ(d1)·sigma / sqrt(T)` — the same ingredients as gamma.
  That's not a coincidence: for a delta-hedged position,
  `theta ≈ -1/2 · gamma · S² · sigma²`. Theta is the rent you pay for
  gamma. Being able to state that relationship is a strong interview
  signal.
- The `r`/`q` terms are the financing legs; the `/365` is again quoting
  convention (per calendar day).

**Rho** — `±K·T·e^{-rT}·N(±d2)/100`. Least loved Greek; know that q does
not add a term to it (only shifts d2), because r enters the formula only
through the discount factor on K.

### Implied vol (`implied_vol`) — bisection

```python
if f_lo * f_hi > 0:            # no sign change -> no root in range
    return float("nan")
```

- Works because **price is strictly monotonic in sigma** (vega > 0). One
  root, guaranteed to be found if bracketed. Bisection gains one bit of
  accuracy per iteration — 100 iterations is absurd overkill, which is the
  point: robustness over speed for a display tool.
- The NaN return is a *feature*: a market price below intrinsic value has
  no implied vol. On the market page those show as `-`, and they flag
  stale quotes.
- Interview follow-up you should expect: "why not Newton-Raphson?" Answer:
  Newton (sigma ← sigma - (price-target)/vega) converges quadratically and
  is what production systems use, but it can diverge where vega ≈ 0 (deep
  ITM/OTM); bisection never diverges. Ideal answer: Newton with a
  bisection fallback, or Brent's method.

---

## 3. `monte_carlo.py` — the model without the calculus

### The one-line simulation (`terminal_prices`)

```python
return S * np.exp((r - q - 0.5 * sigma**2) * T + sigma * np.sqrt(T) * z)
```

This is the *exact solution* of the GBM SDE `dS = (r-q)S dt + sigma S dW`,
not an Euler approximation — for terminal-value payoffs you can jump
straight to T in one step with zero discretization bias. Note the same
Itô `-sigma²/2` correction as in d1/d2. Drift is `r - q`, NOT the stock's
real expected return **mu** — that's risk-neutral pricing made concrete,
and "why doesn't mu appear?" is a top-5 interview question. (Answer:
because the hedge portfolio eliminates the stock's drift; preferences
about real-world growth are already in today's spot price.)

### Price and error (`mc_price`)

```python
discounted = np.exp(-r * T) * payoffs
return float(discounted.mean()), float(discounted.std(ddof=1) / np.sqrt(n_paths))
```

- Price = discounted average payoff. That IS the risk-neutral pricing
  formula, executed literally.
- The standard error line is the statistics half of the interview:
  the estimator's SE is `std/sqrt(n)` (ddof=1 = sample std, Bessel's
  correction). Convergence is O(1/sqrt(n)): **100× more paths buys one
  extra decimal digit.** The test suite asserts |MC - BS| < 4·SE — a
  statistically principled tolerance rather than an arbitrary epsilon
  (know why: under the CLT the standardized error is ~N(0,1), so 4 SEs is
  a ~1/16,000 false-failure rate).
- Natural follow-ups: variance reduction. Antithetic variates (reuse -Z),
  control variates (use the stock itself: you know E[S_T] exactly),
  importance sampling for deep OTM. This project deliberately uses none —
  being able to *name* them and say why they weren't needed is enough.

### Paths for the chart (`sample_paths`)

```python
increments = (r - q - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * z
log_paths = np.cumsum(increments, axis=1)
```

Same exact-GBM step applied per time slice, cumulative-summed in log space
(sums of normals stay normal — this is why we simulate logs, and why the
resulting price distribution is lognormal: never negative, right-skewed).

---

## 4. `strategies.py` — payoffs and structure

### The design decision worth defending

Every strategy is a list of legs `{kind, strike, qty}` and three generic
functions do everything. In an interview this is your software-design
answer: *"I didn't write ten payoff functions; I wrote one representation
and three operators over it — cost, payoff at expiry, and mark-to-model
value. Adding an eleventh strategy is four lines of data, not code."*

### The lines that matter

Cost (signed sum of Black-Scholes leg values):

```python
total += leg["qty"] * float(bs.price(leg["kind"], S, leg["strike"], T, r, sigma))
```

Sign convention: long = +qty = pay premium; short = -qty = collect it.
A negative total means the structure is opened **for a credit** (iron
condor, iron butterfly). The test asserts the condor is a credit — if the
signs were flipped anywhere, that test dies.

Payoff at expiry — options collapse to intrinsic value:

```python
payoff += leg["qty"] * np.maximum(spot_grid - leg["strike"], 0.0)   # call
payoff += leg["qty"] * np.maximum(leg["strike"] - spot_grid, 0.0)   # put
```

PnL is always `payoff - cost`. One line, every chart on the page.

Mark-to-model before expiry (`strategy_value`) re-prices every leg with
Black-Scholes at the remaining time. This is what makes the strategy
heatmap volatility-sensitive: **an iron condor that expires safely can
still be deeply underwater mid-life if vol spikes** — because you're short
the two inner options and short options are short vega. Being able to
explain that from the code is exactly the "do you understand what you
built" interview moment.

### Know the shape of each strategy (they will ask)

| Strategy | Legs | View expressed |
|---|---|---|
| Covered call | +stock, -call(hi) | mildly bullish, income |
| Married put | +stock, +put(lo) | bullish with insurance |
| Bull call spread | +call(K), -call(hi) | bullish, capped both ways |
| Bear put spread | +put(K), -put(lo) | bearish, capped both ways |
| Collar | +stock, +put(lo), -call(hi) | own it, hedged both ways |
| Long straddle | +call(K), +put(K) | long vol, direction-agnostic |
| Long strangle | +call(hi), +put(lo) | long vol, cheaper, needs bigger move |
| Call butterfly | +call(lo), -2call(K), +call(hi) | short vol, pin the middle |
| Iron condor | +put(lo2), -put(lo), -call(hi), +call(hi2) | short vol, wide range |
| Iron butterfly | +put(lo), -put(K), -call(K), +call(hi) | short vol, tight range, bigger credit |

Unifying idea: **long options = long volatility, short options = short
volatility**; every multi-leg structure is a shaped bet on where the stock
ends up AND how much it moves on the way.

---

## 5. `market_data.py` — the empirical side

### Historical volatility (`historical_volatility`)

```python
log_returns = np.log(closes / closes.shift(1)).dropna()
return float(log_returns.std() * np.sqrt(252))
```

- Log returns, not percent returns: they're the quantity GBM says is
  normal, and they add across days.
- `sqrt(252)`: variance is additive over independent periods, so vol
  scales with the square root of time. 252 = trading days. Using calendar
  365 here but 252 for vol scaling is standard market practice, not an
  inconsistency (variance accrues on trading days; discounting accrues on
  calendar days).
- Weaknesses you should volunteer before being asked: it's backward-looking,
  equal-weights old and new days, and assumes vol is constant. Name-drop
  the upgrades: EWMA, GARCH for estimation; realized vol from intraday
  data.

### Dividend yield (`dividend_yield`)

```python
return float(hist["Dividends"].sum() / spot)
```

Trailing 12 months of actual cash dividends over today's price, used as
the continuous q. It's a proxy (real dividends are discrete lumps, and
American calls can be optimally exercised just before ex-div dates), but
it removes the *systematic* bias flat-BS has on dividend payers.

### Time to expiry (`years_to_expiry`)

Calendar days / 365 with a half-day nudge so expiry day itself never reads
as zero. Small detail, but "how do you handle T on expiration day"
distinguishes people who've touched real chains from people who haven't.

---

## 6. The market page — where the model meets reality

The comparison line that everything else feeds:

```python
df["diff"] = df["lastPrice"] - df["model"]
```

Green = market above model = market expects MORE movement than the past
year showed; red = less. The point of this page for interviews is that you
can articulate **why they disagree** — this is the difference between
"I implemented a formula" and "I understand a model":

1. **The volatility smile/skew** (the chart at the bottom of the page).
   One flat sigma vs. the market's per-strike IV. Equity IV skews high on
   the downside because crash protection is bid (post-1987, fat left
   tails). Black-Scholes' lognormal has thinner tails than reality, so it
   underprices wings. The smile page backs IVs out with *this project's
   own solver*, so nothing is taken on faith from Yahoo.
2. **Historical vs. implied vol.** The model is fed backward-looking vol;
   the market prices forward-looking vol. Around earnings, IV detaches
   from history completely — that gap is *information*, not error.
3. **American vs. European exercise.** Listed US single-stock options are
   American; Black-Scholes prices European. The early-exercise premium
   matters mostly for deep-ITM puts (interest on K) and calls right before
   ex-dividend dates. Fix: binomial trees (name Cox-Ross-Rubinstein).
4. **Stale quotes.** `lastPrice` on an illiquid strike can be hours old;
   the NaN implied vols are exactly those rows. Always check volume /
   bid-ask before calling anything "mispriced."

---

## 7. `test_sanity.py` — the validation story

Interviewers care as much about "how do you know it's right" as the code.
Your answer has four layers, all in this file:

1. **Known values**: S=K=100, T=1, r=5%, sigma=20% → call 10.4506,
   put 5.5735 (standard textbook numbers).
2. **Structural identities**: put-call parity to 1e-9, with AND without
   dividends. These hold regardless of parameters — the strongest kind
   of test.
3. **Greeks vs. numerical derivatives**: every closed-form Greek is
   checked against central-difference bump-and-reprice
   `(f(x+h) - f(x-h)) / 2h`. Central difference has O(h²) error vs O(h)
   for one-sided — a numerical-methods talking point in itself. Note the
   theta check bumps T *down* for the forward direction
   (`C(T-eps) - C(T+eps)`) because theta is decay as calendar time
   *passes*, i.e. as time-to-expiry shrinks. Sign conventions are where
   Greek bugs live.
4. **Cross-method agreement**: Monte Carlo within 4 standard errors of the
   closed form, with and without dividends. Two derivations, one
   assumption — if they agree, the algebra is almost certainly right.

Plus limit checks (deep ITM → forward value, deep OTM → 0, T→0 → intrinsic)
and strategy invariants (straddle worth exactly 0 at the strike, condor is
a credit). CI runs all of it on every push.

---

## 8. Rapid-fire interview Q&A (answers grounded in this repo)

**Q: Derive put-call parity.**
Long call, short put, both strike K: at expiry you buy at K no matter
what → it's a forward. Today's cost of a forward is `S·e^{-qT} - K·e^{-rT}`.
Model-free. Tested to 1e-9 in `test_sanity.py`.

**Q: Why doesn't the stock's expected return appear in the price?**
Delta-hedging removes the directional exposure; what's left earns the
risk-free rate by no-arbitrage. Equivalently, we price under the
risk-neutral measure where every drift is r. Concretely: `mu` appears
nowhere in `black_scholes.py` or `monte_carlo.py`, drift is `r - q`.

**Q: Is delta the probability the option expires ITM?**
No — that's N(d2). Delta is e^{-qT}·N(d1), the hedge ratio. They differ
by exactly one sigma·sqrt(T) shift in the argument.

**Q: Why do both calls AND puts get more expensive with volatility?**
Payoffs are convex: unlimited upside, floor at zero. More dispersion
raises the expected payoff of any convex claim (Jensen). This is visible
in every heatmap in the app: price rises down the vol axis on both sides.

**Q: What's the relationship between theta and gamma?**
For a delta-hedged book, theta ≈ -½·gamma·S²·sigma². You can see the
shared `φ(d1)·sigma/sqrt(T)` machinery in the code for both Greeks.
Long gamma costs theta; that's the price of convexity.

**Q: Why is Monte Carlo convergence O(1/sqrt(n)) and what would you do
about it?** CLT: SE = std/sqrt(n) (the exact line in `mc_price`).
Remedies: antithetic variates, control variates, quasi-random (Sobol)
sequences which get ~O(1/n).

**Q: Your model says an option is cheap. Do you trade it?**
Not on this evidence. The "mispricing" is mostly the smile (my flat
historical sigma vs the market's forward-looking, skewed IV), possibly
American-exercise premium, possibly a stale last price on zero volume.
The model is a lens for reading the market's expectations, not an alpha
signal. (Being eager to trade model-vs-market "gaps" is a red flag they
screen for.)

**Q: What are Black-Scholes' assumptions, and which one breaks worst?**
Constant vol, lognormal returns, continuous frictionless hedging, constant
r, no jumps, European exercise. Constant vol breaks worst — the smile
page displays the violation directly. Real returns also jump and have fat
tails (kurtosis), which the smile is partly compensating for.

**Q: How would you extend this project?**
Done: American pricing via a CRR binomial tree (`binomial.py`) — converges
to BS as steps → ∞, and the tree also puts a number on the early-exercise
premium itself (deep ITM American put: worth more even with no dividend;
American call: only picks up a premium once there's a dividend to capture).
Also done: Greeks from Monte Carlo (`monte_carlo.py` — pathwise delta and
vega, likelihood-ratio gamma; gamma needs the score-function method
specifically because a call/put payoff's slope is discontinuous at the
strike, so pathwise differentiation — fine for delta and vega — would have
to differentiate through that kink). Still open: an IV *surface* (smile
per expiry, interpolated); Newton-with-bisection-fallback IV solver; EWMA
or GARCH vol estimates.

---

*Everything in this document maps to code in this repo — if a claim can't
be traced to a line, a formula, or a test, it isn't in here.*
