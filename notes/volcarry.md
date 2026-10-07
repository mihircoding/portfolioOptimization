# The Variance Risk Premium Sleeve

**[Live site &rarr;](https://mihircoding.github.io/options-pricer/)** — prices an option three independent ways (closed form, Monte Carlo, binomial tree) live in the browser, with all five Greeks, the live SPY volatility surface, and the delta-hedging study below.

![tests](https://github.com/mihircoding/options-pricer/actions/workflows/ci.yml/badge.svg)

A Black-Scholes options pricing tool with an interactive Streamlit interface.
Prices calls and puts from scratch (with dividend adjustment), computes the
Greeks, builds ten classic multi-leg strategies, cross-checks the closed form
against a Monte Carlo simulator, and compares model prices against live
S&P 500 option quotes from Yahoo Finance - including the volatility smile
backed out of real market prices.

It also runs the model's own hedge. `hedging_study.py` sells a one-month
at-the-money SPY call every month from 2007 to 2024, delta-hedges it daily, and
decomposes the result: implied vol averaged 20.0% against 16.2% realized, the
seller made 0.405 per $100 a month at a Sharpe of 2.20, and lost 2.675 in March
2020. The gamma decomposition reproduces the realized P&L with r = 0.99, which
is what makes "a delta-hedged option is a bet on variance" a measurement rather
than a slogan. See below.

Built with Python, NumPy, SciPy, Matplotlib, Plotly and Streamlit.

## Running it locally

On Windows, just double-click `run.bat`. Or from a terminal:

```
pip install -r requirements.txt
python -m streamlit run streamlit_app.py
```

Your browser opens at http://localhost:8501 - that IS the website, served
from your machine. `streamlit_app.py` is the landing page; the five tool
pages live in `pages/` and show up in the left sidebar automatically.

Sanity-check the math (textbook values, put-call parity, greeks vs numerical
derivatives, implied-vol round trip):

```
python -m pytest tests/volcarry -q
```

83 checks, including the arbitrage conditions above run against a chain
generated from Black-Scholes (where they must all hold exactly) and against
the same chain with one quote bent (where the right one must fire).

(`python -m pytest test_sanity.py` runs the same checks.)

The project page in `docs/` computes most of itself in the browser. The
volatility surface and the hedging study are real data, written to
`docs/data.js` by `python docs/build_data.py` (or `... build_data.py hedging`
to redo just the hedging part and keep the surface as it is).

## Putting it on the internet (free, no server needed)

Streamlit Community Cloud hosts Streamlit apps straight from a GitHub repo:

1. Make this repo **public** on GitHub (Settings -> General -> Danger Zone
   -> Change visibility). Streamlit's free tier only deploys public repos.
2. Go to https://share.streamlit.io and sign in with GitHub.
3. Click "Create app" -> "Deploy a public app from GitHub".
4. Pick this repo, branch `main`, main file `streamlit_app.py`. Deploy.

A couple of minutes later you get a permanent public URL like
`https://<yourname>-options-pricer.streamlit.app` that anyone can open -
live market data included. It redeploys itself every time you push.

## What's on each page

**1. Options Pricer (main page)** - set spot, strike, expiry, rate and
volatility in the sidebar. You get the fair call/put value up top
(green/red), price heatmaps over a spot x volatility grid, PnL heatmaps
(green = profit, red = loss, relative to what you paid for the option), and
the five Greeks. Hover any heatmap cell for exact numbers; every sidebar
input has a (?) blurb explaining the parameter.

**2. Strategy Builder** - pick one of ten strategies (covered call, married
put, bull call spread, bear put spread, protective collar, long straddle,
long strangle, long call butterfly, iron condor, iron butterfly). The page
shows the legs, the net debit/credit, the payoff diagram at expiry (green
above breakeven, red below), max profit / max loss / break-evens, and a
spot x volatility PnL heatmap you can slide through time.

**3. Market Comparison** - pick any S&P 500 stock and expiration. The page
pulls the live option chain, prices every near-the-money strike with this
project's own Black-Scholes code (using 1-year historical volatility and the
stock's actual trailing dividend yield), and shows model vs. market side by
side. Green rows: the market pays more than history justifies; red: less. It
backs implied volatility out of market prices with the project's own
bisection solver and plots it per strike - the volatility smile.

**4. Monte Carlo** - prices the same option by simulating thousands of
random price paths under the model's own assumption (geometric Brownian
motion) and shows the estimate converging onto the closed-form price, along
with the simulated paths themselves and the terminal price distribution
split into in-the-money (green) and worthless (red) outcomes.

**5. Vol Surface** - builds the implied volatility surface from a live
chain: the smile per expiry plotted in log-moneyness, the term structure of
at-the-money vol, 25-delta risk reversal and butterfly per expiry, a
calendar no-arbitrage check, and a bar chart of what pricing the whole chain
with one flat volatility costs strike by strike. See below for what makes
this different from plotting Yahoo's IV column.

## How the code works

### `black_scholes.py` - the model

The Black-Scholes formula prices a European option under the assumption
that the stock follows a random walk with constant volatility. The whole
formula hangs on two numbers:

```
d1 = [ln(S/K) + (r + sigma^2/2) T] / (sigma sqrt(T))
d2 = d1 - sigma sqrt(T)
```

`N(d2)` is (roughly) the risk-neutral probability the option finishes in the
money; `N(d1)` weights that probability by how big the payoff is when it
happens. The prices are then:

```
call = S N(d1) - K e^(-rT) N(d2)        # what you get - what you pay, discounted
put  = K e^(-rT) N(-d2) - S N(-d1)
```

The only library math used is `scipy.stats.norm` for the normal CDF/PDF -
`d1`, `d2`, prices, all five Greeks and the implied-vol solver are hand-written.

The **Greeks** are the derivatives of that formula, i.e. sensitivity of the
price to each input: delta (per $1 of stock), gamma (delta's own rate of
change), vega (per vol point), theta (per day), rho (per rate point). The
test file verifies each one against a numerical bump-and-reprice derivative,
so the closed forms are provably consistent with the pricing function.

**Implied volatility** goes the other way: given a market price, find the
sigma that reproduces it. There's no closed form, so `implied_vol()` uses
bisection - price is monotonically increasing in volatility, so we keep
halving an interval [0.0001, 5.0] until model price matches market price.

### `strategies.py` - one representation, ten strategies

Every strategy is just a list of legs, e.g. an iron condor is:

```python
[ {kind: put,  strike: 80,  qty: +1},   # buy the far put   (wing)
  {kind: put,  strike: 90,  qty: -1},   # sell the near put
  {kind: call, strike: 110, qty: -1},   # sell the near call
  {kind: call, strike: 120, qty: +1} ]  # buy the far call  (wing)
```

Because of that one representation, three short generic functions do all the
work for all ten strategies:

- `strategy_cost()` - price each option leg with Black-Scholes, stock legs
  at spot, sum with signs. Negative total = you opened the trade for a credit.
- `strategy_payoff()` - value at expiry: `max(S-K, 0)` per call,
  `max(K-S, 0)` per put, times quantity, summed over legs.
- `strategy_value()` - value *before* expiry, repricing every leg with
  Black-Scholes at the remaining time. This is what makes the strategy PnL
  heatmap respond to volatility: an iron condor that's "safe" at expiry can
  still be underwater halfway there if volatility spikes.

PnL everywhere is just `value - cost`.

### Dividends (the `q` parameter)

Real stocks pay dividends, and holding an option doesn't entitle you to
them. The standard fix (Merton's extension) is to discount the spot price by
`e^(-qT)` everywhere it appears in the formulas, where q is the continuous
dividend yield. Every function in `black_scholes.py` takes `q` with a
default of 0, so the rest of the project works unchanged; the market page
estimates q from the stock's actual trailing 12 months of dividends.
Dividends drag the forward price down, so they make calls cheaper and puts
dearer - the test suite checks the dividend-adjusted put-call parity
`C - P = S e^(-qT) - K e^(-rT)` and re-verifies delta and theta against
numerical derivatives with q > 0.

### `monte_carlo.py` - the model priced a second way

Black-Scholes assumes the stock follows geometric Brownian motion. Under
the risk-neutral measure the terminal price is

```
S_T = S * exp( (r - q - sigma^2/2) T + sigma sqrt(T) Z ),   Z ~ N(0,1)
```

`terminal_prices()` draws thousands of those in one vectorized NumPy call;
`mc_price()` averages the payoffs, discounts by `e^(-rT)`, and reports a
standard error (payoff std / sqrt(n)) so you know how tight the estimate
is. `convergence_curve()` shows the running mean homing in on the closed
form - same assumption, completely different method, same answer. The test
suite requires the two prices to agree within 4 standard errors at 500k
paths, with and without dividends.

**Greeks by simulation, two different techniques.** `pathwise_delta()` and
`pathwise_vega()` differentiate the simulated PATH: `S_T` is a smooth
function of both spot and volatility (`dS_T/dS = S_T/S`,
`dS_T/dsigma = S_T*(sqrt(T)*Z - sigma*T)`), so the derivative can be pushed
inside the expectation and estimated straight from the same paths used for
pricing. That trick breaks for gamma - it needs the derivative of the
payoff's *slope*, and a call/put's slope jumps at the strike, so pathwise
differentiation would need to differentiate a discontinuity.
`likelihood_ratio_gamma()` sidesteps this by differentiating the
*probability density* of `S_T` instead of the payoff (the "score function"
method): the density stays smooth even where the payoff doesn't, so the
same trick that fails for gamma via one route works via the other. All
three are estimated from the same underlying draws (`_implied_z()` recovers
each path's `Z` algebraically from its `S_T` rather than redrawing it, so
there's no risk of the Greek estimators quietly using different randomness
than the price they're being compared against) and reported with their own
standard errors via `mc_greeks()`, same "estimate, not exact number"
discipline as `mc_price()`. Live on the Monte Carlo page, and in
`test_sanity.py` checked against the closed-form Greeks within 4 SE, with
and without dividends.

### `binomial.py` - American exercise, priced a third way

Black-Scholes and the Monte Carlo engine above both price *European*
exercise only - the model can't ask "what if I exercised early?" because
the closed-form solution assumes you can't. Most listed US equity options
are American-style, so this module builds a Cox-Ross-Rubinstein binomial
tree, which can: at every node, walking backward from expiry, it compares
holding the option against exercising it immediately and keeps whichever
is worth more.

Two things worth knowing from running it:

- **European binomial converges to Black-Scholes** as the tree gets more
  steps - a third method (discrete tree vs. Monte Carlo vs. closed form)
  landing in the same place, which is the whole point of cross-checking a
  pricing model three different ways instead of trusting one derivation.
- **The early-exercise premium is real and it isn't the same for calls and
  puts.** With no dividend, an American call is worth exactly the same as
  its European twin - there's nothing to gain by exercising early and
  giving up remaining time value for free. A deep in-the-money American
  put is a different story even with no dividend (locking in the strike
  early starts earning interest on it sooner), and once a dividend is
  added, American calls pick up a premium too. `test_sanity.py` checks
  all three of those directly instead of assuming they hold.

### `market_data.py` - live data


- S&P 500 tickers are scraped from Wikipedia with `pandas.read_html`
  (hardcoded 60-ticker fallback if offline).
- `historical_volatility()` computes the classic realized-vol estimate:
  standard deviation of daily log returns, annualized by sqrt(252 trading
  days). That's the sigma the model uses on the market page.
- Spot prices and option chains come from `yfinance`.

### `heatmaps.py` - the grids

`price_grid()` evaluates Black-Scholes over every (spot, vol) combination -
one vectorized NumPy call per row. `heatmap_figure()` renders it with Plotly
so cells respond to mouse-over. PnL mode uses a red-yellow-green colorscale
with the color range forced symmetric around zero, so yellow always sits
exactly on break-even and green/red always mean profit/loss.

### `vol_surface.py` - the smile, measured

Page 3 has always said the market prices a smile while the model uses one
flat volatility. This module turns that sentence into numbers, and three
decisions in it are most of the difference between a surface and a plot of
Yahoo's `impliedVolatility` column.

**The forward comes from put-call parity, not from the spot.** Black-Scholes
needs a forward, and a forward needs a dividend yield and a borrow rate that
nobody publishes. The options are already quoting both: `C - P = e^(-rT)(F - K)`
is a straight line in K, so regressing call-minus-put on strike gives the
discount factor as the slope and the forward from the intercept. No dividend
estimate, no borrow assumption. The fit's R² comes back with the answer, and
a poor fit means the chain is refused rather than quietly built on.

**Only out-of-the-money quotes are used** - puts below the forward, calls
above. In theory both legs carry the same information; in practice the OTM
one is liquid, tighter, and almost all time value, so its price is mostly a
statement about volatility rather than about intrinsic value.

**Mids, and the quotes are filtered first.** `lastPrice` is whenever that
contract last traded, which on a far strike can be days ago at a different
spot. Zero bids, crossed markets and spreads wider than half the mid are
dropped: a quote whose bid-ask straddles ten volatility points does not pin
down a volatility, and averaging it in is how a surface grows spikes that
get explained as skew.

What comes out of SPY on a normal day: ATM vol rising from about 14% at a
week to 17% at two years, a 25-delta risk reversal of +4 to +5 volatility
points at every expiry (downside protection is dearer than upside, which is
the equity skew), a positive butterfly, and forwards that rise with maturity
at roughly the financing rate. The calendar check - total variance must not
fall as maturity rises at fixed moneyness - is run and reported rather than
assumed.

The honest caveat, stated in the module too: US single-name and ETF options
are American and this inverts a European formula. Restricting to OTM quotes
keeps the early-exercise premium small, but it is not zero on deep strikes
and long maturities, so these IVs read slightly high. The fix is inverting
the binomial tree instead, at roughly 500x the cost per quote - a real
trade-off, not an oversight.

### `arbitrage.py` - does the chain price a distribution at all?

`vol_surface.py` checks the time direction: total variance must not fall as
maturity rises, or a calendar spread is free money. `arbitrage.py` checks the
strike direction, where three conditions follow from the payoff alone and need
no model: the call price falls as the strike rises, a call spread never costs
more than it can pay, and the call price is convex in the strike. Puts are
converted to calls through the parity forward first, so one continuous curve
spans the whole strike range.

Convexity is the interesting one, because the second derivative of the call
price *is* the risk-neutral density (Breeden-Litzenberger). A butterfly quoted
at a negative price is the market assigning negative probability to a range of
prices, which never happens - what happened is that three quotes were not alive
at the same instant, or one leg is stale, or the forward is off.

Which is why every violation here is measured against the bid-ask spread of the
legs you would have to trade. On SPY across six expiries:

```
expiry        days  strikes      verticals        butterflies      worst  density
                                raw  tradable     raw  tradable        $     mass
2026-10-01       8      110       1         0      34         0    -0.02    1.000
2026-10-16      22      157       4         0      48         4    -0.10    1.000
2026-11-20      58       82       0         0      11         1    -0.97    0.996
2027-01-29     128      169       8         8      44         5    -1.22    0.983
2027-09-17     358      120       1         1       9         1    -2.03    0.952
2029-01-19     848       80       1         0      28         0    -1.54    0.859
```

**189 conditions violated by the mid prices; 20 by more than the spread.** That
gap is what a screen built on mids with no spread filter reports as
opportunities, and it is the reason those screens produce no trades.

The density column is the implied distribution integrated over the quoted
strikes. It is 1.000 at a week and 0.859 at two years: the far-dated chain
simply does not quote enough of the tails to account for the distribution, so
anything computed from it - an expected value, a tail probability - is missing
14% of its mass, and would read as a confident number if nobody checked.

`arbitrage_study.py` also asks what happens between the quoted strikes, since a
surface quoted at 40 strikes gets used at any strike. Interpolating total
variance and interpolating volatility produce almost identical numbers of bad
butterflies (61 vs 54 on a 200-strike grid at one expiry) - the choice hardly
matters, because a straight line between two quoted IVs inherits whatever
non-convexity the quotes already had. A surface you can price a book with has
to be *fitted* under the convexity constraint rather than joined up dot to dot.

### Fitting it instead: SVI

`svi.py` is that fit. For log-moneyness k = log(K/F) it models total variance
w = sigma^2 T with Gatheral's raw SVI:

```
w(k) = a + b * ( rho * (k - m) + sqrt((k - m)^2 + sigma^2) )
```

Five parameters, each one a feature of the smile you can point at: `a` the
level, `b` the wing slope, `rho` the tilt (negative for an equity index,
because the crash is on the put side), `m` where the minimum sits and `sigma`
how rounded the bottom is. The reason for this function rather than a spline
is its shape: it is a hyperbola, so it is convex in k by construction and goes
linear in both wings, which is exactly what the no-arbitrage conditions ask
for. A cubic spline has no such shape and will turn over in the wings however
well it fits the quoted points.

Two things make it work in practice.

**The fit is constrained, not checked afterwards.** Gatheral and Jacquier give
the risk-neutral density's sign in closed form from the parameters - g(k),
implemented as `durrleman_g` - so a candidate can be rejected before it is
ever accepted. The search only ever keeps parameters whose g stays
non-negative across the quoted range and beyond it. That is the whole
difference from interpolating: the surface cannot price a negative
probability, for a structural reason rather than a lucky one.

**The five-parameter fit is really a two-parameter search.** For fixed
(m, sigma) the model is linear in (a, b*rho, b), so those come from a small
weighted least-squares solve and only (m, sigma) get searched - a grid then a
local refine. SVI has well-documented local minima; reducing the non-convex
part to two dimensions removes that failure mode for about a thousand
three-parameter solves, which is microseconds.

On SPY across six expiries:

```
expiry          n   fit err   worst       in  bad butterflies    min g         wings
                    vol pts  vol pt   spread  linear      svi            left  right
2026-10-06    102      0.19    1.02    9/102      86        0    0.046   0.03   0.00
2026-10-16    156      0.32    1.86   12/156      69        0    0.050   0.05   0.00
2026-11-20    162      0.27    1.38    8/162      35        0    0.162   0.07   0.01
2027-01-29    169      0.13    0.60   25/169      27        0    0.198   0.11   0.01
2027-09-17    120      0.33    1.28    9/120      16        0    0.211   0.18   0.03
2029-01-19     98      0.38    1.24   89/98       53        0    0.089   0.43   0.00
```

**Zero bad butterflies at every expiry**, against 16 to 86 for the linear
interpolation of the same quotes on the same grid. That column is the point of
the exercise.

The `in spread` column is the part worth being honest about, and it is not
flattering. A single five-parameter function **cannot** reprice a liquid SPY
chain inside its own bid-ask spreads: at one week it lands between the bid and
the ask on 9 strikes out of 102, and the worst miss is 32 half-spreads. At two
years it is 89 out of 98 - not because the fit got better (the error in vol
points is *larger* there) but because the spreads got wide enough to hide it.

That is the real trade being made, and it is worth saying out loud rather than
quoting an r-squared: **SVI is a no-arbitrage interpolator, not a repricer.**
Its job is to give a usable surface at every strike that was never quoted, and
the price of guaranteeing no arbitrage everywhere is that it will not match the
most liquid strikes to the penny. A desk that needs both runs SVI for the
surface and carries a per-strike residual on top of it.

One smaller result along the way - weighting each quote by its vega. RMSE is in
volatility points:

```
expiry       weighting     rmse  worst miss  in spread
2026-10-07   vega          1.26       37.1x     11/99
2026-10-07   equal         1.03       78.8x      4/99
2026-10-16   vega          1.38       42.9x      9/156
2026-10-16   equal         1.26      171.0x     21/156
2026-11-20   vega          0.98       39.1x      7/162
2026-11-20   equal         0.57      149.2x      3/162
2027-01-29   vega          0.24       29.7x     20/170
2027-01-29   equal         0.22       36.9x     11/170
```

Equal weighting wins on RMSE and loses on the thing that matters, by a factor
of two to four on the worst repricing error.

Equal weighting spends its accuracy on deep wing quotes whose implied vol is
mostly rounding - the option barely responds to volatility at all, so inverting
its price for a vol produces a number with very little information in it - and
pays for that at the strikes anyone trades. Fitting in the right units is a
smaller decision than choosing the parameterization and a larger one than it
looks.

**And the rmse column above used to be wrong, by a factor that changed from row
to row.** `fit_svi` works in total variance and reported its residuals in
sqrt(total variance), which is sigma·sqrt(T) rather than a volatility - under the
name `rmse_vol_points`. So identical fit quality read as a different number at
every maturity, and the one-week expiry looked about seven times better fitted
than it was: 0.19 where the answer is 1.26. The fix is one division by sqrt(T),
placed in `fit_smile`, which is the first function in the chain that knows what T
is; `fit_svi` now returns `rmse_sqrt_total_var` under that name so nothing can
read a volatility out of it by accident again. The short end of this chain fits
considerably worse than this section used to claim, and a table whose unit
changes from row to row is worse than no table.

```
python arbitrage_study.py          # SPY by default, or pass tickers
```


### `ssvi.py` - one surface, so a calendar spread has a price too

Everything above fits **one expiry at a time**, under the constraint that its own
implied density stays positive. That removes butterfly arbitrage and says nothing
whatever about the direction strikes do not run in. Six slices each individually
sound can still cross each other in maturity, and total variance that falls as
maturity rises is a calendar spread with a negative price. `vol_surface.py` has
checked for that since it was written, and checking is not excluding.

It is not a theoretical worry. Taking the six independently fitted SVI slices and
reading them against each other on a common log-moneyness grid:

```
Every slice has a positive density on its own. Between slices, 27 of
405 grid points price a calendar spread at a negative value.
    2026-10-07 -> 2026-10-16: 27 points, worst total variance drop 0.00051 at k=+0.40
```

Excluding it needs the surface to be **one object with one set of parameters**.
`ssvi.py` is Gatheral and Jacquier's SSVI (2014):

```
w(k, theta) = theta/2 * { 1 + rho*phi(theta)*k
                          + sqrt( (phi(theta)*k + rho)^2 + 1 - rho^2 ) }
```

`theta` is the at-the-money total variance of that expiry - put k=0 and the
braces collapse to 2 - so the term structure is a parameter rather than an
output. Everything else is shared across the whole surface: one `rho` for the
skew's tilt and one `phi(theta)` for how the smile's width moves with variance,
taken as the usual power law. Nine parameters for 802 SPY quotes across six
expiries, against 30 for six independent slices.

On SPY, fitted jointly:

```
  rho     -0.4702     one skew tilt for the whole surface; negative is the equity sign
  eta      0.7587     smile width scale
  gamma    0.5843     how the width decays as variance grows

expiry        days     theta  atm vol
2026-10-07       8   0.00021   10.2%
2026-10-16      16   0.00067   12.2%
2026-11-20      52   0.00259   13.6%
2027-01-29     122   0.00701   14.5%
2027-09-17     352   0.02520   16.2%
2029-01-19     842   0.06417   16.7%

  negative-density grid points   0   (400 strikes x 6 expiries)
  negative calendar spreads      0   (81 strikes x 5 adjacent pairs)
```

Both no-arbitrage conditions are closed-form statements about those three shared
parameters, enforced during the search rather than checked afterwards. The
verification lines above are deliberately run with the project's *existing*
checkers - `svi.durrleman_g` through a conversion that writes each SSVI slice in
raw SVI parameters, and a direct reading of total variance between expiries -
because a theorem quoted from a paper and a theorem implemented correctly are
different claims.

**One thing came out differently from expected.** Of the two calendar conditions,
the exotic-looking one - a bound on d(theta·phi)/dtheta - is *free* for this
phi. Working the derivative through, the ratio it bounds collapses to
(1-gamma)/(1+theta), which is below 1 for any admissible gamma, while the bound
never falls below 1.04. It cannot bind. The entire calendar guarantee therefore
comes from the plain half - theta non-decreasing in maturity - which is exactly
the half a per-expiry fit cannot even express, because each slice solves for its
own level knowing nothing about the slice beside it.

And what it costs, in volatility points:

```
expiry        days   per-slice SVI   SSVI surface   worst SSVI
2026-10-07       8           1.26p          1.91p        5.55p
2026-10-16      16           1.39p          3.43p       12.92p
2026-11-20      52           0.98p          1.50p        5.86p
2027-01-29     122           0.24p          1.12p        4.08p
2027-09-17     352           0.35p          0.95p        1.62p
2029-01-19     842           0.25p          0.35p        0.62p
```

Nine parameters reprice a liquid SPY chain worse than 30 do, and dropping the
arbitrage conditions from the search improves total-variance RMSE by 7.2% - into
a surface that is arbitrageable, which is what enforcement is for. Same
conclusion as the SVI section, one level up: **the constrained surface is not a
better repricer, it is a surface you can differentiate.** Consistency across
expiries is what lets you price a calendar spread, a variance swap, or anything
else touching two maturities at once, and the per-strike residual is what a desk
carries on top.

```
python ssvi_study.py               # SPY by default, or pass a ticker
```

### The implied-vol solver

`bs.implied_vol` was bisection. It is now Newton-Raphson on vega with a
guarded bisection fallback: **6.4 pricing calls per solve instead of 31**,
same tolerance, which matters once a surface means a thousand inversions per
chain. Every Newton step is checked against the bracket and thrown away if
it lands outside it, so the guarantee bisection gives you is never given up.

Two things changed with it that are worth more than the speed:

- **It converges on sigma, not on price.** Stopping when the price error is
  small sounds right and is wrong for exactly the options where vega is
  small - a deep in-the-money call is worth intrinsic-plus-epsilon at 5% vol
  and at 40% vol alike, so "the price matches to a millionth" can be true a
  long way from the right volatility. The old solver returned the top of its
  search range in those cases, silently.
- **It returns `nan` when the quote determines no volatility at all.** Below
  intrinsic, above the underlying, or vega too small to invert. Real chains
  produce all three constantly, and a fabricated number is how a garbage IV
  ends up plotted as a spike on a surface.

`test_sanity.py` round-trips 400 random inputs across strikes from 60 to 160
and maturities from a week to two years (worst error ~1e-8), checks each
refusal case, and rebuilds a synthetic chain from a known smile to confirm
the surface code recovers the forward, the rate and every strike's
volatility from nothing but prices.

### `hedging.py` - what the price is actually worth

Every price in this repo rests on a claim buried in the derivation: that an
option can be replicated by continuously trading the underlying, so its value
is the cost of that replication and nothing more. Nobody trades continuously.
`hedging.py` runs the replication for real - discretely, on actual price paths -
and `hedging_study.py` does it on eighteen years of SPY.

The setup: short one at-the-money SPY call on the first trading day of every
month, sell it at VIX, delta-hedge daily to expiry 21 trading days later, repeat.
215 non-overlapping trades, 2007 to 2024. Everything below is per $100 of
underlying, so 2008 and 2024 are comparable.

```
average P&L                 0.405        average implied vol        20.01%
average premium sold        2.304        average realized vol       16.20%
std dev of P&L              0.640        implied above realized      82.8% of months
annualized Sharpe            2.20
months profitable           81.4%
worst month                -2.675  (March 2020)
best month                  3.692  (December 2008)
```

That is the variance risk premium, measured rather than cited: implied
volatility averaged 20.0% against 16.2% realized, and selling that gap
systematically returned 0.405 per $100 a month at a Sharpe of 2.20. Before
reading that as a strategy, look at the worst month, and note that the study
sells exactly one option a month with no leverage, no position sizing and no
stop. Selling variance is selling insurance: you are paid a small amount very
reliably, and the occasional bill is enormous. The four worst months here are
March 2020, September 2008, November 2008 and August 2011, which is the same
list a credit desk would give you.

**One cost in that table used to be missing: the option was sold at the mid.**
`cost_bps` charges bid-ask on every share the hedge trades, which is most of
the trading here but not all of it — there is also one option sale a month, and
nobody sells at the mid. `option_spread_vol_pts` charges half the option's
bid-ask on that sale, in vol points, selling at the bid while still hedging on
the mid (which is where a desk marks, and the two volatilities were already
separate arguments to `delta_hedge`).

The size is checkable rather than asserted: one-month ATM vega is about
`S·√(T/2π)`, which at SPY 580 and 21 trading days is **$0.67 per vol point**,
and the test suite verifies that against the pricer. SPY's front-month market
runs a penny or two wide, so a cent of half-spread is around 0.02 vol points.
The default charge is 0.10 — two to five times a calm day's width, because this
sleeve trades every month from 2007 and option markets in October 2008 were not
a penny wide. It costs **0.0115 per $100 a month**, taking the average P&L from
0.405 to 0.394 and the Sharpe from 2.20 to 2.13.

So the premium survives its own transaction costs by a wide margin, which is
worth knowing in the direction it points: the edge here is not a spread artifact.
The sweep in `multi_strategy.py` section 7 runs it out to 8.00 vol points and
finds the P&L only turns negative somewhere past 3 — thirty times SPY's quoted
width.

A constant could not capture that a real spread widens in exactly the months
this strategy is most exposed, and `option_spread_vix_anchor` now does: the
half-spread is `0.10 × VIX / 17.5`, so 0.07 vol points in 2017 and 0.39 in
March 2020. The reason it is proportional rather than a step function is that a
maker's bid-ask covers the risk of being wrong about fair value, and fair value
for an option IS a volatility.

What that turns out to be worth is the useful part, and it is less than the
caveat implied. The spread as a **share of the premium sold** comes out at 0.57%
in the calm third of months, the middle third and the loud third alike — flat to
four decimal places, because the premium is vega times sigma and the spread is
vega times a width proportional to sigma, so the ratio is sigma-free. The
constant charge is the regime-dependent one: 0.79% of premium in calm months and
0.37% in loud ones, which overcharges the quiet months and undercharges exactly
the ones with the most premium at risk. `multi_strategy.py` section 8 has the
table, and the control that matters — a constant spread set to the proportional
series' own average width gives the same answer to within 0.01 of a
t-statistic, so the timing is worth almost nothing here and the average width
is worth all of it.

**The point of the exercise is the decomposition, not the Sharpe.** In
continuous time the P&L of a delta-hedged option is exactly

```
integral of  1/2 * Gamma * S^2 * (implied_vol^2 - realized_vol^2) dt
```

so once the delta is hedged away, the direction of the stock is gone and what
remains is a bet on the *difference between two volatilities*, weighted by
gamma. `gamma_pnl()` computes the discrete version term by term. On the real
SPY trades it reproduces the simulated hedge P&L with a correlation of **0.9895**
and a mean absolute error of 0.066 against a 2.30 average premium - the residual
being the third-order terms the expansion drops. This is why traders say "long
gamma" instead of "long calls".

**And the gamma weighting is not a technicality.** Regress each month's P&L on
that month's variance gap alone - implied² minus realized², no gamma - and you
get r² = 0.51 with an intercept of 0.358, which is most of the average P&L.
Add the gamma weighting back and r² goes to 0.98. Half the variation in the
outcome is *not* about how much the stock moved; it is about *when* it moved,
because an option whose spot has drifted away from the strike has almost no
gamma left and stops caring. A delta-hedged option is a path-dependent
approximation to a variance bet, which is the entire reason variance swaps
exist.

**Hedging less often doesn't cost money, it costs certainty.** Boyle & Emanuel
(1980) say the error a discrete hedge adds should have zero mean and a standard
deviation growing like the square root of the rebalancing interval. Measured
against each month's own daily hedge, so the variance premium common to all
frequencies is differenced out:

```
   rebalance   mean P&L   vs daily  error std  / sqrt(n)     worst
    every 1d      0.405          -          -          -    -2.675
    every 2d      0.422     +0.016      0.327      0.231    -2.351
    every 5d      0.378     -0.028      0.659      0.295    -4.382
   every 10d      0.365     -0.041      0.932      0.295    -7.783
   every 21d      0.412     +0.007      1.057      0.231    -4.523
```

The mean change stays inside the noise at every frequency - hedging weekly
instead of daily is not a worse trade, it is the same trade with wider error
bars and a worst case three times as bad. The last column divides out
sqrt(interval) and is flat from 2 to 10 days, then falls off at 21 because the
option is then hedged once at inception and the error has nowhere left to grow.
At 1bp of transaction cost the average P&L goes from 0.405 to 0.382, and at 5bp
to 0.288 - so on this trade, at this size, costs matter less than the choice of
hedging frequency does.

**Caveats, because this one has more than most.** VIX is SPX's implied
volatility, not SPY's, and it is a variance-swap-style index across the whole
strike range rather than the at-the-money vol the study treats it as - it runs
roughly a point above ATM vol, so the measured premium is a touch generous.
Prices are dividend-adjusted and the risk-free rate is zero, folding both carry
terms into the path. The hedge uses the vol the option was sold at and never
re-marks, where a desk would re-hedge on current implied, which damps the tails.
None of these change the shape of any result above; all of them would move the
second decimal place.

The checks in `test_sanity.py` are the part worth reading. A motionless stock
pays the seller the entire premium to within 1e-9. A hedged call and a hedged
put on the same strike earn identical P&L on every path, because with zero rates
their deltas differ by exactly one share held statically - put-call parity
restated as a statement about hedging. Selling at 30% into a stock that realizes
10% wins on 100% of simulated paths and selling at 20% into a stock that realizes
40% loses on 100% of them. And hedging at the path's own volatility has a mean
P&L of zero to within three standard errors, which is the replication argument
itself, checked rather than assumed.

## Things to notice when comparing to the market

- Market prices rarely match the model exactly. The model uses one flat
  historical volatility; the market prices each strike with its own implied
  volatility (the "smile/skew" - downside puts usually carry higher IV
  because crash insurance is in demand). Page 5 measures that gap in dollars
  per contract rather than leaving it as a remark.
- `lastPrice` on illiquid strikes can be hours old - check volume before
  concluding an option is mispriced.
- The closed form prices European exercise only; US single stock options are
  American-style, so deep in-the-money puts on dividend payers will show the
  largest model-vs-market gaps. `binomial.py` prices the American version.
