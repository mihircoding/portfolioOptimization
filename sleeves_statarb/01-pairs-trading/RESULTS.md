# Results

## The scan

4,950 pairs tested across the S&P 100 (`scan.py`, formation window
2013-01-01 - 2020-12-31, Engle-Granger p-value on each pair's OLS residuals).

| | |
|---|---|
| Pairs tested | 4,950 |
| Passed at 5% | 930 (18.8%) |
| Expected false positives at 5%, by chance alone | ~247.5 |
| Bonferroni threshold (0.05 / 4,950) | 1.01e-05 |
| Pairs surviving Bonferroni | **1** (ACN / LIN) |

930 pairs "pass" a 5% threshold when random chance alone predicts ~248 false
positives out of 4,950 tests - the multiple-comparisons problem the
project's own pitfalls checklist warns about, made concrete. Almost all of
those 930 are noise. Only one pair, ACN/LIN, survives correcting for testing
4,950 hypotheses at once.

## Trading the one pair that's statistically real

Formation-window OLS gives ACN/LIN a hedge ratio of 0.9317. Trading that
fixed beta out of sample (2021-01-04 - 2025-12-30, same z-score rule as
every other pair in the scan: 60-bar window, entry 2.0, exit 0.5, 5bps cost):

| | |
|---|---|
| Total return | **-6.06%** |
| Sharpe | **-0.11** |
| Max drawdown | -23.54% |
| Round trips | 23 |
| Out-of-sample ADF p-value | 0.179 |

The pair that passed the hardest statistical bar available didn't just lose
money out of sample - it doesn't even test as cointegrated anymore in the
trading window (p=0.18, nowhere near 0.05). Whatever tied ACN and LIN
together in 2013-2020 either weakened or the fixed beta stopped describing
it. Either way, this is the project's honest headline: passing a
Bonferroni-corrected test on formation data is necessary for "this pair
isn't noise," but it is nowhere near sufficient for "this pair is
tradeable."

## Is that the fixed hedge ratio's fault?

`kalman_vs_static.py` asks that directly instead of assuming an answer.
Same pair, same trading-window prices, same z-score rule, same costs - the
only change is a from-scratch Kalman filter (`src/pairs.ipynb ::
kalman_hedge_ratio`, the stretch goal below) tracking beta continuously
through formation *and* trading, instead of freezing it at the end of
formation.

| | total return | Sharpe | max dd | round trips |
|---|---|---|---|---|
| Static beta (0.9317, frozen) | -6.06% | -0.11 | -23.54% | 23 |
| Kalman beta (continuously updated) | **+3.79%** | **+0.16** | **-10.97%** | 37 |

The Kalman beta starts trading at 0.976 - close to the static value, as it
should be, since it's seen the same formation data - and drifts to 0.632 by
the end of 2025. That's a real, substantial move in the actual relationship
between these two stocks, not filter noise: a hedge ratio frozen in 2020
was pricing the spread against a relationship that had already changed by
2023-2025. Letting beta track that drift turns a losing strategy into a
small winner with roughly half the drawdown - at the cost of 14 extra round
trips (37 vs 23), which is exactly what you'd expect from a beta that's
allowed to keep moving.

**What this does and doesn't show.** One pair, one delta setting, no
transaction-cost sensitivity on the extra turnover, no walk-forward
validation of the Kalman parameters themselves. It does not rescue the
project's headline finding - a single pair flipping from a small loss to a
small gain under a smarter estimator is not evidence that stat-arb "works,"
and 4,949 other pairs still failed the Bonferroni bar entirely regardless
of what hedge ratio they'd have used. What it does show: for the one pair
that cleared the statistical bar, the specific way it lost money out of
sample (a stale hedge ratio, not a broken relationship) was diagnosable and
partially fixable. That distinction - "this pair is real but was traded
wrong" vs. "this pair was never real" - is exactly the kind of honest
follow-up question a scan like this one should raise.

## Trading all of them at once

Everything above rests on one pair. That is a fair headline and a fragile
one - ACN/LIN is a single draw, and a single draw can say anything. The
README has listed the fix as a stretch goal from the start: trade the
survivors as a portfolio and ask whether *any* aggregate edge is there once
you stop picking one pair to look at. `portfolio.py` does that.

Every one of the 930 survivors, re-run through the same backtest (same
formation beta, same 60-bar z-score, same entry 2.0 / exit 0.5, same 5bps),
then equal-weighted. Taking the top N by formation p-value:

| N pairs | total return | Sharpe | max dd | ann vol |
|---|---|---|---|---|
| 1 | −6.06% | −0.11 | −23.54% | 8.30% |
| 5 | +4.46% | 0.17 | −9.68% | 6.59% |
| 10 | +6.44% | 0.32 | −6.35% | 4.19% |
| 25 | +5.45% | **0.34** | −5.21% | 3.29% |
| 50 | +1.74% | 0.12 | −6.02% | 3.20% |
| 100 | +0.97% | 0.08 | −4.42% | 2.75% |
| 250 | −1.06% | −0.07 | −4.26% | 2.49% |
| 500 | −1.25% | −0.10 | −2.78% | 2.18% |
| 930 | −2.14% | −0.19 | −4.04% | 2.13% |

The top-25 book at 0.34 is the most flattering number this project has
produced, and it is the one to be most careful with: N was chosen after
seeing the column. The honest reading of that table is the whole shape, and
the whole shape says the peak is where a peak has to be when you sweep a
parameter over noise.

### Does the screen rank anything?

If the formation p-value carries information, the strongest survivors should
out-trade the weakest. Split the 930 into five equal buckets by p-value and
trade each as its own book:

| Bucket | p-value range | total return | Sharpe |
|---|---|---|---|
| 1 (strongest) | 6.8e-08 – 3.8e-03 | −2.50% | −0.17 |
| 2 | 3.9e-03 – 1.1e-02 | +0.41% | 0.05 |
| 3 | 1.1e-02 – 2.3e-02 | −1.90% | −0.15 |
| 4 | 2.3e-02 – 3.5e-02 | −3.32% | −0.28 |
| 5 (weakest) | 3.5e-02 – 5.0e-02 | −3.51% | −0.28 |

There is a hint of a gradient in the bottom half and none at the top: the
strongest bucket does worse than the second. Across all 930 pairs the rank
correlation between formation p-value and out-of-sample Sharpe is **−0.053**
(z = −1.6 against the null of no association) - the direction you would want,
and nowhere near large enough to act on.

That −0.05 has been quoted in this repo's README since the first scan.
What's new is that it is now reproducible from `portfolio.py` rather than an
unsourced number, and that it comes with the thing it implies: a book built
from the strongest fifth of the survivors does not beat a book built from
the weakest fifth. **The screen does not rank.** Eight years of formation
data, a test with a Nobel in its ancestry, 4,950 hypotheses - and the
resulting ordering carries about as much information as a coin.

That is a stronger statement than the ACN/LIN result, because it is made
across 930 pairs instead of one.

### The control group: trade the pairs the screen threw away

Everything above describes the 930 pairs that passed. Compared to what? A
screen that keeps 930 of 4,950 is only worth running if the 4,020 it rejected
would have done worse, and nobody checks, because the rejects are by definition
the ones you don't trade. `control.py` trades them anyway — same window, same
hedge ratios from the same formation regression, same 60-day z-score, same
2.0/0.5 thresholds, same 5bps a side. The only difference between the groups is
which side of p = 0.05 they landed on.

| | passed the screen | rejected |
|---|---|---|
| pairs | 930 | 4,020 |
| mean Sharpe | −0.009 | **+0.009** |
| median Sharpe | −0.008 | **+0.010** |
| median total return | −3.66% | **−2.78%** |
| share profitable | 41.8% | **45.5%** |
| mean round trips | 20.9 | 20.5 |

**The pairs that failed the cointegration test did slightly better than the
ones that passed.** The difference in mean Sharpe is −0.018 in favour of the
rejects, and a permutation test over 10,000 relabellings puts it at p = 0.21 —
so the honest reading is not "the screen is backwards", it is that the two
groups are the same group. A t-test would be the wrong tool here: each ticker
appears in 99 pairs, so the samples overlap heavily and the usual standard
errors are too small in the direction that flatters the screen.

As books:

| Book | Pairs | Total return | Sharpe |
|---|---|---|---|
| screened (passed) | 930 | −2.14% | −0.19 |
| rejected | 4,020 | −1.30% | −0.10 |
| everything | 4,950 | −1.45% | −0.12 |

And the ranking question, now over the full range of the test instead of the
tenth of it the survivors occupy — out-of-sample Sharpe by formation p-value
decile:

```
 decile          p-value range   pairs  mean sharpe  median ret  profitable
      1     0.0000-0.0190          495        0.011      -2.80%       43.4%
      2     0.0190-0.0551          495       -0.039      -4.87%       39.2%
      3     0.0551-0.1013          495        0.020      -1.15%       47.9%
      4     0.1016-0.1631          495        0.017      -2.91%       44.8%
      5     0.1631-0.2309          495        0.004      -2.91%       45.7%
      6     0.2318-0.3166          495        0.015      -2.35%       46.1%
      7     0.3166-0.4345          495        0.026      -1.14%       48.1%
      8     0.4348-0.6024          495        0.036      -0.85%       48.5%
      9     0.6024-0.8450          495       -0.012      -4.29%       43.4%
     10     0.8455-1.0000          495       -0.020      -5.62%       41.2%
```

No gradient anywhere in it, and the Spearman correlation between p-value and
out-of-sample Sharpe across all 4,950 pairs is **−0.001**. The −0.05 measured
on the survivors alone was the interesting tenth of a flat line.

This is the cleanest result in the project. The bucket table earlier shows the
screen doesn't rank the pairs it keeps; this shows it doesn't separate the ones
it keeps from the ones it discards either. Everything else here — the single
pair, the Kalman filter, the quarterly refits — is an argument about how to
trade a signal. This is the check on whether there is a signal, and it is the
one experiment in the repo that a desk would ask for first.

### Where the book's Sharpe comes from

| | |
|---|---|
| Mean single-pair Sharpe | −0.009 |
| Median single-pair Sharpe | −0.008 |
| Share of pairs with positive Sharpe | 48.8% |
| Share of pairs that actually made money | 41.8% |
| Average pairwise correlation of pair returns | 0.033 |
| Book Sharpe, predicted by the arithmetic | −0.05 |
| Book Sharpe, realized equal-risk | −0.05 |
| Book Sharpe, realized equal-weight | −0.19 |

Just under half the pairs have a positive Sharpe. That is a coin, and it is
the finding: the survivors' returns are centered on zero, so averaging them
gives zero however many you average.

The two "share" rows are worth separating, because they are not the same
test and the repo has quoted both. 48.8% have a positive mean daily return;
only 41.8% finish the trading window above where they started. The seven
points in between are volatility drag - a spread strategy that averages zero
compounds to slightly less than zero, and the wider the swings the more it
loses to that. Nothing is wrong with either number; they answer different
questions, and the second is the one an investor would ask.

The arithmetic is worth spelling out because it is the reason a portfolio
cannot rescue this. Averaging N equally-risked series with mean pairwise
correlation ρ multiplies Sharpe by `sqrt(N) / sqrt(1 + (N-1)ρ)`. At ρ = 0.033
and N = 930 that factor is about 5.4, and it is applied to a mean pair
Sharpe of −0.009. Diversification works exactly as advertised here - it
faithfully magnifies a number that is zero.

The gap between the two realized rows is its own small lesson. The prediction
is about equally-*risked* series, and these pairs' volatilities span 6.8x. The
correlation between a pair's volatility and its out-of-sample Sharpe is −0.27:
the noisier pairs did worse. Equal *dollar* weighting therefore hands the most
risk to the pairs that deserve the least, which is where −0.05 becomes −0.19.
Equal-risk weighting uses trading-window volatility and so isn't a strategy
anyone could have run - it is a diagnostic, and what it diagnoses is that
even the sizing question was working against the book.

### And it isn't 930 independent bets anyway

930 pairs drawn from 100 tickers is 1,860 legs over 100 names. The most
frequent leg is GILD at 65 pairs - 3.5% of the book's gross exposure in one
stock, before counting the 64 different things it is paired against. A book
like this is a handful of concentrated single-name positions wearing 930
labels, and the ρ = 0.033 above is small largely because the legs point in
different directions, not because the bets are unrelated.

## Re-estimating every quarter

Every result above this line estimates once. One formation window picks the
pairs, fixes each hedge ratio, and then five years are traded without ever
looking again. That is the single most obvious objection to the whole study,
because it is not how anyone runs a pairs book: a relationship that was
cointegrated over 2015-2020 has no obligation to stay that way, and a hedge
ratio estimated in 2020 is a statement about 2020.

`walkforward.py` settles it. Every quarter, re-scan all 4,950 pairs on the
trailing three years, keep whatever passes, re-estimate its hedge ratio on the
same window, and trade that book for one quarter. Nothing ever sees data from
its own quarter. 20 quarterly re-estimates, 99,000 cointegration tests, on the
identical 2021-2025 window as everything above.

| Variant | Return | Sharpe | Vol | Max DD | Pairs held | Roster churn |
|---|---|---|---|---|---|---|
| frozen selection, frozen beta | −1.27% | −0.12 | 1.98% | −4.16% | 930 | 0% |
| frozen selection, quarterly beta | +0.08% | 0.02 | 2.11% | −3.12% | 930 | 0% |
| quarterly selection and beta | −0.57% | −0.04 | 2.26% | −2.99% | 876 | 55% |
| quarterly, p ≤ 0.01 | +0.06% | 0.02 | 2.53% | −3.92% | 278 | 66% |
| quarterly, z-window from half-life | −0.02% | 0.01 | 2.25% | −3.69% | 876 | 55% |

The first row reproduces the frozen book from the section above. It reads
−1.27% here against −2.14% there because this driver restarts each pair's
z-score at every quarter boundary with a trailing warmup, which is a slightly
different rule, not a different result — both are zero.

**Refitting does not rescue it, and the interesting part is how completely it
fails to matter.** Five different answers to "what should we re-estimate, and
how often" produce five numbers between −1.27% and +0.08% over five years.
Nothing helps. Nothing hurts. Tightening the screen from 5% to 1% cuts the book
from 876 pairs to 278 and the five-year return comes out at +0.06%. Replacing
the fixed 60-bar z-score window with each pair's own measured half-life — the
last open stretch goal in this file — gives −0.02%. Each is about half a point
away from the 5% book it modifies, over five years.

That is what it looks like when a signal is zero rather than mis-implemented.
A parameter genuinely on the wrong setting shows up as a result that changes
when you change it.

### The number that explains the whole project

```
Pairs passing at 5% per quarter:  min 651, median 799, max 1555  (out of 4,950)

Of 4,950 pairs:      856  never pass in any of the 20 quarters
                       0  pass in all 20
                   4,094  pass in some quarters and not others

Median quarters passed, among pairs that ever pass:  4 of 20
```

**Not one pair out of 4,950 is cointegrated in all twenty quarters.** The
typical pair that passes at all passes in four quarters out of twenty and fails
in the other sixteen. The roster turns over 55% every three months.

Cointegration is supposed to be a property of a relationship, not of a window.
Two stocks that are genuinely tied together — same business, same inputs, same
customers — do not become untied for nine months and then tied again. A screen
whose verdict survives one quarter in five is not measuring a property of the
pair; it is measuring which noise happened to look like reversion in the last
three years of data, and re-running it quarterly just draws a fresh sample of
noise.

This is the same conclusion the p-value quintile test reached from the other
direction. That one showed the screen cannot *rank* pairs: the strongest fifth
did worse out of sample than the second, and the rank correlation between
formation p-value and out-of-sample Sharpe was −0.05. This one shows it cannot
*repeat*. Between them there is not much left of the screen, and the honest
summary is that 930 pairs "passing at 5%" was always a statement about testing
4,950 hypotheses rather than about 930 relationships.

Worth being precise about what is and is not being claimed: 651 to 1,555 pairs
pass per quarter against the ~248 that pure chance predicts, so there is
genuine common structure in the universe — 100 large-cap US stocks share
sectors, factors and a market. The claim is narrower. That structure is not
stable at the pair level, on this horizon, at a strength this screen can find,
and none of the standard fixes change that.

### Is the screen's p-value a p-value?

Everything above takes the screen's own arithmetic at face value and shows
the arithmetic does not pay. This asks the question underneath it: is the
number being thresholded the thing it claims to be?

The screen does what nearly every pairs tutorial does. Regress one log price
on the other, take the residual, run an augmented Dickey-Fuller test on it:

```
beta   = OLS(y ~ x)
spread = y - a - beta * x
p      = adfuller(spread).pvalue
```

**The residual is not data.** It is the output of a regression chosen to make
that residual as small as possible, so it looks more stationary than a series
of the same statistical character would — the fit has already spent some of
the wandering. Dickey-Fuller's tables were computed for a series nobody fitted
anything to. This is exactly why Engle and Granger, and then Phillips and
Ouliaris, published *separate* critical values for the case where beta is
estimated, and why statsmodels ships `coint()` alongside `adfuller()`.

`screen_validity.py` measures what that costs, starting with the only
definition of a test's size that needs no theory: feed it pairs of independent
random walks, where there is nothing to find, and count how often it says
there is.

```
test                                         rejects at 5%    at 1%
adfuller on the OLS residual (the screen)           15.3%     4.5%
Engle-Granger critical values (coint)                5.7%     1.1%
                                   (2,000 simulated pairs, 1,384 observations)
```

**A test labelled 5% rejects 15.3% of the time on data with nothing in it.**
Three times its nominal size. The screen's `p < 0.05` is, in truth, roughly
`p < 0.011`.

Re-running all 4,950 real pairs through the correct table:

```
passed the screen as written         930  (18.8%)
pass with the right table            541  (10.9%)
pass both                            486
rank correlation of the two p-values  0.909
```

**Four hundred of the 930 were an artefact of the wrong lookup table**, and the
two p-values rank pairs almost identically (0.91), so nothing above changes
qualitatively — the screen was reading the same evidence, just calling more of
it significant than it was.

Then multiple testing on top, on the corrected p-values:

```
estimated share of pairs with nothing there    73.7%
expected false positives at a flat 5%            182
survive Benjamini-Hochberg at FDR 5%               8
survive Benjamini-Hochberg at FDR 10%             17
survive Benjamini-Hochberg at FDR 20%             41
survive Bonferroni                                 1
```

**Seventeen.** Of 4,950 pairs, seventeen clear a false-discovery rate of 10%.
This project has been trading 930. Bonferroni's answer of one was already in
`meta.json` and is easy to dismiss as too blunt to be useful — it asks whether
*any* of the discoveries might be false. Benjamini-Hochberg asks the question
with money attached: of the pairs I take, what share are noise? A book where
one pair in ten is noise is still a business. A book where 98% of it is noise
is not, and that is what 17-out-of-930 says.

### This corrects a number stated earlier in this document

The quarterly re-estimation section above notes that 651 to 1,555 pairs pass
per quarter "against the ~248 that pure chance predicts", and reads the excess
as evidence of genuine common structure in the universe. That 248 is
4,950 × 5% — the nominal size. The measured size is 15.3%, so **chance predicts
about 757 passes per quarter, not 248**, and the median quarter's 799 is barely
above it.

The claim that there is real shared structure among 100 large-cap US stocks is
still true — they share sectors, factors and a market, and nobody needs a
cointegration test to know that. What is not true is that the pass counts were
evidence of it. They were mostly the test's own mis-sizing, and the paragraph
above overstated the case by roughly a factor of three. It is left standing
with this correction attached rather than quietly edited, because the mistake
is the more useful artefact: the nominal size of a test is an assumption, and
this one had never been checked against the procedure that was actually run.

### And fixing the statistics does not rescue the strategy

The tempting conclusion is that the screen was reading the wrong table and the
seventeen survivors are the real pairs. They are not. Same trading window, same
rules, same costs:

```
group                             pairs  mean sharpe  book sharpe
survive BH at 10%                    17        -0.03        -0.19
passed the old screen, not BH       913        -0.01        -0.19
rejected by the old screen        4,020         0.01        -0.10
```

**The seventeen best-evidenced pairs in the universe trade no better than the
4,020 the screen threw away.** This is the control-group result again, at the
other end of the evidence scale: the earlier section showed the screen cannot
rank within the pairs it kept, and this shows that making the screen
statistically correct — and then demanding far more of it — still selects
nothing that trades.

Which is the cleanest statement of what this project found. The screen had two
separate problems: it was mis-sized, so it kept three times as many pairs as
its own threshold implied; and the property it tests for, even measured
correctly, does not survive into the next five years at a strength worth
trading. Fixing the first does not touch the second. That is worth knowing
before building anything on top of a cointegration screen, and it is not what
the tutorials say.

### What would be next, and why it isn't more refitting

Having ruled out the estimation schedule, and now the screen's own statistics,
the remaining candidates are the data and the universe rather than the method:
point-in-time index membership (the survivorship caveat below is real and
unaddressed), intraday rather than daily bars, and a universe chosen by
economic relationship rather than by running every pair through a test. All
three are ways of making the *screen* unnecessary. That is probably the actual
lesson of this project, and the size measurement above is the sharpest version
of it: the screen was not merely unhelpful, it was miscalibrated by a factor of
three, and correcting it changed nothing about what trades.

## Stretch goals

- **Kalman-filter hedge ratio** - done, see above and
  `src/pairs.ipynb :: kalman_hedge_ratio`. Implemented from the update
  equations (not `pykalman`): a single-state filter tracking `beta_t`
  through-origin, matching this project's own `compute_spread = y - beta *
  x` convention (no separately-floating intercept - see the function's
  docstring for why that would actually be a worse model here, not just a
  simpler one). 6 new tests in `tests/test_kalman.py`.
- ~~**Use `half_life()` to set the z-score window per pair**~~ - done, see
  the walk-forward section above. The five-year return comes out at -0.02%,
  which is the answer but not the interesting part: the interesting part is
  that nothing else moved it either.
- ~~**Walk-forward analysis**~~ - done, `walkforward.py`, 20 quarterly
  re-estimates over 99,000 cointegration tests. Refitting the hedge ratio,
  re-running the screen, tightening the screen and resizing the z-score
  window all land between -1.27% and +0.08% over five years. The finding that
  came out of it: **0 of 4,950 pairs pass the cointegration test in all 20
  quarters**, and the median pair that ever passes, passes in 4 of 20.
- ~~**A control group**~~ - done, `control.py`. Backtests the 4,020 rejected
  pairs under identical rules: they come out marginally ahead of the 930
  survivors (mean Sharpe +0.009 vs -0.009), at p = 0.21 on a permutation
  test, and the p-value/Sharpe rank correlation over all 4,950 pairs is
  -0.001. 7 tests in `tests/test_control.py`.
- ~~**Portfolio of pairs**~~ - done, see "Trading all of them at once"
  above and `portfolio.py`. The equal-weight book of all 930 returns
  -2.14% at a Sharpe of -0.19, 48.8% of pairs make money, and the
  formation p-value's rank correlation with out-of-sample Sharpe is
  -0.05. 17 tests in `tests/test_portfolio.py`.

## Survivorship, measured and bounded instead of admitted

Every page of this project has carried the same warning: the universe is the S&P 100 **as it
stands today**, membership is awarded for having already grown large, and firms that were in the
index and then collapsed or were bought are simply absent. The warning always ended by saying the
fix needs a point-in-time constituent list, which free data does not provide.

That is true - Yahoo returns nothing for MON, TWX, CELG, RTN, ATVI or any of the thirty-odd other
large caps that left the index in this window; delisted tickers are gone, and the ones that still
resolve have been recycled onto different companies. It is also not a reason to leave the size of
the bias unstated, because an unmeasured caveat is indistinguishable from a large one.
`survivorship.py` does what can be done without the data that does not exist, in two directions.

### What the selection actually looks like

```
  median surviving name's formation return        96%
  mean                                            200%
  SPY over the same window                       100%
  share of the universe that beat it              48%
  worst / best                                   -52% / 3636%
```

Not what I expected to find, and worth stating plainly: **the median surviving name roughly
matched SPY, and fewer than half of them beat it.** That comparison is close to circular anyway -
SPY is cap-weighted and largely made of these same companies. The mean is 200% because of the
right tail; one name returned 3,636%.

The tell is the **minimum**. The worst ten-year outcome in this list is GE at -52%. A genuine 2013
large-cap universe followed to 2025 contains companies that went to zero, were acquired at a
premium, or shrank out of the index entirely, and none of those is here. Survivorship bias in this
universe is not mainly a story about the average being too high. It is a story about the left tail
being absent, which matters because the left tail is the only part of the distribution a
market-neutral strategy has no defence against.

### Does the screen pass co-winners more often? Yes, by a third

Two stocks that both tripled between 2013 and 2020 share a strong upward trend, and a common trend
is exactly what makes an ADF test on a fitted residual reject when it should not - which the
screen-validity section already showed it does, three times too often. So: bucket all 4,950 pairs
by formation-window return, each pair taking its *weaker* leg's quartile, so "top quartile" means
both legs were top-quartile winners.

```
both legs at least     pairs  pass rate  median p  mean formation ret
0%-25%                 2,175      17.9%     0.276                -1%
25%-50%                1,550      18.6%     0.197                73%
50%-75%                  925      19.5%     0.227               146%
75%-100%                 300      24.0%     0.199               583%
```

Monotone, and a 34% relative increase from bottom bucket to top. Some of the 930 survivors is
selection rather than structure, and this is how much.

Then the part that makes it matter:

```
bucket                 pairs  mean sharpe   median  profitable
quartile 1               389       -0.091   -0.081       34.4%
quartile 2               289        0.063    0.036       48.8%
quartile 3               180        0.082    0.075       52.8%
quartile 4                72       -0.084   -0.146       26.4%
```

**The bucket the screen likes most is the one that trades worst.** Top-quartile co-winners pass at
24% and are profitable out of sample 26% of the time, against 53% for the third quartile. Passing
because you both went up is not the same as passing because you are tethered, and the out-of-sample
window is where the difference shows up.

### What a leg being acquired would cost, since none ever is

The event a survivors-only universe can never contain is a takeover. A target gaps to near the
offer price in one print, stops moving, then delists - so a cointegrated spread jumps and never
mean-reverts, and a strategy that makes small money on reversion has nothing that caps a gap.

There is no way to observe those events in this universe. There is a way to inject them at a stated
rate and measure what they cost. Each trial draws a Poisson number of acquisitions across the 100
names, gaps the target, holds it flat for five months, then delists it.

Three things had to be got right or the measurement flatters itself, and they are worth listing
because the naive version of this table says takeovers are *profitable*:

- **The window is matched.** A takeover truncates its leg's history; this book loses money, so an
  affected pair trading fewer days looks better for a reason that has nothing to do with the
  merger. Every delta scores the real prices over exactly the dates the modified history had.
- **Pairs killed outright are excluded from the deltas.** A pair that cannot trade earns zero, and
  zero beats this book's average pair. Counted separately.
- **Untouched pairs must be unchanged.** Otherwise the comparison is measuring the random number
  generator. Tested in `tests/test_survivorship.py`.

```
 premium  deals/yr  deals  % of book   median     mean   5th pct     worst  lost >10%
    20%      0.5%    1.7         3%    0.80%    0.41%   -14.90%   -30.37%        15%
    20%      1.0%    4.9         9%    0.64%    1.23%   -15.10%   -39.34%        15%
    20%      2.0%    9.4        17%    0.67%    0.74%   -15.17%   -47.68%        12%
    20%      4.0%   17.7        31%    0.16%    0.44%   -16.48%   -44.36%        16%
    30%      0.5%    1.7         3%    0.17%    0.22%   -19.01%   -35.59%        23%
    30%      1.0%    4.9         9%    0.84%    1.27%   -20.21%   -48.06%        22%
    30%      2.0%    9.4        17%    0.69%    0.94%   -18.39%   -48.72%        18%
    30%      4.0%   17.7        31%    0.22%    0.62%   -20.58%   -49.63%        23%
```

Read the median against the 5th percentile, because they say different things. **A takeover is not
a drag.** The strategy is short one leg and long the other and flips between them, so a gap is
about as likely to land your way as against you, and the median affected pair moves less than a
percent. What it is is a **tail**: the bottom twentieth of affected pairs lose 15-21% of their
value, the worst single pair loses about half, and between 12% and 23% of affected pairs lose more
than 10%. At a plausible 1-2% annual deal rate, 9-17% of this book holds a name that gets acquired
somewhere in the five years.

The rate and the premium are external estimates - roughly 1-2 of a hundred large caps a year
leaving for a deal, at a 20-30% premium - and the grid exists so the conclusion does not rest on
one cell. It does not: every row says the same thing about the shape.

**For a book whose per-pair edge is already indistinguishable from zero, a risk shaped like that is
the whole story.** The strategy has no source of return large enough to pay for a left tail it
cannot see, and this universe cannot contain a single instance of one.

### What this still does not fix

Nothing here is a point-in-time universe. The 100 names are still the 2025 list, the screen is
still run on ten years of their history, and the injected deals land on names chosen at random
rather than on the ones that would really have been bought. What has changed is that the caveat now
has numbers next to it in both directions - how much of the pass rate is selection, and how big the
unobservable tail is - instead of a sentence saying it exists.

## Caveats (unchanged by any of the above)

- **Survivorship bias**: the universe is the S&P 100 as it stands today, so
  the scan asks how today's winners behaved on their way to winning. Still
  true, and now measured rather than only admitted - see the survivorship
  section above for how much of the pass rate is selection (a third, from
  bottom quartile to top) and how large the unobservable takeover tail is
  (the bottom twentieth of affected pairs lose 15-21%).
- Prices are Yahoo Finance `Close` (auto-adjusted) - fine for research,
  not for anything live.
- Transaction costs are modeled as a flat 5bps per side; nothing here is
  investment advice.
