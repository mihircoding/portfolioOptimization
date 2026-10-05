"""
Quick sanity checks for the pricing math. Run with:  python test_sanity.py

These aren't exhaustive unit tests, just the classic identities every
Black-Scholes implementation must satisfy. If any of these break, the
model is wrong.
"""

import numpy as np

import binomial as bino
import black_scholes as bs
import monte_carlo as mc
import strategies as strat

S, K, T, r, sigma = 100.0, 100.0, 1.0, 0.05, 0.20

failures = []


def check(name, ok):
    print(f"{'PASS' if ok else 'FAIL'}  {name}")
    if not ok:
        failures.append(name)


# 1. Known textbook value: S=K=100, T=1, r=5%, vol=20%
#    call = 10.4506, put = 5.5735 (quoted in most derivatives textbooks)
c = float(bs.call_price(S, K, T, r, sigma))
p = float(bs.put_price(S, K, T, r, sigma))
check(f"textbook call 10.4506 (got {c:.4f})", abs(c - 10.4506) < 1e-3)
check(f"textbook put   5.5735 (got {p:.4f})", abs(p - 5.5735) < 1e-3)

# 2. Put-call parity: C - P must equal S - K*e^(-rT), always.
parity = c - p - (S - K * np.exp(-r * T))
check(f"put-call parity (residual {parity:.2e})", abs(parity) < 1e-9)

# 3. Greeks vs numerical differentiation (bump-and-reprice)
eps = 1e-4
num_delta = (bs.call_price(S + eps, K, T, r, sigma)
             - bs.call_price(S - eps, K, T, r, sigma)) / (2 * eps)
check("call delta matches numerical derivative",
      abs(float(bs.delta('call', S, K, T, r, sigma)) - float(num_delta)) < 1e-5)

num_vega = (bs.call_price(S, K, T, r, sigma + eps)
            - bs.call_price(S, K, T, r, sigma - eps)) / (2 * eps) / 100
check("vega matches numerical derivative",
      abs(float(bs.vega(S, K, T, r, sigma)) - float(num_vega)) < 1e-5)

num_theta = (bs.call_price(S, K, T - eps, r, sigma)
             - bs.call_price(S, K, T + eps, r, sigma)) / (2 * eps) / 365
check("theta matches numerical derivative",
      abs(float(bs.theta('call', S, K, T, r, sigma)) - float(num_theta)) < 1e-5)

num_rho = (bs.call_price(S, K, T, r + eps, sigma)
           - bs.call_price(S, K, T, r - eps, sigma)) / (2 * eps) / 100
check("rho matches numerical derivative",
      abs(float(bs.rho('call', S, K, T, r, sigma)) - float(num_rho)) < 1e-5)

# 4. Implied vol round-trip: price at 20% vol, invert, get 20% back
iv = bs.implied_vol("call", c, S, K, T, r)
check(f"implied vol round-trip (got {iv:.4%})", abs(iv - sigma) < 1e-4)

# 5. Deep in/out of the money limits
check("deep ITM call ~ S - K*e^(-rT)",
      abs(float(bs.call_price(200, 100, T, r, sigma))
          - (200 - 100 * np.exp(-r * T))) < 0.01)
check("deep OTM call ~ 0",
      float(bs.call_price(20, 100, T, r, sigma)) < 1e-6)

# 6. Strategy payoffs at expiry make sense
grid = np.linspace(50, 150, 101)
for name in strat.STRATEGY_NAMES:
    legs, _ = strat.build_strategy(name, S, K)
    payoff = strat.strategy_payoff(legs, grid)
    cost = strat.strategy_cost(legs, S, T, r, sigma)
    check(f"{name}: finite cost/payoff",
          np.isfinite(cost) and np.all(np.isfinite(payoff)))

# straddle payoff at strike should be exactly 0 (both legs expire worthless)
legs, _ = strat.build_strategy("Long Straddle", S, K)
check("straddle payoff at strike = 0",
      abs(float(strat.strategy_payoff(legs, np.array([K]))[0])) < 1e-9)

# iron condor opens for a credit (you get paid to take the trade)
legs, _ = strat.build_strategy("Iron Condor", S, K)
check("iron condor opens for a credit",
      strat.strategy_cost(legs, S, T, r, sigma) < 0)

# strategy_value at T=0 must equal the expiry payoff
legs, _ = strat.build_strategy("Bull Call Spread", S, K)
v0 = strat.strategy_value(legs, grid, 0.0, r, sigma)
pay = strat.strategy_payoff(legs, grid)
check("strategy value at T=0 equals expiry payoff",
      np.allclose(v0, pay, atol=1e-4))

# 7. Dividend-adjusted model (q > 0)
q = 0.03
cq = float(bs.call_price(S, K, T, r, sigma, q))
pq = float(bs.put_price(S, K, T, r, sigma, q))
# put-call parity with dividends: C - P = S*e^(-qT) - K*e^(-rT)
parity_q = cq - pq - (S * np.exp(-q * T) - K * np.exp(-r * T))
check(f"dividend put-call parity (residual {parity_q:.2e})",
      abs(parity_q) < 1e-9)
check("dividends make calls cheaper and puts dearer", cq < c and pq > p)

num_delta_q = (bs.call_price(S + eps, K, T, r, sigma, q)
               - bs.call_price(S - eps, K, T, r, sigma, q)) / (2 * eps)
check("dividend delta matches numerical derivative",
      abs(float(bs.delta('call', S, K, T, r, sigma, q))
          - float(num_delta_q)) < 1e-5)

num_theta_q = (bs.call_price(S, K, T - eps, r, sigma, q)
               - bs.call_price(S, K, T + eps, r, sigma, q)) / (2 * eps) / 365
check("dividend theta matches numerical derivative",
      abs(float(bs.theta('call', S, K, T, r, sigma, q))
          - float(num_theta_q)) < 1e-5)

iv_q = bs.implied_vol("call", cq, S, K, T, r, q)
check(f"dividend implied vol round-trip (got {iv_q:.4%})",
      abs(iv_q - sigma) < 1e-4)

# 8. Monte Carlo agrees with the closed form (within 4 standard errors -
#    a ~1 in 16,000 false-failure rate at 500k paths)
for opt in ("call", "put"):
    mc_val, mc_err = mc.mc_price(opt, S, K, T, r, sigma, 500_000, seed=0)
    bs_val = float(bs.price(opt, S, K, T, r, sigma))
    check(f"MC {opt} {mc_val:.4f} within 4 SE of BS {bs_val:.4f}",
          abs(mc_val - bs_val) < 4 * mc_err)

mc_q, mc_q_err = mc.mc_price("call", S, K, T, r, sigma, 500_000, seed=0, q=q)
check("MC with dividends matches dividend-adjusted BS",
      abs(mc_q - cq) < 4 * mc_q_err)

# 9. Binomial tree (American/European) - two checks that don't depend on
#    each other agreeing by construction, since they easily could if I'd
#    made a copy-paste mistake.
#
#    a) European binomial -> Black-Scholes as steps grows. Different model
#       (discrete tree vs. closed-form integral), same no-arbitrage
#       argument, should land in the same place.
euro_tree = bino.crr_price("call", "european", S, K, T, r, sigma, steps=500)
check(f"European binomial -> Black-Scholes ({euro_tree:.4f} vs {c:.4f})",
      abs(euro_tree - c) < 0.02)

#    b) No dividend -> American call should never be worth exercising early
#       (a well-known result: you'd throw away remaining time value for
#       nothing, since there's no dividend to capture). American and
#       European calls should price identically when q=0.
prem_call_no_div = bino.early_exercise_premium("call", S, K, T, r, sigma, q=0.0, steps=300)
check(f"American call = European call when q=0 (premium {prem_call_no_div:.2e})",
      abs(prem_call_no_div) < 1e-6)

#    c) Puts are different: even with no dividend, it can be worth
#       exercising a deep ITM put early to start earning interest on the
#       strike now instead of waiting. Premium should be strictly positive
#       for a put that's meaningfully in the money.
prem_put_itm = bino.early_exercise_premium("put", 70.0, 100.0, T, r, sigma, q=0.0, steps=300)
check(f"American put has positive early-exercise premium when deep ITM "
      f"({prem_put_itm:.4f})", prem_put_itm > 0.01)

#    d) With a dividend, American calls CAN be worth exercising early too
#       (to capture the dividend before the stock drops on ex-div date) -
#       premium should turn positive once q > 0.
prem_call_div = bino.early_exercise_premium("call", S, K, T, r, sigma, q=0.05, steps=300)
check(f"American call premium turns positive with a dividend ({prem_call_div:.4f})",
      prem_call_div > 0.0)

# 10. Monte Carlo Greeks (pathwise delta/vega, likelihood-ratio gamma)
#     agree with the closed-form Black-Scholes Greeks, within 4 standard
#     errors - same bar as the MC price check above (#8), and the same
#     reason: these are estimates with real sampling noise, so "matches
#     within its own reported uncertainty" is the honest check, not
#     "matches exactly."
for opt in ("call", "put"):
    d_bs = float(bs.delta(opt, S, K, T, r, sigma))
    v_bs = float(bs.vega(S, K, T, r, sigma))
    g_bs = float(bs.gamma(S, K, T, r, sigma))

    d_mc, d_se = mc.pathwise_delta(opt, S, K, T, r, sigma, 500_000, seed=0)
    v_mc, v_se = mc.pathwise_vega(opt, S, K, T, r, sigma, 500_000, seed=0)
    g_mc, g_se = mc.likelihood_ratio_gamma(opt, S, K, T, r, sigma, 500_000, seed=0)

    check(f"MC pathwise {opt} delta {d_mc:.4f} within 4 SE of BS {d_bs:.4f}",
          abs(d_mc - d_bs) < 4 * d_se)
    check(f"MC pathwise {opt} vega {v_mc:.4f} within 4 SE of BS {v_bs:.4f}",
          abs(v_mc - v_bs) < 4 * v_se)
    check(f"MC likelihood-ratio {opt} gamma {g_mc:.4f} within 4 SE of BS {g_bs:.4f}",
          abs(g_mc - g_bs) < 4 * g_se)

# same three, with a dividend - checks that q flows correctly into all
# three estimators, not just into mc_price
d_bs_q = float(bs.delta("call", S, K, T, r, sigma, q))
v_bs_q = float(bs.vega(S, K, T, r, sigma, q))
g_bs_q = float(bs.gamma(S, K, T, r, sigma, q))
d_mc_q, d_se_q = mc.pathwise_delta("call", S, K, T, r, sigma, 500_000, seed=0, q=q)
v_mc_q, v_se_q = mc.pathwise_vega("call", S, K, T, r, sigma, 500_000, seed=0, q=q)
g_mc_q, g_se_q = mc.likelihood_ratio_gamma("call", S, K, T, r, sigma, 500_000, seed=0, q=q)
check("MC dividend delta matches dividend-adjusted BS delta",
      abs(d_mc_q - d_bs_q) < 4 * d_se_q)
check("MC dividend vega matches dividend-adjusted BS vega",
      abs(v_mc_q - v_bs_q) < 4 * v_se_q)
check("MC dividend gamma matches dividend-adjusted BS gamma",
      abs(g_mc_q - g_bs_q) < 4 * g_se_q)

# _implied_z must actually be the inverse of terminal_prices' own formula -
# if this ever drifted out of sync, every Greek above would be silently
# wrong in a way none of the "matches BS" checks would clearly point to.
s_t_check = mc.terminal_prices(S, T, r, sigma, 10_000, seed=1)
import numpy as _np
z_recovered = mc._implied_z(s_t_check, S, T, r, sigma)
s_t_rebuilt = S * _np.exp((r - 0.5 * sigma**2) * T + sigma * _np.sqrt(T) * z_recovered)
check("_implied_z round-trips through terminal_prices' own formula",
      _np.allclose(s_t_rebuilt, s_t_check, rtol=1e-9))


# 11. The implied-vol solver, over the whole range instead of one point
#     Check 6 above does a single round trip at the money, which every
#     inversion method passes. These are the cases that separate them:
#     far from the money, very short dated, very long dated.
print()
rng = np.random.default_rng(4)
worst_err, solved, refused = 0.0, 0, 0
for _ in range(400):
    k = float(rng.uniform(60, 160))
    t = float(rng.uniform(0.02, 2.0))
    qq = float(rng.uniform(0.0, 0.04))
    vol = float(rng.uniform(0.05, 1.2))
    kind = "call" if rng.random() < 0.5 else "put"
    px = float(bs.price(kind, S, k, t, r, vol, qq))
    got = bs.implied_vol(kind, px, S, k, t, r, qq)
    if np.isfinite(got):
        solved += 1
        worst_err = max(worst_err, abs(got - vol))
    else:
        refused += 1
check(f"implied vol recovers 400 random inputs (worst error {worst_err:.2e}, "
      f"{refused} refused as vega-less)", worst_err < 1e-6 and solved > 300)

# A quote below intrinsic value has no implied volatility. Returning some
# number anyway is how a bad print ends up plotted as a spike on a surface.
below_intrinsic = float(S - K * np.exp(-r * T)) - 1.0
check("implied vol returns nan for a price below intrinsic",
      np.isnan(bs.implied_vol("call", below_intrinsic, S, K, T, r)))
check("implied vol returns nan for a price above the underlying",
      np.isnan(bs.implied_vol("call", S * 1.1, S, K, T, r)))

# A deep ITM call is worth intrinsic-plus-epsilon at 5% vol and at 40% vol
# alike: vega is ~0, so no volatility is implied. The old bisection solver
# returned the top of its search range here, with no way to tell.
deep_itm = float(bs.price("call", S, 40.0, 0.25, r, 0.05))
check("implied vol refuses a quote with no vega instead of guessing",
      np.isnan(bs.implied_vol("call", deep_itm, S, 40.0, 0.25, r)))

# Newton must not be slower than the bisection it replaced. Counted in
# pricing calls rather than seconds, so the check means the same thing on
# any machine.
_price = bs.price
_n_calls = [0]


def _counting_price(*a, **kw):
    _n_calls[0] += 1
    return _price(*a, **kw)


bs.price = _counting_price
_n_calls[0] = 0
for _ in range(100):
    k = float(rng.uniform(80, 130))
    t = float(rng.uniform(0.05, 1.5))
    vol = float(rng.uniform(0.1, 0.8))
    bs.implied_vol("call", float(_price("call", S, k, t, r, vol)), S, k, t, r)
calls_per_solve = _n_calls[0] / 100
bs.price = _price
check(f"implied vol converges in {calls_per_solve:.1f} pricing calls "
      f"(bisection needs ~31 for the same tolerance)", calls_per_solve < 15)


# 12. Volatility surface construction, on a chain built from a known smile
#     Real chains can't be an assertion - the answer isn't known. So build
#     a synthetic one: pick a forward, a discount factor and a smile, price
#     every strike with them, and check the module recovers all three from
#     nothing but the prices.
print()

# 12. Delta hedging. These are the checks that tie the price to something you
#     can actually do: run the model's own replication strategy on a path and
#     see whether the money works out. hedging.py needs nothing but numpy, so
#     unlike the surface checks below, these always run in CI.
import hedging as hedge


def gbm_paths(n_paths, n_steps, S0, T, sigma, seed):
    """GBM paths under the risk-neutral measure (zero drift, since r = 0 in
    these checks). Returned including S0, so each row has n_steps + 1 points."""
    rng = np.random.default_rng(seed)
    dt = T / n_steps
    shocks = rng.normal(0.0, 1.0, size=(n_paths, n_steps))
    log_path = np.cumsum(-0.5 * sigma ** 2 * dt + sigma * np.sqrt(dt) * shocks, axis=1)
    return S0 * np.concatenate([np.ones((n_paths, 1)), np.exp(log_path)], axis=1)


T_h, sigma_h = 30 / 252, 0.20

# A path that never moves: every rebalance trades at the same price, so the
# round trip nets to zero and the seller keeps the entire premium. Any carry
# or accounting error in the loop shows up here as a residual.
flat = np.full(31, 100.0)
flat_result = hedge.delta_hedge(flat, 100.0, T_h, 0.0, sigma_h)
check(f"a motionless stock pays the whole premium "
      f"(P&L {flat_result['pnl']:.6f} vs premium {flat_result['premium']:.6f})",
      abs(flat_result["pnl"] - flat_result["premium"]) < 1e-9)

# Put-call parity, restated as a hedging claim. With r = q = 0 a call's delta
# exceeds a put's by exactly 1, so hedging the two differs by a single share
# held statically - which is riskless. The two hedges must therefore end at
# the same P&L on every path, and this is the cleanest possible test that the
# delta and the payoff are consistent with each other.
parity_path = gbm_paths(1, 30, 100.0, T_h, 0.25, seed=11)[0]
call_hedge = hedge.delta_hedge(parity_path, 100.0, T_h, 0.0, sigma_h, "call")
put_hedge = hedge.delta_hedge(parity_path, 100.0, T_h, 0.0, sigma_h, "put")
check(f"hedged call and hedged put earn the same P&L "
      f"(difference {call_hedge['pnl'] - put_hedge['pnl']:.2e})",
      abs(call_hedge["pnl"] - put_hedge["pnl"]) < 1e-9)

# Sell at 30% vol into a stock that only realizes 10%: the seller should win on
# essentially every path, because the hedge is being paid for movement that
# never arrives. Reverse the two and it should lose on essentially every path.
# If either direction were mixed, the P&L would not be a variance bet.
cheap = gbm_paths(200, 30, 100.0, T_h, 0.10, seed=3)
rich = gbm_paths(200, 30, 100.0, T_h, 0.40, seed=4)
sold_high = np.array([hedge.delta_hedge(path, 100.0, T_h, 0.0, 0.30)["pnl"]
                      for path in cheap])
sold_low = np.array([hedge.delta_hedge(path, 100.0, T_h, 0.0, 0.20)["pnl"]
                     for path in rich])
check(f"selling vol above what the stock realizes wins "
      f"({float((sold_high > 0).mean()):.0%} of paths)",
      (sold_high > 0).mean() > 0.95)
check(f"selling vol below what the stock realizes loses "
      f"({float((sold_low < 0).mean()):.0%} of paths)",
      (sold_low < 0).mean() > 0.95)

# Hedge at the same vol the path was generated with and the expected P&L is
# zero - that is the replication argument. It holds only on average, so the
# check is on the mean against its own standard error.
fair = gbm_paths(400, 30, 100.0, T_h, sigma_h, seed=5)
fair_pnl = np.array([hedge.delta_hedge(path, 100.0, T_h, 0.0, sigma_h)["pnl"]
                     for path in fair])
standard_error = fair_pnl.std(ddof=1) / np.sqrt(len(fair_pnl))
check(f"hedging at the path's own vol has zero expected P&L "
      f"(mean {fair_pnl.mean():+.4f}, 3 s.e. = {3 * standard_error:.4f})",
      abs(fair_pnl.mean()) < 3 * standard_error)

# The gamma decomposition has to reproduce the simulated hedge, or the story
# "a delta-hedged option is a bet on variance" is just a slogan. It is a
# second-order expansion, so it is checked to a tolerance in premium terms
# rather than exactly.
decomposed = np.array([hedge.gamma_pnl(path, 100.0, T_h, 0.0, sigma_h)[0]
                       for path in fair])
gap = np.abs(decomposed - fair_pnl).mean()
check(f"gamma decomposition reproduces the hedge P&L "
      f"(mean error {gap:.4f} on a {flat_result['premium']:.3f} premium)",
      gap < 0.05 * flat_result["premium"])
check(f"...and tracks it path by path "
      f"(r = {float(np.corrcoef(decomposed, fair_pnl)[0, 1]):.4f})",
      np.corrcoef(decomposed, fair_pnl)[0, 1] > 0.95)

# Boyle-Emanuel: hedging less often leaves the expected P&L alone and widens
# the distribution like sqrt(interval). Differencing against each path's own
# daily hedge removes everything the two frequencies have in common, so what
# is left is the error the coarser hedge introduced.
errors = {}
for interval in (2, 5, 10):
    coarse = np.array([hedge.delta_hedge(path, 100.0, T_h, 0.0, sigma_h,
                                         rebalance_every=interval)["pnl"]
                       for path in fair])
    errors[interval] = coarse - fair_pnl
check(f"hedging error grows with the interval "
      f"({errors[2].std(ddof=1):.3f} -> {errors[10].std(ddof=1):.3f})",
      errors[2].std(ddof=1) < errors[5].std(ddof=1) < errors[10].std(ddof=1))
scaled = {k: v.std(ddof=1) / np.sqrt(k) for k, v in errors.items()}
check(f"...and grows like sqrt(interval), not faster "
      f"(std/sqrt(n): {scaled[2]:.3f}, {scaled[5]:.3f}, {scaled[10]:.3f})",
      max(scaled.values()) / min(scaled.values()) < 1.5)
check("hedging less often does not change the expected P&L",
      all(abs(v.mean()) < 3 * v.std(ddof=1) / np.sqrt(len(v))
          for v in errors.values()))

# Costs are charged on every share traded, so a coarser hedge trades less and
# pays less. Both halves of that are worth asserting: the cost is real, and it
# is proportional to the trading, not to the position.
free = hedge.delta_hedge(fair[0], 100.0, T_h, 0.0, sigma_h, cost_bps=0.0)
paid = hedge.delta_hedge(fair[0], 100.0, T_h, 0.0, sigma_h, cost_bps=10.0)
check(f"transaction costs reduce P&L by exactly what they charge "
      f"({paid['costs']:.4f})",
      abs((free["pnl"] - paid["pnl"]) - paid["costs"]) < 1e-9)

print()

# vol_surface works on dataframes, so this section needs pandas. The CI job
# installs numpy and scipy only (its workflow file needs a token scope this
# repo's automation doesn't have), so rather than fail the build on a
# missing dependency, say plainly that the checks didn't run.
try:
    import pandas as pd

    import vol_surface as vsurf
except ImportError as exc:                                   # pragma: no cover
    vsurf = None
    print(f"SKIP  volatility surface checks - {exc}")

F_true, disc_true, T_s = 105.0, np.exp(-0.04 * 0.5), 0.5
spot_eq = F_true * disc_true
r_true = -np.log(disc_true) / T_s
if vsurf is not None:
    strikes = np.arange(80.0, 131.0, 2.5)


    def _true_iv(k):
        """A downward-sloping smile: 22% at the forward, steeper on the downside."""
        m = np.log(k / F_true)
        return 0.22 - 0.35 * m + 0.60 * m**2


    def _chain(kind):
        prices = np.array([float(bs.price(kind, spot_eq, k, T_s, r_true, _true_iv(k)))
                           for k in strikes])
        return pd.DataFrame({"strike": strikes,
                             "bid": prices * 0.995, "ask": prices * 1.005,
                             "volume": 100.0, "openInterest": 100.0})


    calls_df, puts_df = _chain("call"), _chain("put")
    fwd = vsurf.forward_from_parity(vsurf.clean_quotes(calls_df, "call"),
                                    vsurf.clean_quotes(puts_df, "put"), T_s)
    check(f"put-call parity recovers the forward ({fwd['forward']:.4f} vs {F_true})",
          abs(fwd["forward"] - F_true) < 0.05)
    check(f"put-call parity recovers the discount rate ({fwd['rate']:.4%} vs 4.00%)",
          abs(fwd["rate"] - 0.04) < 0.005)
    check(f"parity fit is a straight line (R^2 = {fwd['r2']:.6f})", fwd["r2"] > 0.9999)

    smile = vsurf.smile_for_expiry(calls_df, puts_df, T_s, "synthetic")
    recovered = np.array([abs(row.iv - _true_iv(row.strike)) for row in smile.itertuples()])
    check(f"surface recovers the smile it was built from (worst error "
          f"{recovered.max():.2e} across {len(smile)} strikes)", recovered.max() < 5e-3)

    atm = vsurf.atm_vol(smile)
    check(f"at-the-money vol interpolates to the smile's own level "
          f"({atm:.4f} vs {_true_iv(F_true):.4f})", abs(atm - _true_iv(F_true)) < 2e-3)

    sk = vsurf.skew_25d(smile)
    check(f"risk reversal is positive for a downward-sloping smile "
          f"({sk['risk_reversal']:.4f})", sk["risk_reversal"] > 0.02)
    check(f"butterfly is positive for a convex smile ({sk['butterfly']:.4f})",
          sk["butterfly"] > 0)

    # Only out-of-the-money quotes may survive: puts below the forward, calls
    # above it. Mixing in the ITM leg would double-count and, worse, import the
    # early-exercise error the OTM restriction exists to avoid.
    wrong_side = smile[((smile["type"] == "put") & (smile["strike"] >= fwd["forward"]))
                       | ((smile["type"] == "call") & (smile["strike"] < fwd["forward"]))]
    check("surface keeps only out-of-the-money quotes", len(wrong_side) == 0)

    # A one-sided or crossed quote is not a price. These are the rows real
    # chains are full of, and every one of them must be dropped before it can
    # become an implied volatility.
    junk = pd.DataFrame({"strike": [100.0, 101.0, 102.0, 103.0],
                         "bid": [0.0, 5.0, -1.0, 2.0],
                         "ask": [1.0, 4.0, 2.0, 12.0],   # row 2 crossed, row 4 too wide
                         "volume": [1, 1, 1, 1], "openInterest": [1, 1, 1, 1]})
    check("quote filter drops zero bids, crossed markets and very wide spreads",
          len(vsurf.clean_quotes(junk, "call")) == 0)

    # Calendar arbitrage: total variance may not fall as maturity rises at a
    # fixed moneyness. Two smiles from the same vol function at different
    # maturities satisfy that by construction, so the check must stay quiet on
    # them - and must fire the moment the far variance is pushed below the near
    # one, because a check that has never fired is an untested check.
    def _dated_chain(kind, t):
        disc = np.exp(-0.04 * t)
        spot = F_true * disc
        px = np.array([float(bs.price(kind, spot, k, t, 0.04, _true_iv(k)))
                       for k in strikes])
        return pd.DataFrame({"strike": strikes, "bid": px * 0.995, "ask": px * 1.005,
                             "volume": 100.0, "openInterest": 100.0})


    near = vsurf.smile_for_expiry(_dated_chain("call", 0.25), _dated_chain("put", 0.25),
                                  0.25, "near")
    far = vsurf.smile_for_expiry(_dated_chain("call", 1.00), _dated_chain("put", 1.00),
                                 1.00, "far")
    check("calendar-arbitrage check passes a well-ordered surface",
          len(vsurf.calendar_arbitrage(pd.concat([near, far], ignore_index=True))) == 0)

    broken = far.copy()
    broken["total_var"] = broken["total_var"] * 0.1   # far below near: free money
    check("calendar-arbitrage check catches falling total variance",
          len(vsurf.calendar_arbitrage(pd.concat([near, broken], ignore_index=True))) > 0)



# ---------------------------------------------------------------------------
# Arbitrage across strikes (arbitrage.py)
#
# The conditions are statements about the call price that follow from the
# payoff, so a chain generated from Black-Scholes must satisfy all of them
# exactly, and a chain with one quote nudged must fail the right one. A check
# that has only ever been run on clean data is not a check.
import arbitrage as arbi


def _cross_strike_checks():

    _bf_strikes = np.arange(60.0, 141.0, 5.0)
    _bf_F, _bf_T, _bf_r = 100.0, 0.5, 0.03
    _bf_disc = np.exp(-_bf_r * _bf_T)


    def _bf_chain(kind, bump=None):
        """A Black-Scholes chain with a 1% spread, optionally with one mid moved."""
        px = np.array([float(bs.price(kind, _bf_F * _bf_disc, k, _bf_T, _bf_r, 0.22))
                       for k in _bf_strikes])
        if bump is not None:
            strike, amount = bump
            px[_bf_strikes == strike] += amount
        return pd.DataFrame({"strike": _bf_strikes, "bid": px * 0.995,
                             "ask": px * 1.005, "volume": 100.0, "openInterest": 100.0})


    _bf_smile = vsurf.smile_for_expiry(_bf_chain("call"), _bf_chain("put"),
                                       _bf_T, "clean")
    _bf_curve = arbi.call_curve(_bf_smile)
    _bf_res = arbi.check_smile(_bf_smile)

    check("call curve spans every quoted strike after parity conversion",
          len(_bf_curve) == len(_bf_smile))
    check("call price falls as strike rises",
          bool((_bf_curve["call"].diff().dropna() < 0).all()))
    check(f"clean chain has no vertical violations "
          f"({_bf_res['vertical_violations']} found)",
          _bf_res["vertical_violations"] == 0)
    check(f"clean chain has no butterfly violations "
          f"({_bf_res['butterfly_violations']} found)",
          _bf_res["butterfly_violations"] == 0)

    # Breeden-Litzenberger: the second derivative of the call price is the
    # risk-neutral density. The quote filter keeps strikes within 40% of the
    # forward, so it integrates to a little under 1 - the missing few percent
    # is the tails the chain does not quote - and its mean is the forward.
    _bf_b = _bf_res["butterflies"]
    _bf_mass = float((_bf_b["density"] * _bf_b["width"]).sum())
    _bf_mean = float((_bf_b["strike"] * _bf_b["density"] * _bf_b["width"]).sum() / _bf_mass)
    check(f"implied density integrates to just under 1 over the quoted strikes "
          f"(got {_bf_mass:.3f})", 0.93 < _bf_mass <= 1.001)
    check(f"implied density is centred on the forward (got {_bf_mean:.1f} "
          f"vs {_bf_F:.1f})", abs(_bf_mean - _bf_F) < 1.0)

    # A body quote $3 too dear makes the butterfly centred on it cost less than
    # nothing: that is the whole point of the check, and it must also be large
    # enough to survive the spread, since a violation inside the spread is not
    # a trade.
    _bent = vsurf.smile_for_expiry(_bf_chain("call", bump=(100.0, 3.0)),
                                   _bf_chain("put", bump=(100.0, 3.0)), _bf_T, "bent")
    _bent_res = arbi.check_smile(_bent)
    check("an overpriced body is caught as a butterfly violation",
          _bent_res["butterfly_violations"] > 0)
    check("and it is big enough to be tradable through the spread",
          _bent_res["butterfly_tradable"] > 0)

    # A violation smaller than the spread must NOT be reported as tradable. Same
    # bad quote, but on a chain quoted 20% wide, where paying to trade the three
    # legs costs an order of magnitude more than the violation is worth.
    def _wide_chain(kind, bump=None):
        px = np.array([float(bs.price(kind, _bf_F * _bf_disc, k, _bf_T, _bf_r, 0.22))
                       for k in _bf_strikes])
        if bump is not None:
            px[_bf_strikes == bump[0]] += bump[1]
        return pd.DataFrame({"strike": _bf_strikes, "bid": px * 0.90,
                             "ask": px * 1.10, "volume": 100.0, "openInterest": 100.0})

    _tiny = vsurf.smile_for_expiry(_wide_chain("call", bump=(100.0, 0.5)),
                                   _wide_chain("put", bump=(100.0, 0.5)), _bf_T, "wide")
    _tiny_res = arbi.check_smile(_tiny)
    check(f"a violation inside the bid-ask spread is not called tradable "
          f"({_tiny_res['butterfly_violations']} found, "
          f"{_tiny_res['butterfly_tradable']} tradable)",
          _tiny_res["butterfly_violations"] > 0 and _tiny_res["butterfly_tradable"] == 0)

    # Unequal strike spacing: the butterfly weights have to account for it, or a
    # ragged strike ladder reads as arbitrage everywhere.
    _ragged_k = np.array([80.0, 90.0, 95.0, 100.0, 110.0, 130.0])
    _ragged = pd.DataFrame({
        "strike": _ragged_k,
        "call": [float(bs.price("call", _bf_F * _bf_disc, k, _bf_T, _bf_r, 0.22))
                 for k in _ragged_k],
        "half_spread": np.zeros(len(_ragged_k)),
    })
    check("unevenly spaced strikes do not read as arbitrage",
          bool((arbi.butterfly_checks(_ragged, _bf_r, _bf_T)["butterfly"] > 0).all()))

    # The slope bound: a call spread cannot cost more than the strikes it can pay.
    _steep = pd.DataFrame({"strike": [90.0, 100.0], "call": [30.0, 5.0],
                           "half_spread": [0.0, 0.0]})
    check("a call spread priced above its maximum payoff is caught",
          bool(arbi.vertical_checks(_steep, _bf_disc)["violation"].all()))


if vsurf is not None:
    _cross_strike_checks()
else:  # pragma: no cover
    print('SKIP  cross-strike arbitrage checks - pandas unavailable')


def _raises(fn, exc) -> bool:
    try:
        fn()
    except exc:
        return True
    except Exception:
        return False
    return False


# ---------------------------------------------------------------------------
# SVI smile fit. Pure numpy, so unlike the surface checks above these always
# run. The properties worth pinning are the ones the module exists for: that
# the fit recovers a smile it was given, that it refuses to return a smile
# with a negative density, and that g(k) is a formula in the parameters
# rather than something measured off a grid of prices afterwards.
# ---------------------------------------------------------------------------
import svi as _svi

_truth = _svi.SVIParams(a=0.040, b=0.40, rho=-0.45, m=-0.02, sigma=0.12)
_k = np.linspace(-0.45, 0.35, 31)
_w = _truth.total_variance(_k)
_fit = _svi.fit_svi(_k, _w)
_p = _fit["params"]

check("SVI recovers the parameters of a smile it generated",
      abs(_p.a - _truth.a) < 1e-3 and abs(_p.b - _truth.b) < 1e-2
      and abs(_p.rho - _truth.rho) < 1e-2)
check("SVI fit error on an exact SVI smile is numerically zero",
      _fit["rmse_total_var"] < 1e-6)
check("a fitted smile passes its own butterfly test",
      _fit["butterfly_free"] and _fit["min_g"] > 0)

# g(k) against a finite-difference version of the same expression. If these
# disagree, the closed form has a typo - which would be invisible in every
# other check here, because a wrong g would still be smooth and positive.
_h = 1e-5
_kk = np.linspace(-0.6, 0.6, 41)
_w0 = _truth.total_variance(_kk)
_w1 = (_truth.total_variance(_kk + _h) - _truth.total_variance(_kk - _h)) / (2 * _h)
_w2 = (_truth.total_variance(_kk + _h) - 2 * _w0
       + _truth.total_variance(_kk - _h)) / _h ** 2
_g_numeric = ((1 - _kk * _w1 / (2 * _w0)) ** 2
              - (_w1 / 4) * (1 / _w0 + 0.25) * _w1 + _w2 / 2)
check("Durrleman g(k) matches a finite-difference evaluation",
      bool(np.max(np.abs(_svi.durrleman_g(_truth, _kk) - _g_numeric)) < 1e-6))

# A smile that really does admit butterfly arbitrage must be reported as
# such. b large with |rho| near 1 makes the wing slope blow past Lee's bound.
_bad = _svi.SVIParams(a=0.01, b=1.8, rho=-0.95, m=0.0, sigma=0.02)
_bad_checks = _svi.is_arbitrage_free(_bad)
check("an arbitrable smile is reported as arbitrable",
      not _bad_checks["butterfly_free"] and not _bad_checks["lee_bound_ok"])

check("total variance is positive across the fitted range",
      bool(np.all(_p.total_variance(np.linspace(-1.5, 1.5, 101)) > 0)))
check("wing slopes are b(1 -/+ rho), as Lee's bound is stated",
      abs(_p.left_slope - _p.b * (1 - _p.rho)) < 1e-12
      and abs(_p.right_slope - _p.b * (1 + _p.rho)) < 1e-12)

# Noisy quotes, the realistic case: the fit should stay arbitrage-free even
# when the data it is handed is not.
_rng = np.random.default_rng(7)
_noisy = _truth.total_variance(_k) * (1 + _rng.normal(0, 0.03, len(_k)))
_nfit = _svi.fit_svi(_k, _noisy)
check("a fit to non-convex noisy quotes is still butterfly-free",
      _nfit["butterfly_free"])
check("SVI needs at least as many quotes as parameters",
      _raises(lambda: _svi.fit_svi(_k[:4], _w[:4]), ValueError))


# ---------------------------------------------------------------------------
# SSVI: the whole surface under both no-arbitrage conditions at once.
#
# The important checks here are the ones that tie the theory to the code. The
# conditions in the paper are statements about (rho, eta, gamma); what a user
# cares about is whether the fitted SURFACE has negative densities or negative
# calendar spreads in it. Those are different claims, and a bug in the
# conversion between them would be invisible to either one alone - so both are
# checked, and the fact that they agree is the real test.
# ---------------------------------------------------------------------------
import ssvi as _ssvi

_ss = _ssvi.SSVIParams(rho=-0.5, eta=0.8, gamma=0.4,
                       thetas=(0.002, 0.01, 0.04), ts=(0.08, 0.5, 2.0))

# theta IS the at-the-money total variance: put k=0 in the SSVI formula and the
# braces collapse to 2. If this drifts, every term structure the model reports
# is mislabelled.
check("theta is at-the-money total variance",
      all(abs(float(_ss.total_variance(0.0, th)) - th) < 1e-12
          for th in _ss.thetas))

# Every SSVI slice is a raw SVI slice. The conversion is what lets svi.py's
# density check be pointed at this surface instead of being rewritten.
_kgrid = np.linspace(-1.2, 1.2, 241)
check("as_svi_slice reproduces the SSVI slice exactly",
      all(float(np.max(np.abs(_ss.total_variance(_kgrid, th)
                              - _ss.as_svi_slice(th).total_variance(_kgrid))))
          < 1e-10 for th in _ss.thetas))

check("the converted slice carries the surface's own rho",
      all(_ss.as_svi_slice(th).rho == _ss.rho for th in _ss.thetas))

# Butterfly and calendar conditions have to reject as well as accept, or they
# are decoration. eta large pushes theta*phi*(1+|rho|) past 4.
check("the butterfly condition accepts a sane parameter set",
      _ssvi.butterfly_ok(-0.5, 0.8, 0.4))
check("the butterfly condition rejects eta far too large",
      not _ssvi.butterfly_ok(-0.5, 40.0, 0.05))
check("admissible() refuses rho outside (-1, 1)",
      not _ssvi.admissible(-1.2, 0.8, 0.4) and not _ssvi.admissible(1.0, 0.8, 0.4))
check("admissible() refuses a non-positive eta",
      not _ssvi.admissible(-0.5, 0.0, 0.4))
check("admissible() refuses gamma outside (0, 1)",
      not _ssvi.admissible(-0.5, 0.8, 1.4) and not _ssvi.admissible(-0.5, 0.8, 0.0))

check("an admissible surface has no negative calendar spread anywhere",
      _ssvi.calendar_violations(_ss) == 0)
check("an admissible surface has no negative density anywhere",
      _ssvi.density_violations(_ss) == 0)

# The calendar condition is free for the power law, and the derivation says why:
# d(theta*phi)/dtheta / phi collapses to (1-gamma)/(1+theta), bounded above by
# 1-gamma < 1, while the bound (1+sqrt(1-rho^2))/rho^2 never falls below 1.
# Checking the identity is checking the proof; checking that calendar_ok never
# fires is checking the conclusion. Both, because either alone would hide a
# transcription error.
_tg = np.geomspace(1e-4, 3.0, 4_000)
for _gamma in (0.05, 0.3, 0.5, 0.85):
    _phi = 1.0 / (_tg ** _gamma * (1 + _tg) ** (1 - _gamma))
    _ratio = np.gradient(_tg * _phi, _tg) / _phi
    check(f"d(theta*phi)/dtheta / phi is (1-gamma)/(1+theta) at gamma={_gamma}",
          float(np.max(np.abs(_ratio - (1 - _gamma) / (1 + _tg)))) < 5e-3)
check("that ratio is always below the bound, so calendar_ok cannot bind",
      max((1 - g) for g in (0.02, 0.5, 0.98)) < min(
          (1 + np.sqrt(1 - r ** 2)) / r ** 2
          for r in np.linspace(-0.999, 0.999, 401) if abs(r) > 1e-6))
check("calendar_ok accepts every admissible power-law parameter set",
      all(_ssvi.calendar_ok(r, e, g)
          for r in (-0.95, -0.4, 0.0, 0.6) for e in (0.1, 1.0, 12.0)
          for g in (0.05, 0.5, 0.9)))

# Which leaves theta monotone as the half that actually does the work, so it has
# to be the half that gets tested. A surface whose at-the-money variance falls
# with maturity prices a negative calendar spread at every strike.
_backwards = _ssvi.SSVIParams(rho=-0.5, eta=0.8, gamma=0.4,
                              thetas=(0.03, 0.02), ts=(1.0, 2.0))
check("a falling theta is a negative calendar spread at every strike",
      _ssvi.calendar_violations(_backwards, -0.4, 0.4, 41) == 41)
check("and admissible parameters do not rescue it",
      _ssvi.admissible(-0.5, 0.8, 0.4)
      and _ssvi.calendar_violations(_backwards) > 0)

# Round trip: generate quotes FROM a known admissible surface, fit them back,
# and the fit must recover a surface that is also admissible and close in level.
_true = _ssvi.SSVIParams(rho=-0.45, eta=1.1, gamma=0.45,
                         thetas=(0.0015, 0.008, 0.03, 0.09),
                         ts=(0.05, 0.25, 1.0, 3.0))
_slices = [{"T": t, "k": np.linspace(-0.5, 0.5, 31),
            "w": _true.total_variance(np.linspace(-0.5, 0.5, 31), th),
            "weights": np.ones(31)}
           for t, th in zip(_true.ts, _true.thetas)]
_sfit = _ssvi.fit_ssvi(_slices)
check(f"a surface generated by SSVI is refitted to itself "
      f"(rmse {_sfit['rmse_w']:.2e})", _sfit["rmse_w"] < 5e-4)
check("the refit is admissible and its theta is monotone",
      _sfit["butterfly_ok"] and _sfit["calendar_ok"] and _sfit["theta_monotone"])
check("the refit recovers the at-the-money term structure",
      float(np.max(np.abs(np.array(_sfit["params"].thetas)
                          - np.array(_true.thetas))
                   / np.array(_true.thetas))) < 0.10)
check("the fitted surface is free of both kinds of arbitrage",
      _ssvi.density_violations(_sfit["params"]) == 0
      and _ssvi.calendar_violations(_sfit["params"]) == 0)

# theta is forced monotone by construction, so quotes whose at-the-money
# variance FALLS with maturity - which real chains do, from a stale quote - must
# come back monotone anyway, at the cost of fit on the offending expiry.
_falling = [{"T": t, "k": np.linspace(-0.4, 0.4, 25),
             "w": _true.total_variance(np.linspace(-0.4, 0.4, 25), th),
             "weights": np.ones(25)}
            for t, th in zip((0.25, 1.0), (0.02, 0.012))]
_ffit = _ssvi.fit_ssvi(_falling)
check("a falling at-the-money term structure is forced monotone",
      _ffit["theta_monotone"]
      and _ffit["params"].thetas[1] >= _ffit["params"].thetas[0])

check("theta_at interpolates the term structure and stays monotone",
      all(_true.theta_at(a) <= _true.theta_at(b)
          for a, b in zip(np.linspace(0.01, 5.0, 60),
                          np.linspace(0.01, 5.0, 60)[1:])))
check("theta_at reproduces the fitted thetas at the fitted maturities",
      all(abs(_true.theta_at(t) - th) < 1e-12
          for t, th in zip(_true.ts, _true.thetas)))

# ---------------------------------------------------------------------------
# The units bug in the fit diagnostic. fit_svi() reports residuals in
# sqrt(total variance), which is sigma*sqrt(T) and therefore a different unit
# at every maturity; fit_smile() divides by sqrt(T) to get volatility. The old
# code reported the former under the name "rmse_vol_points", which made a
# one-week expiry look about seven times better fitted than it was.
# ---------------------------------------------------------------------------
_synth_k = np.linspace(-0.3, 0.3, 21)
for _T in (0.02, 1.0):
    _pp = _svi.SVIParams(a=0.04 * _T, b=0.12 * _T, rho=-0.4, m=0.0, sigma=0.15)
    _fit_u = _svi.fit_svi(_synth_k, _pp.total_variance(_synth_k))
    _vol_rmse = _fit_u["rmse_sqrt_total_var"] / np.sqrt(_T)
    check(f"sqrt-total-variance and volatility error differ by sqrt(T) "
          f"at T={_T}",
          abs(_vol_rmse * np.sqrt(_T) - _fit_u["rmse_sqrt_total_var"]) < 1e-15)
check("fit_svi no longer claims to report volatility points",
      "rmse_vol_points" not in _svi.fit_svi(_synth_k,
                                            _svi.SVIParams(0.04, 0.12, -0.4, 0.0,
                                                           0.15)
                                            .total_variance(_synth_k)))


def test_sanity_checks():
    """Lets `python -m pytest test_sanity.py` see the checks above. They all
    ran when the module was imported; this only reports whether any failed."""
    assert not failures, failures


print()
if failures:
    print(f"{len(failures)} FAILURES: {failures}")
    raise SystemExit(1)
print("All sanity checks passed.")
