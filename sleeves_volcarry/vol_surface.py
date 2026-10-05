"""
Build an implied volatility surface out of a real option chain.

Page 3 of the app has said since it was written that "the market prices in
a volatility SMILE (different implied vol per strike), while our model uses
one flat historical vol for everything." This module is that sentence turned
into numbers: pull a live chain, invert every usable quote with this
project's own Black-Scholes code, and report what shape comes out.

Three things here are less obvious than the inversion itself, and they are
most of what separates a surface from a plot of Yahoo's impliedVolatility
column.

1. THE FORWARD COMES FROM PUT-CALL PARITY, NOT FROM THE SPOT.
   Black-Scholes needs the forward price, and the forward needs a dividend
   yield and a borrow rate that nobody publishes. But the market already
   prices both, and put-call parity says so:

       C - P = e^(-rT) * (F - K)

   which is a straight line in K. Regress call-minus-put on strike across
   the chain and the slope gives the discount factor, the intercept gives
   the forward. No dividend estimate, no borrow assumption - the number the
   options themselves are quoting. When that regression is poor (a thin
   chain, stale quotes), we say so rather than quietly using a bad forward.

2. ONLY OUT-OF-THE-MONEY QUOTES ARE USED.
   Below the forward take the put, above it take the call. In theory the
   two carry the same information; in practice the OTM one is the liquid
   one, has the tighter spread, and is almost all time value, so its price
   is mostly a statement about volatility rather than about intrinsic value
   - which is exactly the quantity being extracted. It also sidesteps most
   of the American-exercise error (see the caveat at the bottom).

3. MID PRICES, AND QUOTES ARE FILTERED BEFORE THEY COUNT.
   `lastPrice` is whenever that contract last traded, which for a far-out
   strike can be days ago on a different spot price. A mid of live bid and
   ask is a price now. Zero bids, crossed markets and spreads wider than a
   set fraction of the mid are dropped outright: a quote whose bid-ask
   straddles ten volatility points does not pin down a volatility, and
   averaging it in is how a surface grows spikes that get explained as
   "skew".

Caveat worth stating up front: single-name and ETF options in the US are
American, and this inverts a European formula. Restricting to OTM quotes
keeps the early-exercise premium small (it is worth almost nothing on an
option with no intrinsic value), but on deep strikes and long maturities it
is not zero, and the IVs here will read slightly high as a result. The
honest fix is inverting the binomial model in binomial.py instead, which is
~500x slower per quote - a real trade-off, not an oversight.
"""

import numpy as np
import pandas as pd

import black_scholes as bs

# market_data is imported inside the two functions that fetch chains rather
# than at module scope. Everything else here is arithmetic on a dataframe
# somebody already has, and it is worth being able to test that arithmetic -
# and use it on a chain from any source - without dragging in yfinance and
# a network connection.

# A quote whose bid-ask spread is wider than this fraction of its mid is
# telling you it doesn't know the price. Half is loose by design - far
# strikes on single names are genuinely wide - and the filter still removes
# most of the garbage, because the garbage is much worse than half.
MAX_REL_SPREAD = 0.5

# Strikes further than this (in log-forward moneyness) are dropped. Beyond
# roughly +/-40% the quotes are pennies, the spread is most of the price,
# and the IV that comes back is noise wearing a number.
MAX_ABS_MONEYNESS = 0.40


def clean_quotes(df: pd.DataFrame, option_type: str) -> pd.DataFrame:
    """Keep the rows that are actually quoting a price. Returns columns
    strike / mid / spread / rel_spread / type, one row per strike."""
    out = df[["strike", "bid", "ask", "volume", "openInterest"]].copy()
    out["bid"] = pd.to_numeric(out["bid"], errors="coerce")
    out["ask"] = pd.to_numeric(out["ask"], errors="coerce")

    out = out[(out["bid"] > 0) & (out["ask"] > 0) & (out["ask"] >= out["bid"])]
    out["mid"] = 0.5 * (out["bid"] + out["ask"])
    out["spread"] = out["ask"] - out["bid"]
    out["rel_spread"] = out["spread"] / out["mid"]
    out = out[out["rel_spread"] <= MAX_REL_SPREAD]
    out["type"] = option_type
    return out.sort_values("strike").reset_index(drop=True)


def forward_from_parity(calls: pd.DataFrame, puts: pd.DataFrame, T: float,
                        min_pairs: int = 6) -> dict:
    """Recover (F, discount factor, implied r) from C - P = e^(-rT)(F - K).

    Fitted on strikes where BOTH a call and a put pass the quote filter, and
    only on the strikes nearest the money: parity holds at every strike in
    theory, but far from the money one leg is a penny option whose mid is
    dominated by its own spread, and including those tilts the line.

    Returns a dict including `r2` and `n_pairs` so the caller can refuse a
    bad fit instead of building a surface on top of one.
    """
    merged = calls.merge(puts, on="strike", suffixes=("_c", "_p"))
    if len(merged) < min_pairs:
        return {"forward": np.nan, "discount": np.nan, "rate": np.nan,
                "r2": np.nan, "n_pairs": len(merged)}

    # nearest-the-money half of the strikes, judged by |C - P| which is
    # smallest where the forward is
    merged["cmp"] = merged["mid_c"] - merged["mid_p"]
    near = merged.reindex(merged["cmp"].abs().sort_values().index)
    near = near.head(max(min_pairs, len(merged) // 2))

    slope, intercept = np.polyfit(near["strike"], near["cmp"], 1)
    predicted = slope * near["strike"] + intercept
    ss_res = float(((near["cmp"] - predicted) ** 2).sum())
    ss_tot = float(((near["cmp"] - near["cmp"].mean()) ** 2).sum())
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    discount = -slope                      # slope is -e^(-rT)
    if not (0 < discount <= 1.5):          # a nonsense fit, not a cheap rate
        return {"forward": np.nan, "discount": np.nan, "rate": np.nan,
                "r2": r2, "n_pairs": len(merged)}

    forward = intercept / discount
    rate = -np.log(discount) / T if T > 0 else np.nan
    return {"forward": float(forward), "discount": float(discount),
            "rate": float(rate), "r2": float(r2), "n_pairs": int(len(merged))}


def smile_for_expiry(calls_raw: pd.DataFrame, puts_raw: pd.DataFrame,
                     T: float, expiry: str) -> pd.DataFrame:
    """One expiry's smile: OTM quotes inverted to implied vol.

    Every row that survives has a finite IV from a quote that passed the
    filters, priced against the forward the chain itself implies.
    """
    calls = clean_quotes(calls_raw, "call")
    puts = clean_quotes(puts_raw, "put")
    fwd = forward_from_parity(calls, puts, T)
    if not np.isfinite(fwd["forward"]) or fwd["r2"] < 0.99:
        return pd.DataFrame()

    F, disc = fwd["forward"], fwd["discount"]
    otm = pd.concat([puts[puts["strike"] < F], calls[calls["strike"] >= F]])
    otm = otm[np.abs(np.log(otm["strike"] / F)) <= MAX_ABS_MONEYNESS].copy()

    # Price off the forward rather than the spot: with S = F * e^(-rT) fed
    # in as spot and q = 0, the Black-Scholes d1/d2 see exactly the forward
    # the market is quoting, dividends and borrow already inside it.
    spot_equiv = F * disc
    rate = fwd["rate"]

    otm["iv"] = [
        bs.implied_vol(row.type, row.mid, spot_equiv, row.strike, T, rate)
        for row in otm.itertuples()
    ]
    otm = otm[np.isfinite(otm["iv"])].copy()
    if otm.empty:
        return pd.DataFrame()

    otm["expiry"] = expiry
    otm["T"] = T
    otm["forward"] = F
    otm["discount"] = disc
    otm["rate"] = rate
    otm["moneyness"] = np.log(otm["strike"] / F)
    otm["total_var"] = otm["iv"] ** 2 * T
    # Delta of the quoted (OTM) leg, for the 25-delta conventions below.
    otm["delta"] = [
        float(bs.delta(row.type, spot_equiv, row.strike, T, rate, row.iv))
        for row in otm.itertuples()
    ]
    return otm.sort_values("strike").reset_index(drop=True)


def _spread_expiries(expiries: list, min_days: int, max_expiries: int) -> list:
    """Pick expiries spread across the term, not just the nearest ones.

    A chain on a liquid ETF has thirty expiries, most of them in the next
    fortnight. Taking the first eight gives eight nearly identical smiles
    and a term structure that covers six weeks. Spacing the picks evenly in
    log-maturity gives a week, a month, a quarter and a year instead, which
    is the range where term structure does anything interesting.
    """
    import market_data as md

    dated = [(e, md.years_to_expiry(e)) for e in expiries]
    dated = [(e, t) for e, t in dated if t * 365 >= min_days]
    if len(dated) <= max_expiries:
        return [e for e, _ in dated]

    logs = np.log([t for _, t in dated])
    targets = np.linspace(logs[0], logs[-1], max_expiries)
    picked, used = [], set()
    for target in targets:
        i = int(np.argmin(np.abs(logs - target)))
        while i in used:               # nearest already taken, step outward
            i += 1
            if i >= len(dated):
                break
        if i < len(dated):
            used.add(i)
            picked.append(dated[i][0])
    return picked


def build_surface(ticker: str, max_expiries: int = 8, min_days: int = 7,
                  verbose: bool = False) -> pd.DataFrame:
    """Every usable OTM quote across expiries spanning the available term.

    Expiries inside `min_days` are skipped. Their IVs are dominated by event
    risk and by the fact that one stale quote is a large part of the
    remaining life, and they distort a term structure far more than they
    inform it.
    """
    import market_data as md

    chosen = _spread_expiries(md.get_expirations(ticker), min_days, max_expiries)
    frames = []
    for expiry in chosen:
        T = md.years_to_expiry(expiry)
        try:
            calls_raw, puts_raw = md.get_option_chain(ticker, expiry)
        except Exception as exc:
            if verbose:
                print(f"  {expiry}: chain unavailable ({type(exc).__name__})")
            continue
        smile = smile_for_expiry(calls_raw, puts_raw, T, expiry)
        if smile.empty:
            if verbose:
                print(f"  {expiry}: no usable quotes after filtering")
            continue
        if verbose:
            print(f"  {expiry}: {len(smile):>4} quotes, forward "
                  f"{smile['forward'].iloc[0]:.2f}")
        frames.append(smile)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def atm_vol(smile: pd.DataFrame) -> float:
    """IV at the forward, interpolated in total variance across moneyness.

    Interpolating variance rather than volatility is the convention, and it
    is not cosmetic: variance is what adds across time and what the
    no-arbitrage conditions are stated in, so an interpolation done in vol
    can produce a surface that violates them between the quoted strikes.
    """
    s = smile.dropna(subset=["iv"]).sort_values("moneyness")
    if len(s) < 2:
        return float("nan")
    w = np.interp(0.0, s["moneyness"], s["total_var"])
    T = float(s["T"].iloc[0])
    return float(np.sqrt(w / T)) if T > 0 else float("nan")


def skew_25d(smile: pd.DataFrame) -> dict:
    """The two numbers a desk actually quotes a smile with.

    Risk reversal (25-delta put IV minus 25-delta call IV) measures the
    TILT: positive means downside protection costs more than upside, which
    is the normal state of an equity index and the reason a flat-vol model
    misprices puts.

    Butterfly (the average of the two wings minus the at-the-money level)
    measures the CURVATURE: how much more the market charges for tails than
    a lognormal would.

    Both are read off the quoted deltas by interpolation, so "25-delta"
    means the strike the market would call 25-delta, not a strike computed
    from an assumed volatility.
    """
    puts = smile[smile["type"] == "put"].copy()
    calls = smile[smile["type"] == "call"].copy()
    atm = atm_vol(smile)
    if puts.empty or calls.empty or not np.isfinite(atm):
        return {"put_25d": np.nan, "call_25d": np.nan,
                "risk_reversal": np.nan, "butterfly": np.nan, "atm": atm}

    # put deltas are negative; -0.25 is the 25-delta put
    puts = puts.sort_values("delta")
    calls = calls.sort_values("delta")
    put_25 = float(np.interp(-0.25, puts["delta"], puts["iv"]))
    call_25 = float(np.interp(0.25, calls["delta"], calls["iv"]))
    return {"put_25d": put_25, "call_25d": call_25,
            "risk_reversal": put_25 - call_25,
            "butterfly": 0.5 * (put_25 + call_25) - atm,
            "atm": atm}


def term_structure(surface: pd.DataFrame) -> pd.DataFrame:
    """ATM vol, skew and curvature by expiry — the surface in five columns."""
    rows = []
    for expiry, smile in surface.groupby("expiry", sort=False):
        stats = skew_25d(smile)
        rows.append({"expiry": expiry, "T": float(smile["T"].iloc[0]),
                     "forward": float(smile["forward"].iloc[0]),
                     "n_quotes": len(smile), **stats})
    return pd.DataFrame(rows).sort_values("T").reset_index(drop=True)


def calendar_arbitrage(surface: pd.DataFrame,
                       grid=(-0.15, -0.075, 0.0, 0.075, 0.15)) -> pd.DataFrame:
    """Total variance must not fall as maturity rises, at fixed moneyness.

    If it does, a calendar spread is free money, and the surface is wrong
    somewhere - stale quotes, a bad forward, or a dividend the parity fit
    didn't see. Worth checking rather than assuming: a surface built from
    real quotes violates this often enough that presenting one without the
    check is presenting something unverified.
    """
    by_expiry = {exp: smile.sort_values("moneyness")
                 for exp, smile in surface.groupby("expiry", sort=False)}
    order = sorted(by_expiry, key=lambda e: by_expiry[e]["T"].iloc[0])

    rows = []
    for k in grid:
        prev_w, prev_exp = None, None
        for exp in order:
            s = by_expiry[exp]
            if not (s["moneyness"].min() <= k <= s["moneyness"].max()):
                continue
            w = float(np.interp(k, s["moneyness"], s["total_var"]))
            if prev_w is not None and w < prev_w - 1e-9:
                rows.append({"moneyness": k, "from": prev_exp, "to": exp,
                             "w_from": prev_w, "w_to": w})
            prev_w, prev_exp = w, exp
    return pd.DataFrame(rows)


def flat_vol_error(smile: pd.DataFrame, flat_sigma: float) -> pd.DataFrame:
    """What one flat volatility costs, priced strike by strike.

    This is the point of the whole module. The app's market-comparison page
    prices every strike with a single historical volatility; here is the
    same chain priced that way next to its actual market price, so the
    disagreement is a dollar figure per contract rather than an assertion
    that a smile exists somewhere.

    Priced against the same forward the IVs were solved against, so the only
    thing that differs between the two columns is the volatility input.
    """
    T = float(smile["T"].iloc[0])
    spot_equiv = float(smile["forward"].iloc[0]) * float(smile["discount"].iloc[0])
    rate = float(smile["rate"].iloc[0])

    out = smile[["strike", "type", "mid", "iv", "moneyness"]].copy()
    out["model"] = [
        float(bs.price(row.type, spot_equiv, row.strike, T, rate, flat_sigma))
        for row in out.itertuples()
    ]
    out["error"] = out["model"] - out["mid"]
    out["rel_error"] = out["error"] / out["mid"]
    return out
