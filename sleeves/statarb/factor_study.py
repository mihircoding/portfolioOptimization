"""Hedge every spread against the cross-section's first principal component.

Usage:  python -m sleeves.statarb.factor_study     (reads results/ and data/, no network)

continuation_study.py ended on a correlation: the same-sector continuation
book and the cross-sector reverting book move together at -0.89 day by day,
their best and worst days are the same two days of April 2025 with the signs
swapped, and so the sector was never a property of the pair. A hundred names
make one cross-section, every name sits in 99 pairs, and the day the market
reverses a factor move reverses most of the 4,950 spreads at once. The sector
sign only decides which side of those days a book is on.

The direct test is to take the factor out. Each day the first principal
component of the universe's returns is fitted on the trailing 250 days ending
the day before (refitted every 21 days, so nothing in the fit has seen the
day it is applied to), and each name's loading on that factor is the slope
of its returns on the factor over the same window. The factor return on the
day is the universe's returns on the lagged loadings. A pair's P&L is then
its spread return less its net loading times the factor return, and the
hedge is charged 5bps a side on the change in its notional like any other
leg. The positions are the shipped ones - the z-score is still read on the
raw spread - so the only thing that changes is what the trade is paid in.
For the first year of the tape the fit runs into the formation tape, so the
hedge is on from the first day of 2021.

Two more views of the same thing, both ex ante:

  quiet regime   trade the shipped positions only when the trailing 20-day
                 vol of the factor is under its trailing-year median, flat
                 otherwise. The days the factor is loud are the reversal
                 days, so this should hold whatever is left once they are
                 taken out - with no hedging error.
  attribution    the raw book's gross P&L split by the size of the factor's
                 move that day (after the fact, so a reading and not a trade).

Six predictions, written before the run:

  1. Hedged, the correlation between the same-sector continuation book and
     the cross-sector reverting book goes from -0.89 to between -0.50 and 0.
  2. Hedged, the cross-sector reverting book stops being lumpy: excess
     kurtosis under 5 (from 9.1), best-5%-of-days share of gains under 30%
     (from 37%), and 3 and 9 April 2025 are no longer its worst and best day.
  3. Hedged, the sign-by-sector book is within 0.05 of Sharpe of the
     everything-reverting book: the sector sign is worth nothing once the
     reversal days are taken out.
  4. Hedged, the cross-sector reverting book keeps a positive gross Sharpe,
     between 0 and +0.20 - the pair-level reversion is real and small - and
     the hedge's costs put it under the raw book net.
  5. Per pair, the hedged same-sector mean gross Sharpe stays within 0.01 of
     zero in either direction.
  6. More than half of the raw everything-reverting book's gross P&L comes
     from the loudest fifth of factor days, and the quiet-regime trade has
     a lower gross Sharpe than the full-time book.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sleeves.statarb.control import book_stats, permutation_test
from sleeves.statarb.continuation_study import lumpiness, mirror
from sleeves.statarb.scan import (COST_BPS, ZSCORE_WINDOW, backtest, generate_positions,
                                  rolling_zscore, sharpe)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
DATA = ROOT / "data"
OUT = RESULTS / "factor_study.json"
FORMATION_TAPE = DATA / "universe_102_2013-01-01_2020-12-31.csv"

FIT_WINDOW = 250      # trading days behind the fit
REFIT_EVERY = 21      # a month
REGIME_WINDOW = 20    # the factor's trailing vol
REGIME_MEDIAN = 250   # and what that vol is compared against


def first_component(returns: np.ndarray) -> np.ndarray:
    """Loadings of the first principal component of a (days x names) return
    block, signed so the names load positive on average and scaled to sum
    to one, so the factor's return is the return of a basket with those
    weights and a loading on it is a notional. Fitted on the covariance,
    not the correlation: a factor that moves the whole cross-section is a
    dollar thing, and the high-vol names should carry more of it."""
    x = returns - returns.mean(axis=0)
    cov = x.T @ x / max(len(x) - 1, 1)
    vals, vecs = np.linalg.eigh(cov)
    w = vecs[:, np.argmax(vals)]
    if w.sum() < 0:
        w = -w
    return w / w.sum()


def rolling_factor(returns: pd.DataFrame, window: int = FIT_WINDOW,
                   refit_every: int = REFIT_EVERY, start: pd.Timestamp | None = None) -> dict:
    """The factor return and every name's loading, day by day, with nothing
    in either fitted on the day it is used. `returns` should include the
    warm-up; `start` is the first day the factor is wanted for."""
    r = returns.to_numpy(dtype=float)
    idx = returns.index
    first = idx.get_indexer([start])[0] if start is not None else window
    first = max(first, window)
    factor = pd.Series(np.nan, index=idx)
    betas = pd.DataFrame(np.nan, index=idx, columns=returns.columns)
    explained = pd.Series(np.nan, index=idx)
    w = b = None
    for t in range(first, len(idx)):
        if w is None or (t - first) % refit_every == 0:
            block = r[t - window:t]
            w = first_component(block)
            f_hist = block @ w
            f_c = f_hist - f_hist.mean()
            b = ((block - block.mean(axis=0)).T @ f_c) / (f_c @ f_c)
            vals = np.linalg.eigvalsh(np.cov(block, rowvar=False))
            explained.iloc[t] = float(vals.max() / vals.sum())
        factor.iloc[t] = r[t] @ w
        betas.iloc[t] = b
    return {"factor": factor, "betas": betas, "explained": explained.ffill()}


def hedged_backtest(y: pd.Series, x: pd.Series, positions: pd.Series, factor: pd.Series,
                    beta_y: pd.Series, beta_x: pd.Series, cost_bps: float = COST_BPS) -> pd.DataFrame:
    """The shipped backtest with the factor taken out of the P&L. The hedge
    notional is the pair's net loading, lagged like the position, and it is
    charged on its changes the way the legs are charged on theirs."""
    lagged = positions.shift(1)
    spread_ret = (y.pct_change() - x.pct_change()) / 2.0
    exposure = (lagged * (beta_y.shift(1) - beta_x.shift(1)) / 2.0).fillna(0.0)
    ret = (lagged * spread_ret - exposure * factor).fillna(0.0)
    turnover = positions.diff()
    turnover.iloc[0] = positions.iloc[0]
    hedge_turnover = exposure.diff().abs().fillna(0.0)
    ret_net = ret - (turnover.abs() + hedge_turnover) * (cost_bps / 10_000.0)
    return pd.DataFrame({"ret": ret, "ret_net": ret_net,
                         "equity": (1.0 + ret_net).cumprod()}, index=y.index)


def quiet_regime(factor: pd.Series, window: int = REGIME_WINDOW,
                 median_window: int = REGIME_MEDIAN) -> pd.Series:
    """True on the days the factor has been quiet lately: its trailing vol
    is under the trailing-year median of that vol. Both read through the
    day before, so the flag is known when the position is put on."""
    vol = factor.rolling(window).std()
    ref = vol.rolling(median_window, min_periods=window).median()
    quiet = (vol < ref).to_numpy()
    out = np.r_[False, quiet[:-1]]
    return pd.Series(out, index=factor.index, dtype=bool)


def run_all(scan: pd.DataFrame, trading: pd.DataFrame, factor: pd.Series, betas: pd.DataFrame,
            quiet: pd.Series, cost_bps: float = COST_BPS) -> dict:
    """Every pair, raw and hedged and in the quiet regime only, with the
    shipped positions in all three."""
    labels = ["raw", "hedged", "quiet"]
    gross = {k: {} for k in labels}
    net = {k: {} for k in labels}
    rows = []
    for row in scan.itertuples():
        y, x = trading[row.a], trading[row.b]
        key = f"{row.a}/{row.b}"
        z = rolling_zscore(y - row.beta * x, ZSCORE_WINDOW)
        pos = generate_positions(z)
        results = {"raw": backtest(y, x, pos, cost_bps),
                   "hedged": hedged_backtest(y, x, pos, factor, betas[row.a], betas[row.b], cost_bps),
                   "quiet": backtest(y, x, pos.where(quiet, 0), cost_bps)}
        stats = {"a": row.a, "b": row.b, "same_sector": bool(row.same_sector)}
        for k, r in results.items():
            gross[k][key] = r["ret"]
            net[k][key] = r["ret_net"]
            stats[f"{k}_gross_sharpe"] = sharpe(r["ret"])
            stats[f"{k}_sharpe"] = sharpe(r["ret_net"])
            stats[f"{k}_return"] = float(r["equity"].iloc[-1] - 1.0)
        rows.append(stats)
    return {"stats": pd.DataFrame(rows),
            "gross": {k: pd.DataFrame(v) for k, v in gross.items()},
            "net": {k: pd.DataFrame(v) for k, v in net.items()}}


def book_row(gross: pd.DataFrame, net: pd.DataFrame, mask: np.ndarray) -> dict:
    g = book_stats(gross.loc[:, mask])
    n = book_stats(net.loc[:, mask])
    return {"pairs": int(mask.sum()), "gross_sharpe": g["sharpe"], "gross_return": g["total_return"],
            "sharpe": n["sharpe"], "total_return": n["total_return"]}


def books_for(gross: pd.DataFrame, net: pd.DataFrame, same: np.ndarray, passed: np.ndarray) -> dict:
    everything = np.ones(len(same), dtype=bool)
    mg, mn = mirror_panels(gross, net)
    mixed_gross = pd.concat([gross.loc[:, ~same], mg.loc[:, same]], axis=1)
    mixed_net = pd.concat([net.loc[:, ~same], mn.loc[:, same]], axis=1)
    return {
        "everything, reverting": book_row(gross, net, everything),
        "screened, reverting (shipped)": book_row(gross, net, passed),
        "cross sector, reverting": book_row(gross, net, ~same),
        "same sector, reverting": book_row(gross, net, same),
        "same sector, mirrored": book_row(mg, mn, same),
        "sign by sector": book_row(mixed_gross, mixed_net, everything),
        "sign by sector, screened": book_row(mixed_gross, mixed_net, passed),
    }


def mirror_panels(gross: pd.DataFrame, net: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    costs = gross - net
    return -gross, -gross - costs


def attribution(daily: pd.Series, factor: pd.Series, bins: int = 5) -> list[dict]:
    """The book's gross P&L by quintile of the factor's absolute move that
    day. Read after the fact; the loud days are known only once they have
    happened."""
    d = pd.DataFrame({"pnl": daily, "size": factor.abs()}).dropna()
    d["bin"] = pd.qcut(d["size"].rank(method="first"), bins, labels=False)
    total = d["pnl"].sum()
    out = []
    for b, g in d.groupby("bin"):
        out.append({"quintile": int(b) + 1, "days": int(len(g)),
                    "factor_abs_mean": float(g["size"].mean()),
                    "gross_pnl": float(g["pnl"].sum()),
                    "share_of_total": float(g["pnl"].sum() / total) if total != 0 else float("nan"),
                    "sharpe": sharpe(g["pnl"])})
    return out


def extreme_days(daily: pd.Series) -> dict:
    d = daily.dropna()
    return {"best_day": str(d.idxmax().date()), "best": float(d.max()),
            "worst_day": str(d.idxmin().date()), "worst": float(d.min())}


def verdicts(raw_books: dict, hedged_books: dict, quiet_books: dict, corr: dict, lumps: dict,
             extremes: dict, per_pair: dict, attrib: list[dict]) -> list[dict]:
    h_cross, r_cross = hedged_books["cross sector, reverting"], raw_books["cross sector, reverting"]
    h_sign, h_all = hedged_books["sign by sector"], hedged_books["everything, reverting"]
    top_share = attrib[-1]["share_of_total"]
    return [
        {"claim": "hedged, the continuation/reversion correlation is between -0.50 and 0 (from -0.89)",
         "measured": f"raw {corr['raw']:+.2f}, hedged {corr['hedged']:+.2f}",
         "held": -0.50 <= corr["hedged"] <= 0.0},
        {"claim": "hedged cross-sector reversion is not lumpy: kurtosis < 5, top-5% share < 30%, "
                  "and 3/9 April 2025 are no longer its worst/best days",
         "measured": f"kurtosis {lumps['hedged']['excess_kurtosis']:.1f}, top-5% share "
                     f"{lumps['hedged']['top5pct_share_of_gains']:.1%}, worst {extremes['hedged']['worst_day']}, "
                     f"best {extremes['hedged']['best_day']}",
         "held": (lumps["hedged"]["excess_kurtosis"] < 5 and lumps["hedged"]["top5pct_share_of_gains"] < 0.30
                  and extremes["hedged"]["best_day"] != "2025-04-09"
                  and extremes["hedged"]["worst_day"] != "2025-04-03")},
        {"claim": "hedged, sign-by-sector is within 0.05 Sharpe of everything-reverting",
         "measured": f"sign by sector {h_sign['sharpe']:+.2f} vs everything {h_all['sharpe']:+.2f}",
         "held": abs(h_sign["sharpe"] - h_all["sharpe"]) <= 0.05},
        {"claim": "hedged cross-sector reversion: gross Sharpe in (0, +0.20], net below the raw book's net",
         "measured": f"gross {h_cross['gross_sharpe']:+.2f} (raw {r_cross['gross_sharpe']:+.2f}), "
                     f"net {h_cross['sharpe']:+.2f} (raw {r_cross['sharpe']:+.2f})",
         "held": 0 < h_cross["gross_sharpe"] <= 0.20 and h_cross["sharpe"] < r_cross["sharpe"]},
        {"claim": "per pair, hedged same-sector mean gross Sharpe within 0.01 of zero",
         "measured": f"same {per_pair['hedged']['same_gross_mean']:+.4f}, cross "
                     f"{per_pair['hedged']['cross_gross_mean']:+.4f}",
         "held": abs(per_pair["hedged"]["same_gross_mean"]) <= 0.01},
        {"claim": "over half the raw book's gross P&L is in the loudest fifth of factor days, "
                  "and the quiet-regime trade has the lower gross Sharpe",
         "measured": f"loudest fifth {top_share:+.0%} of gross P&L; gross Sharpe quiet "
                     f"{quiet_books['everything, reverting']['gross_sharpe']:+.2f} vs full-time "
                     f"{raw_books['everything, reverting']['gross_sharpe']:+.2f}",
         "held": (top_share > 0.5 and quiet_books["everything, reverting"]["gross_sharpe"]
                  < raw_books["everything, reverting"]["gross_sharpe"])},
    ]


def name_weights(scan: pd.DataFrame, trading: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    """What a book of pairs holds, name by name: each pair's lagged position
    puts +1/2 on its first leg and -1/2 on its second, summed over the pairs
    in `mask` and divided by their number. The book's gross daily return is
    these weights on the names' returns, which is a check the tests make."""
    weights = pd.DataFrame(0.0, index=trading.index, columns=trading.columns)
    rows = scan[mask]
    for row in rows.itertuples():
        z = rolling_zscore(trading[row.a] - row.beta * trading[row.b], ZSCORE_WINDOW)
        pos = generate_positions(z).shift(1).fillna(0).to_numpy() / 2.0
        weights[row.a] += pos
        weights[row.b] -= pos
    return weights / max(len(rows), 1)


def reversal_factor(trading: pd.DataFrame, lookback: int = 60, decile: int = 10) -> pd.Series:
    """Long the ten names with the worst trailing `lookback`-day return,
    short the ten best, equal weight, known the day before. The simplest
    cross-sectional reversal portfolio there is."""
    past = trading.pct_change(lookback).shift(1)
    ret = trading.pct_change()
    out = pd.Series(np.nan, index=trading.index)
    for t in range(lookback + 1, len(trading)):
        m = past.iloc[t].dropna()
        if len(m) < 2 * decile:
            continue
        lo, hi = m.nsmallest(decile).index, m.nlargest(decile).index
        out.iloc[t] = ret.iloc[t][lo].mean() - ret.iloc[t][hi].mean()
    return out


def what_the_books_hold(scan: pd.DataFrame, trading: pd.DataFrame, same: np.ndarray) -> dict:
    """The cross-sector and same-sector reverting books as name weights:
    how alike they are day by day, and how much each is the trailing-60-day
    reversal portfolio."""
    cross, same_w = name_weights(scan, trading, ~same), name_weights(scan, trading, same)
    past = trading.pct_change(ZSCORE_WINDOW).shift(1)
    ret = trading.pct_change().fillna(0.0)
    daily_corr, past_corr = [], []
    for t in range(len(trading)):
        a, b, m = cross.iloc[t].to_numpy(), same_w.iloc[t].to_numpy(), past.iloc[t].to_numpy()
        if a.std() > 0 and b.std() > 0:
            daily_corr.append(np.corrcoef(a, b)[0, 1])
        if a.std() > 0 and not np.isnan(m).any():
            past_corr.append(np.corrcoef(a, m)[0, 1])
    rev = reversal_factor(trading)
    books = {"cross": (cross * ret).sum(axis=1), "same": (same_w * ret).sum(axis=1)}
    return {"weights_corr_mean": float(np.mean(daily_corr)),
            "weights_corr_share_positive": float(np.mean(np.array(daily_corr) > 0)),
            "cross_weight_vs_trailing_return": float(np.mean(past_corr)),
            "corr_with_reversal_decile": {k: float(v.corr(rev)) for k, v in books.items()},
            "reversal_decile_sharpe": sharpe(rev),
            "cross_gross_names_share_top10": float((np.sort(cross.abs().to_numpy(), axis=1)[:, -10:].sum(axis=1)
                                                   / cross.abs().sum(axis=1).replace(0, np.nan)).mean())}


def load_returns(trading: pd.DataFrame) -> pd.DataFrame:
    """The trading tape with the formation tape in front of it, as daily
    returns on the trading tape's names, so the first fit has a year behind it."""
    formation = pd.read_csv(FORMATION_TAPE, index_col=0, parse_dates=True)
    prices = pd.concat([formation.reindex(columns=trading.columns).dropna(), trading])
    prices = prices[~prices.index.duplicated(keep="last")].sort_index()
    return prices.pct_change().iloc[1:]


def main() -> None:
    scan = pd.read_parquet(RESULTS / "scan.parquet")
    trading = pd.read_parquet(RESULTS / "trading.parquet")
    same = scan["same_sector"].to_numpy()
    passed = (scan["pvalue"] <= 0.05).to_numpy()

    returns = load_returns(trading)
    fit = rolling_factor(returns, start=trading.index[0])
    factor = fit["factor"].reindex(trading.index)
    betas = fit["betas"].reindex(trading.index)
    quiet = quiet_regime(fit["factor"]).reindex(trading.index).fillna(False)
    explained = float(fit["explained"].reindex(trading.index).mean())
    print(f"First component explains {explained:.0%} of the trailing-year variance on average; "
          f"factor daily vol {factor.std() * 100:.2f}%; the quiet regime is on {quiet.mean():.0%} "
          f"of days.")
    print(f"{len(scan):,} pairs, raw / hedged / quiet-regime, shipped positions (the slow part)...")
    run = run_all(scan, trading, factor, betas, quiet)
    stats, gross, net = run["stats"], run["gross"], run["net"]

    books = {k: books_for(gross[k], net[k], same, passed) for k in ("raw", "hedged", "quiet")}
    print(f"\n{'book':<32} {'arm':<7} {'pairs':>6} {'gross Sh':>9} {'gross ret':>10} {'net Sh':>7} {'net ret':>8}")
    for label in books["raw"]:
        for arm in ("raw", "hedged", "quiet"):
            b = books[arm][label]
            print(f"{label if arm == 'raw' else '':<32} {arm:<7} {b['pairs']:>6,} {b['gross_sharpe']:>+9.2f} "
                  f"{b['gross_return']:>+10.2%} {b['sharpe']:>+7.2f} {b['total_return']:>+8.2%}")

    # the correlation the last study ended on, raw and hedged
    corr = {}
    for arm in ("raw", "hedged"):
        cont = -gross[arm].loc[:, same].mean(axis=1)
        rev = gross[arm].loc[:, ~same].mean(axis=1)
        corr[arm] = float(cont.corr(rev))
    print(f"\nSame-sector continuation vs cross-sector reversion, daily gross: raw {corr['raw']:+.2f}, "
          f"hedged {corr['hedged']:+.2f}")

    # how much of each pair's P&L the hedge removes, and how big the hedge is
    exposure = (betas[scan["a"]].to_numpy() - betas[scan["b"]].to_numpy()) / 2.0
    hedge_size = {"mean_abs_net_loading": float(np.nanmean(np.abs(exposure))),
                  "share_of_pairs_under_0p1": float((np.nanmean(np.abs(exposure), axis=0) < 0.1).mean())}
    raw_book = gross["raw"].mean(axis=1)
    hedged_book = gross["hedged"].mean(axis=1)
    removed = raw_book - hedged_book
    net_exposure = pd.Series(np.nan_to_num(exposure).sum(axis=1) / len(scan), index=trading.index)
    hedge_size["book_var_removed"] = float(1.0 - hedged_book.var() / raw_book.var())
    hedge_size["removed_corr_with_factor"] = float(removed.corr(factor))
    print(f"\nThe hedge: mean |net loading| {hedge_size['mean_abs_net_loading']:.3f}, "
          f"{hedge_size['share_of_pairs_under_0p1']:.0%} of pairs under 0.1; it removes "
          f"{hedge_size['book_var_removed']:.0%} of the everything-book's daily variance.")

    lumps = {arm: lumpiness(gross[arm].loc[:, ~same].mean(axis=1)) for arm in ("raw", "hedged", "quiet")}
    extremes = {arm: extreme_days(gross[arm].loc[:, ~same].mean(axis=1)) for arm in ("raw", "hedged")}
    print("\nCross-sector reverting book, shape of the gross daily P&L:")
    for arm in ("raw", "hedged", "quiet"):
        l = lumps[arm]
        print(f"  {arm:<7} excess kurtosis {l['excess_kurtosis']:>5.1f}   best 5% of days = "
              f"{l['top5pct_share_of_gains']:.1%} of the gains", end="")
        if arm in extremes:
            e = extremes[arm]
            print(f"   worst {e['worst_day']} {e['worst']:+.2%}, best {e['best_day']} {e['best']:+.2%}")
        else:
            print()

    per_pair = {}
    for arm in ("raw", "hedged"):
        g_same = stats.loc[same, f"{arm}_gross_sharpe"].to_numpy()
        g_cross = stats.loc[~same, f"{arm}_gross_sharpe"].to_numpy()
        n_same = stats.loc[same, f"{arm}_sharpe"].to_numpy()
        n_cross = stats.loc[~same, f"{arm}_sharpe"].to_numpy()
        per_pair[arm] = {"same_gross_mean": float(g_same.mean()), "cross_gross_mean": float(g_cross.mean()),
                         "same_net_mean": float(n_same.mean()), "cross_net_mean": float(n_cross.mean()),
                         **permutation_test(n_same, n_cross)}
        p = per_pair[arm]
        print(f"\nPer pair, {arm}: mean gross Sharpe same {p['same_gross_mean']:+.4f} vs cross "
              f"{p['cross_gross_mean']:+.4f}; net {p['same_net_mean']:+.3f} vs {p['cross_net_mean']:+.3f} "
              f"(p = {p['p_value']:.3f})")

    holds = what_the_books_hold(scan, trading, same)
    print(f"\nWhat the two reverting books hold: the cross-sector and same-sector name weights "
          f"correlate at {holds['weights_corr_mean']:+.2f} across names on an average day (positive on "
          f"{holds['weights_corr_share_positive']:.0%} of days); the cross-sector book's weight on a name "
          f"correlates at {holds['cross_weight_vs_trailing_return']:+.2f} with its trailing 60-day return; "
          f"the books correlate at {holds['corr_with_reversal_decile']['cross']:+.2f} and "
          f"{holds['corr_with_reversal_decile']['same']:+.2f} with a long-worst-ten short-best-ten "
          f"60-day reversal portfolio (Sharpe {holds['reversal_decile_sharpe']:+.2f}).")

    attrib = attribution(raw_book, factor)
    attrib_hedged = attribution(hedged_book, factor)
    print(f"\nRaw everything-reverting book's gross P&L by quintile of |factor move| that day:")
    print(f"  {'quintile':<9} {'days':>5} {'|f| mean':>9} {'gross P&L':>10} {'share':>7} {'Sharpe':>7}")
    for a in attrib:
        print(f"  {a['quintile']:<9} {a['days']:>5} {a['factor_abs_mean'] * 100:>8.2f}% {a['gross_pnl']:>+10.2%} "
              f"{a['share_of_total']:>+7.0%} {a['sharpe']:>+7.2f}")

    scored = verdicts(books["raw"], books["hedged"], books["quiet"], corr, lumps, extremes, per_pair, attrib)
    print("\nPredictions, scored:")
    for v in scored:
        print(f"  [{'held' if v['held'] else 'FAILED'}] {v['claim']}\n         measured: {v['measured']}")

    monthly = {}
    for label, series in (("raw, everything", net["raw"].mean(axis=1)),
                          ("hedged, everything", net["hedged"].mean(axis=1)),
                          ("quiet, everything", net["quiet"].mean(axis=1))):
        m = (1.0 + series).resample("ME").prod().sub(1.0).round(6)
        monthly[label] = {str(d.date()): float(v) for d, v in m.items()}
    payload = {"factor": {"explained_variance": explained, "daily_vol": float(factor.std()),
                          "quiet_share": float(quiet.mean()), "fit_window": FIT_WINDOW,
                          "refit_every": REFIT_EVERY},
               "books": books, "correlation": corr, "hedge": hedge_size, "lumpiness": lumps,
               "extremes": extremes, "per_pair": per_pair, "what_the_books_hold": holds,
               "attribution": {"raw": attrib, "hedged": attrib_hedged},
               "net_exposure_abs_mean": float(net_exposure.abs().mean()),
               "verdicts": scored, "monthly": monthly}
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
