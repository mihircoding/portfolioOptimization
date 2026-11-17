"""Take the sector as the sign of the trade, instead of as a filter.

Usage:  python -m sleeves.statarb.continuation_study     (reads results/, no network)

sector_study.py ended with a reading, not a trade: a two-sigma move between
two names in the same sector is the market re-ranking them and it sticks,
while the same move between two unrelated names is noise and comes back. If
that is right, the sector is not a filter, it is a sign. A cross-sector pair
should be traded the way every section of this project trades it, against
the move, and a same-sector pair should be traded WITH the move. This runs
both, on all 4,950 pairs, with the control group's rules and costs.

Three ways of going with the move, because the exit matters:

  mirrored   the shipped rules with the sign flipped: enter at |z| = 2 in the
             direction of the move, exit when |z| is back under 0.5. Same
             turnover, same costs, gross P&L exactly the negative of the
             reverting trade's. The trade ends when the 60-day window has
             absorbed the move, which is what "the re-ranking sticks" means
             in this backtest's terms.
  h-day      enter on a fresh crossing, hold h days, go flat. The post-hoc
             after-entry table of the sector study, turned into a trade
             with costs on it.
  by sector  the mixed book: cross-sector pairs reverting, same-sector pairs
             mirrored. This is the reading as a book.

There is no finer GICS field in the universe than the sector (universe.py
carries one code per ticker), so the split by industry group that the
sector study suggested cannot be made here without inventing a mapping,
and is not.

Five predictions, written before the backtest ran:

  1. Mirrored on the 585 same-sector pairs, the continuation book makes
     money gross and gives it back to costs: gross Sharpe above zero, net
     Sharpe between -0.20 and +0.10.
  2. Mirrored on the 4,365 cross-sector pairs it is a bad trade: the net
     book Sharpe is below the reverting cross-sector book's (-0.09) by more
     than 0.20.
  3. The sign-by-sector book beats the shipped book (the screened 930 at
     -0.19) by more than 0.10 in Sharpe, and beats the everything book
     (-0.12).
  4. Held for a fixed 20 days, the same-sector continuation trade is
     positive gross and negative net at 5bps a side; the cross-sector
     continuation trade is negative gross at every horizon.
  5. Re-rankings are events, not drift, so the same-sector continuation
     book's daily P&L is lumpier than the cross-sector reverting book's:
     higher excess kurtosis, and its best 5% of days are a larger share
     of its gross P&L.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

from sleeves.statarb.control import book_stats, permutation_test
from sleeves.statarb.scan import (COST_BPS, ENTRY, ZSCORE_WINDOW, backtest, generate_positions,
                                  rolling_zscore, sharpe)

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"
OUT = RESULTS / "continuation_study.json"
HORIZONS = (5, 10, 20)


def mirror(result: pd.DataFrame) -> pd.DataFrame:
    """The same trade with the sign flipped. Costs are paid on turnover and
    turnover does not know the sign, so the mirrored net return is the
    negative of the gross return less the same costs."""
    costs = result["ret"] - result["ret_net"]
    ret = -result["ret"]
    ret_net = ret - costs
    return pd.DataFrame({"ret": ret, "ret_net": ret_net,
                         "equity": (1.0 + ret_net).cumprod()}, index=result.index)


def hold_positions(z: pd.Series, horizon: int, entry: float = ENTRY) -> pd.Series:
    """Go with the move on a fresh crossing of |z| = entry, hold `horizon`
    days, go flat; while in a trade, new crossings are ignored. Long the
    spread when it has just gone up, which in this backtest's convention is
    position +1."""
    zz = z.to_numpy(dtype=float)
    out = np.zeros(len(zz), dtype=int)
    i = 1
    while i < len(zz):
        fresh = (not np.isnan(zz[i]) and not np.isnan(zz[i - 1])
                 and abs(zz[i]) >= entry and abs(zz[i - 1]) < entry)
        if fresh:
            out[i:i + horizon] = int(np.sign(zz[i]))
            i += horizon
        else:
            i += 1
    return pd.Series(out, index=z.index, dtype=int)


def run_all(scan: pd.DataFrame, trading: pd.DataFrame, cost_bps: float = COST_BPS) -> dict:
    """One pass over every pair, all three trades at once. Returns per-pair
    stats and the daily return panels (gross and net) for each trade."""
    labels = ["revert", "mirrored"] + [f"hold{h}" for h in HORIZONS]
    gross = {k: {} for k in labels}
    net = {k: {} for k in labels}
    rows = []
    for row in scan.itertuples():
        y, x = trading[row.a], trading[row.b]
        key = f"{row.a}/{row.b}"
        z = rolling_zscore(y - row.beta * x, ZSCORE_WINDOW)
        results = {"revert": backtest(y, x, generate_positions(z), cost_bps)}
        results["mirrored"] = mirror(results["revert"])
        for h in HORIZONS:
            results[f"hold{h}"] = backtest(y, x, hold_positions(z, h), cost_bps)
        stats = {"a": row.a, "b": row.b, "same_sector": bool(row.same_sector),
                 "sector_a": row.sector_a}
        for k, r in results.items():
            gross[k][key] = r["ret"]
            net[k][key] = r["ret_net"]
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


def lumpiness(daily: pd.Series) -> dict:
    """How much of the book's gross P&L sits in its best few days. A trade
    paid by discrete events should have more of it there than a trade paid
    by a small drift."""
    d = daily.dropna()
    positive = d[d > 0].sort_values(ascending=False)
    top = positive.iloc[:max(1, int(round(0.05 * len(d))))].sum()
    return {"excess_kurtosis": float(d.kurt()),
            "top5pct_share_of_gains": float(top / positive.sum()) if positive.sum() > 0 else 0.0}


def verdicts(books: dict, lumps: dict) -> list[dict]:
    same_m, cross_m = books["same sector, mirrored"], books["cross sector, mirrored"]
    cross_r = books["cross sector, reverting"]
    shipped, everything = books["screened, reverting (shipped)"], books["everything, reverting"]
    by_sector = books["sign by sector"]
    same_20 = books["same sector, hold 20"]
    cross_holds = [books[f"cross sector, hold {h}"] for h in HORIZONS]
    return [
        {"claim": "same-sector mirrored: gross Sharpe > 0, net between -0.20 and +0.10",
         "measured": f"gross {same_m['gross_sharpe']:+.2f}, net {same_m['sharpe']:+.2f}",
         "held": same_m["gross_sharpe"] > 0 and -0.20 <= same_m["sharpe"] <= 0.10},
        {"claim": "cross-sector mirrored: net below the reverting cross book by more than 0.20",
         "measured": f"mirrored {cross_m['sharpe']:+.2f} vs reverting {cross_r['sharpe']:+.2f}",
         "held": cross_r["sharpe"] - cross_m["sharpe"] > 0.20},
        {"claim": "sign-by-sector beats the shipped book by more than 0.10 and beats everything",
         "measured": f"by sector {by_sector['sharpe']:+.2f}, shipped {shipped['sharpe']:+.2f}, "
                     f"everything {everything['sharpe']:+.2f}",
         "held": (by_sector["sharpe"] - shipped["sharpe"] > 0.10
                  and by_sector["sharpe"] > everything["sharpe"])},
        {"claim": "held 20 days: same-sector continuation positive gross and negative net; "
                  "cross-sector continuation negative gross at every horizon",
         "measured": f"same gross {same_20['gross_return']:+.2%} net {same_20['total_return']:+.2%}; "
                     f"cross gross " + ", ".join(f"{b['gross_return']:+.2%}" for b in cross_holds),
         "held": (same_20["gross_return"] > 0 and same_20["total_return"] < 0
                  and all(b["gross_return"] < 0 for b in cross_holds))},
        {"claim": "same-sector continuation is lumpier than cross-sector reversion: "
                  "higher kurtosis and a larger top-5%-of-days share of gains",
         "measured": f"kurtosis {lumps['same sector, mirrored']['excess_kurtosis']:.1f} vs "
                     f"{lumps['cross sector, reverting']['excess_kurtosis']:.1f}; top-5% share "
                     f"{lumps['same sector, mirrored']['top5pct_share_of_gains']:.1%} vs "
                     f"{lumps['cross sector, reverting']['top5pct_share_of_gains']:.1%}",
         "held": (lumps["same sector, mirrored"]["excess_kurtosis"]
                  > lumps["cross sector, reverting"]["excess_kurtosis"]
                  and lumps["same sector, mirrored"]["top5pct_share_of_gains"]
                  > lumps["cross sector, reverting"]["top5pct_share_of_gains"])},
    ]


def main() -> None:
    scan = pd.read_parquet(RESULTS / "scan.parquet")
    trading = pd.read_parquet(RESULTS / "trading.parquet")
    same = scan["same_sector"].to_numpy()
    passed = (scan["pvalue"] <= 0.05).to_numpy()
    print(f"{len(scan):,} pairs, {int(same.sum())} same-sector. Every pair, reverting and "
          f"continuing, {len(HORIZONS)} fixed horizons (the slow part)...")
    run = run_all(scan, trading)
    stats, gross, net = run["stats"], run["gross"], run["net"]

    everything = np.ones(len(scan), dtype=bool)
    books = {
        "screened, reverting (shipped)": book_row(gross["revert"], net["revert"], passed),
        "everything, reverting": book_row(gross["revert"], net["revert"], everything),
        "cross sector, reverting": book_row(gross["revert"], net["revert"], ~same),
        "same sector, reverting": book_row(gross["revert"], net["revert"], same),
        "same sector, mirrored": book_row(gross["mirrored"], net["mirrored"], same),
        "cross sector, mirrored": book_row(gross["mirrored"], net["mirrored"], ~same),
    }
    for h in HORIZONS:
        books[f"same sector, hold {h}"] = book_row(gross[f"hold{h}"], net[f"hold{h}"], same)
        books[f"cross sector, hold {h}"] = book_row(gross[f"hold{h}"], net[f"hold{h}"], ~same)
    # the reading as a book: each pair traded in the direction its sector says
    mixed_gross = pd.concat([gross["revert"].loc[:, ~same], gross["mirrored"].loc[:, same]], axis=1)
    mixed_net = pd.concat([net["revert"].loc[:, ~same], net["mirrored"].loc[:, same]], axis=1)
    books["sign by sector"] = book_row(mixed_gross, mixed_net, everything)
    books["sign by sector, screened"] = book_row(mixed_gross, mixed_net, passed)

    print(f"\n{'book':<32} {'pairs':>6} {'gross Sh':>9} {'gross ret':>10} {'net Sh':>7} {'net ret':>8}")
    for label, b in books.items():
        print(f"{label:<32} {b['pairs']:>6,} {b['gross_sharpe']:>+9.2f} {b['gross_return']:>+10.2%} "
              f"{b['sharpe']:>+7.2f} {b['total_return']:>+8.2%}")

    # per pair, the way the sector study read it
    per_pair = {}
    for label, col in (("reverting", "revert_sharpe"), ("mirrored", "mirrored_sharpe")):
        a, b = stats.loc[same, col].to_numpy(), stats.loc[~same, col].to_numpy()
        per_pair[label] = {"same_mean": float(a.mean()), "cross_mean": float(b.mean()),
                           "same_profitable": float((stats.loc[same, col.replace("sharpe", "return")] > 0).mean()),
                           "cross_profitable": float((stats.loc[~same, col.replace("sharpe", "return")] > 0).mean()),
                           **permutation_test(a, b)}
        p = per_pair[label]
        print(f"\nPer pair, {label}: mean net Sharpe same {p['same_mean']:+.3f} vs cross "
              f"{p['cross_mean']:+.3f} (p = {p['p_value']:.3f}); profitable {p['same_profitable']:.1%} "
              f"vs {p['cross_profitable']:.1%}")

    # mirrored same-sector book, sector by sector
    sectors = []
    for sector, idx in stats[same].groupby("sector_a").groups.items():
        mask = np.zeros(len(scan), dtype=bool)
        mask[np.asarray(idx)] = True
        if mask.sum() < 10:
            continue
        sectors.append({"sector": sector, **book_row(gross["mirrored"], net["mirrored"], mask)})
    sectors.sort(key=lambda r: -r["pairs"])
    print(f"\nSame-sector pairs mirrored, sector by sector:\n{'sector':<24} {'pairs':>6} "
          f"{'gross Sh':>9} {'net Sh':>7} {'net ret':>8}")
    for r in sectors:
        print(f"{r['sector']:<24} {r['pairs']:>6} {r['gross_sharpe']:>+9.2f} {r['sharpe']:>+7.2f} "
              f"{r['total_return']:>+8.2%}")

    lumps = {"same sector, mirrored": lumpiness(gross["mirrored"].loc[:, same].mean(axis=1)),
             "cross sector, reverting": lumpiness(gross["revert"].loc[:, ~same].mean(axis=1)),
             "same sector, reverting": lumpiness(gross["revert"].loc[:, same].mean(axis=1))}
    print("\nShape of the gross daily book P&L:")
    for label, l in lumps.items():
        print(f"  {label:<26} excess kurtosis {l['excess_kurtosis']:>6.1f}   best 5% of days = "
              f"{l['top5pct_share_of_gains']:.1%} of the gains")

    # how long the mirrored trade lasts: the window absorbing the move
    hold = []
    for row in scan[scan["same_sector"]].itertuples():
        z = rolling_zscore(trading[row.a] - row.beta * trading[row.b], ZSCORE_WINDOW)
        pos = generate_positions(z).to_numpy()
        runs = np.diff(np.flatnonzero(np.diff(np.r_[0, (pos != 0).astype(int), 0])))[::2]
        hold.extend(runs.tolist())
    holding = {"mean_days": float(np.mean(hold)), "median_days": float(np.median(hold)),
               "trades": len(hold)}
    print(f"\nA same-sector trade (either sign) lasts {holding['median_days']:.0f} days at the median, "
          f"{holding['mean_days']:.1f} on average, over {holding['trades']:,} trades.")

    scored = verdicts(books, lumps)
    print("\nPredictions, scored:")
    for v in scored:
        print(f"  [{'held' if v['held'] else 'FAILED'}] {v['claim']}\n         measured: {v['measured']}")

    payload = {"books": books, "per_pair": per_pair, "sectors": sectors, "lumpiness": lumps,
               "holding": holding, "verdicts": scored,
               "monthly": {label: (1.0 + series).resample("ME").prod().sub(1.0).round(6).to_dict()
                           for label, series in (
                               ("same sector, mirrored", net["mirrored"].loc[:, same].mean(axis=1)),
                               ("cross sector, reverting", net["revert"].loc[:, ~same].mean(axis=1)),
                               ("sign by sector", mixed_net.mean(axis=1)))}}
    payload["monthly"] = {k: {str(d.date()): v for d, v in m.items()} for k, m in payload["monthly"].items()}
    with open(OUT, "w") as fh:
        json.dump(payload, fh, indent=1)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
