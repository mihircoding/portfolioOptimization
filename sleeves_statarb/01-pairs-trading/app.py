

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import scan
from universe import sector

ROOT = Path(__file__).resolve().parent
RESULTS = ROOT / "results"

st.set_page_config(page_title="Pairs Trading Lab", page_icon=":chart:", layout="wide")


# Data

@st.cache_data
def load():
    if not (RESULTS / "meta.json").exists():
        return None
    return (
        pd.read_parquet(RESULTS / "scan.parquet"),
        pd.read_parquet(RESULTS / "results.parquet"),
        pd.read_parquet(RESULTS / "formation.parquet"),
        pd.read_parquet(RESULTS / "trading.parquet"),
        json.loads((RESULTS / "meta.json").read_text()),
    )


data = load()
if data is None:
    st.title("Pairs Trading Lab")
    st.error("No results found. Run `python scan.py` first — it takes about a minute.")
    st.stop()

full_scan, results, formation, trading, meta = data

st.title("Pairs Trading Lab")
st.caption(
    f"S&P 100 · formation {meta['formation'][0][:4]}–{meta['formation'][1][:4]} "
    f"· traded out of sample {meta['trading'][0][:4]}–{meta['trading'][1][:4]} "
    f"· {meta['zscore_window']}-day z-score, entry {meta['entry']}, exit {meta['exit']}, "
    f"{meta['cost_bps']:.0f} bps costs"
)

overview, verdict, explorer, table = st.tabs(
    ["Overview", "Did the screen work?", "Pair explorer", "All results"]
)


# Overview

with overview:
    c = st.columns(4)
    c[0].metric("Pairs tested", f"{meta['n_tests']:,}")
    c[1].metric("Passed at 5%", f"{meta['n_passed']:,}",
                f"{meta['n_passed'] - meta['expected_false_positives']:+,.0f} vs chance")
    c[2].metric("Expected by chance", f"{meta['expected_false_positives']:,.0f}")
    c[3].metric("Survive Bonferroni", f"{meta['n_bonferroni']:,}")

    st.markdown(
        f"""
With {meta['n_tickers']} tickers you run **{meta['n_tests']:,} tests**. At a 5% threshold about
**{meta['expected_false_positives']:.0f} pairs pass on pure luck** even if nothing is genuinely
related — so the raw count of survivors tells you almost nothing on its own.

{meta['n_passed']:,} pairs passed, which is more than chance alone predicts. That is not evidence
of {meta['n_passed']:,} tradeable relationships: large-cap stocks share market-wide factors, so
many pairs co-move enough to fool a test that was designed assuming independence.

The strict correction for running {meta['n_tests']:,} tests is Bonferroni:
p ≤ 0.05/{meta['n_tests']:,} = **{meta['bonferroni_threshold']:.2e}**.
**{meta['n_bonferroni']} pair clears it.**
"""
    )

    st.warning(
        "**Survivorship bias.** This is the S&P 100 as it stands *today*. Membership is awarded "
        "for having already grown large, so testing it from 2013 asks how today's winners behaved "
        "on their way to winning. Firms that were in the index and then collapsed or were acquired "
        "are simply absent. Fixing this needs a point-in-time constituent list, which free data "
        "cannot provide."
    )

    st.subheader("Where the p-values landed")
    fig, ax = plt.subplots(figsize=(9, 3.2))
    ax.hist(full_scan["pvalue"], bins=60, color="#4C72B0", edgecolor="white", linewidth=0.4)
    ax.axvline(0.05, color="#C44E52", ls="--", lw=1.5, label="0.05 threshold")
    ax.set_xlabel("Engle-Granger p-value (formation window)")
    ax.set_ylabel("pairs")
    ax.legend(frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    st.pyplot(fig, width="stretch")
    st.caption(
        "Under a true null the p-values would be flat across [0,1]. The pile-up near zero is "
        "partly genuine co-movement and partly the ADF test being too permissive when beta is "
        "estimated rather than known."
    )


# Did the screen work?

with verdict:
    st.subheader("The honest answer: no")

    win = (results["total_return"] > 0).mean()
    still = (results["pvalue_oos"] < 0.05).mean()
    c = st.columns(4)
    c[0].metric("Mean OOS Sharpe", f"{results['sharpe'].mean():+.3f}")
    c[1].metric("Profitable", f"{win:.1%}")
    c[2].metric("Sharpe > 1", f"{(results['sharpe'] > 1).sum()} of {len(results)}")
    c[3].metric("Still cointegrated OOS", f"{still:.1%}")

    st.markdown(
        f"""
All {len(results):,} pairs that passed the screen were then traded on data they had never been
selected on. The average Sharpe was **{results['sharpe'].mean():+.3f}** — indistinguishable from
zero — and only **{win:.1%} made money**, which is worse than a coin flip. The average maximum
drawdown was **{results['max_drawdown'].mean():.1%}**.

Only **{still:.1%}** of them were still cointegrated in the trading window. Roughly three quarters
of the relationships the screen found had stopped existing by the time you traded them.
"""
    )

    left, right = st.columns(2)

    with left:
        st.markdown("**Distribution of out-of-sample Sharpe**")
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.hist(results["sharpe"], bins=50, color="#4C72B0", edgecolor="white", linewidth=0.4)
        ax.axvline(0, color="#555", lw=1)
        ax.axvline(results["sharpe"].mean(), color="#C44E52", ls="--", lw=1.5,
                   label=f"mean {results['sharpe'].mean():+.3f}")
        ax.set_xlabel("Sharpe ratio")
        ax.set_ylabel("pairs")
        ax.legend(frameon=False)
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width="stretch")
        sd = results["sharpe"].std()
        st.caption(
            f"Centred on zero with standard deviation {sd:.2f}. A handful of winners is exactly "
            f"what a distribution this shape produces by chance — the tail is not a discovery."
        )

    with right:
        st.markdown("**Does a smaller p-value predict a better trade?**")
        q = results.assign(bucket=pd.qcut(results["pvalue"], 5,
                                          labels=["best p", "2nd", "3rd", "4th", "worst p"]))
        by = q.groupby("bucket", observed=True)["sharpe"].mean()
        fig, ax = plt.subplots(figsize=(6, 3.6))
        ax.bar(by.index.astype(str), by.values,
               color=["#55A868" if v > 0 else "#C44E52" for v in by.values])
        ax.axhline(0, color="#555", lw=1)
        ax.set_ylabel("mean out-of-sample Sharpe")
        ax.spines[["top", "right"]].set_visible(False)
        st.pyplot(fig, width="stretch")
        st.caption(
            f"Correlation between in-sample p-value and out-of-sample Sharpe: "
            f"**{results['pvalue'].corr(results['sharpe']):+.3f}**. Essentially none. Ranking "
            f"candidates by p-value does not sort them by how well they will actually trade."
        )

    st.subheader("Same sector vs cross sector")
    grp = results.groupby("same_sector").agg(
        pairs=("sharpe", "size"),
        mean_sharpe=("sharpe", "mean"),
        median_sharpe=("sharpe", "median"),
        pct_profitable=("total_return", lambda s: (s > 0).mean()),
        mean_return=("total_return", "mean"),
        still_cointegrated=("pvalue_oos", lambda s: (s < 0.05).mean()),
    ).rename(index={True: "Same sector", False: "Cross sector"})

    grp_display = pd.DataFrame({
        "Pairs": grp["pairs"].astype(int),
        "Mean Sharpe": grp["mean_sharpe"].map("{:+.3f}".format),
        "Median Sharpe": grp["median_sharpe"].map("{:+.3f}".format),
        "Profitable": grp["pct_profitable"].map("{:.1%}".format),
        "Mean return": grp["mean_return"].map("{:+.2%}".format),
        "Still cointegrated": grp["still_cointegrated"].map("{:.1%}".format),
    })
    st.dataframe(grp_display, width="stretch")
    st.info(
        "**This did not go the way the 12-ticker experiment suggested.** There, the one pair with "
        "a real economic story (Mastercard/Visa) beat the coincidence that topped the p-value "
        "ranking, and the tidy conclusion was 'require an economic reason'. At 100 tickers, "
        "same-sector pairs did *worse* than cross-sector ones. Being in the same GICS sector is a "
        "blunt proxy for economic linkage — it lumps Berkshire Hathaway with Visa — and 119 "
        "same-sector pairs is a small sample. The honest reading is that this crude filter does "
        "not rescue the strategy, not that economic reasoning is worthless."
    )


# Pair explorer

with explorer:
    st.subheader("Inspect any pair")

    ranked = results.sort_values("sharpe", ascending=False)
    labels = [f"{r.a}/{r.b}   Sharpe {r.sharpe:+.2f}   p={r.pvalue:.4f}"
              for r in ranked.itertuples()]
    choice = st.selectbox("Pair (sorted by out-of-sample Sharpe)", range(len(labels)),
                          format_func=lambda i: labels[i])
    row = ranked.iloc[choice]
    a, b = row["a"], row["b"]

    st.caption(f"{a} ({sector(a)})  vs  {b} ({sector(b)})")

    p = st.columns(4)
    window = p[0].slider("z-score window", 20, 250, int(meta["zscore_window"]), 5)
    entry = p[1].slider("entry |z|", 0.5, 4.0, float(meta["entry"]), 0.1)
    exit_ = p[2].slider("exit |z|", 0.0, 2.0, float(meta["exit"]), 0.1)
    cost = p[3].slider("cost (bps)", 0.0, 25.0, float(meta["cost_bps"]), 0.5)

    y, x = trading[a], trading[b]
    beta = float(row["beta"])   # estimated on the FORMATION window only
    result, z, pos, stats = scan.run_pair(y, x, beta, window, entry, exit_, cost)

    m = st.columns(5)
    m[0].metric("Total return", f"{stats['total_return']:+.2%}")
    m[1].metric("Sharpe", f"{stats['sharpe']:+.3f}")
    m[2].metric("Max drawdown", f"{stats['max_drawdown']:.2%}")
    m[3].metric("Round trips", stats["n_round_trips"])
    m[4].metric("Hedge ratio β", f"{beta:.3f}")

    fig, axes = plt.subplots(3, 1, figsize=(11, 8), sharex=True,
                             gridspec_kw={"height_ratios": [2, 1, 2]})

    axes[0].plot(z.index, z.values, lw=0.9, color="#4C72B0")
    axes[0].axhline(entry, ls="--", c="#C44E52", lw=1)
    axes[0].axhline(-entry, ls="--", c="#C44E52", lw=1)
    axes[0].axhline(exit_, ls=":", c="#55A868", lw=1)
    axes[0].axhline(-exit_, ls=":", c="#55A868", lw=1)
    axes[0].axhline(0, c="#999", lw=0.8)
    axes[0].set_ylabel("z-score")
    axes[0].set_title(f"{a}/{b} — rolling z-score of the spread")

    axes[1].fill_between(pos.index, pos.values, step="post", alpha=0.6, color="#8172B2")
    axes[1].set_ylabel("position")
    axes[1].set_yticks([-1, 0, 1])

    axes[2].plot(result.index, result["equity"], lw=1.2, color="#55A868")
    axes[2].axhline(1.0, ls="--", c="#999", lw=1)
    axes[2].set_ylabel("equity (net)")
    axes[2].set_title("Growth of $1, net of costs")

    for ax in axes:
        ax.spines[["top", "right"]].set_visible(False)
    plt.tight_layout()
    st.pyplot(fig, width="stretch")

    d = st.columns(2)
    d[0].metric("In-sample p (formation)", f"{row['pvalue']:.4g}")
    d[1].metric("Out-of-sample p (trading)", f"{row['pvalue_oos']:.4g}",
                "still cointegrated" if row["pvalue_oos"] < 0.05 else "relationship broke",
                delta_color="normal" if row["pvalue_oos"] < 0.05 else "inverse")

    st.caption(
        "β is fixed at its formation-window value and never re-estimated on the trading window — "
        "re-fitting it here would leak the future into your position sizes. Moving these sliders "
        "re-runs the backtest, which is itself a demonstration of the problem: hunt long enough "
        "for parameters that make one pair look good and you are overfitting in real time."
    )


# All results

with table:
    st.subheader("Every pair that passed the screen")

    f = st.columns(4)
    which = f[0].selectbox("Sector relationship", ["All", "Same sector only", "Cross sector only"])
    min_trips = f[1].number_input("Min round trips", 0, 100, 0)
    only_oos = f[2].checkbox("Still cointegrated out of sample")
    sort_by = f[3].selectbox("Sort by", ["sharpe", "total_return", "pvalue", "max_drawdown"])

    view = results.copy()
    if which == "Same sector only":
        view = view[view["same_sector"]]
    elif which == "Cross sector only":
        view = view[~view["same_sector"]]
    if only_oos:
        view = view[view["pvalue_oos"] < 0.05]
    view = view[view["n_round_trips"] >= min_trips]
    view = view.sort_values(sort_by, ascending=sort_by in ("pvalue", "max_drawdown"))

    st.caption(f"{len(view):,} pairs shown")
    display = pd.DataFrame({
        "A": view["a"], "B": view["b"],
        "Sector A": view["sector_a"], "Sector B": view["sector_b"],
        "Same sector": view["same_sector"],
        "Beta": view["beta"].round(3),
        "p (in-sample)": view["pvalue"],
        "p (out-of-sample)": view["pvalue_oos"],
        "Return %": (view["total_return"] * 100).round(2),
        "Sharpe": view["sharpe"].round(3),
        "Max DD %": (view["max_drawdown"] * 100).round(2),
        "Trips": view["n_round_trips"],
    })
    st.dataframe(
        display, width="stretch", height=520, hide_index=True,
        column_config={
            "p (in-sample)": st.column_config.NumberColumn(format="%.4g"),
            "p (out-of-sample)": st.column_config.NumberColumn(format="%.4g"),
            "Return %": st.column_config.NumberColumn(format="%+.2f"),
            "Sharpe": st.column_config.NumberColumn(format="%+.3f"),
            "Max DD %": st.column_config.NumberColumn(format="%.2f"),
        },
    )
    st.download_button("Download as CSV", view.to_csv(index=False),
                       "pairs_results.csv", "text/csv")
