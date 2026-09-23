"""Builds docs/data.js for the GitHub Pages site.

Runs the same analysis run_optimization.py prints and dumps it as JSON, so
the site charts the actual optimizer output instead of numbers copied into
HTML by hand. Re-run after changing anything in src/ and the site follows:

    python docs/build_data.py
"""

import json
import sys
from pathlib import Path

import numpy as np
import yfinance as yf

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import factor_study
from factor_study import k_sweep_estimators
from run_optimization import (ALT_ANCHORS, END, ETF_BAND, LOOKBACK_YEARS, MARKET_SHARPE,
                              START, UNIVERSE, build_portfolios, cost_section,
                              market_weights, walk_forward)
from src.black_litterman import equilibrium_returns, implied_risk_aversion
from src.costs import COST_LEVELS_BPS, DEFAULT_COST_BPS
from src.cvar import cvar_of_weights, var_of_weights
from src.frontier import efficient_frontier
from src.optimizer import portfolio_performance
from src.factor_model import marchenko_pastur_edge, num_significant_factors
from src.returns import annualized_cov, annualized_mean, daily_returns
from src.risk_parity import risk_contributions

OUT = Path(__file__).resolve().parent / "data.js"


def main():
    print("downloading prices...")
    prices = yf.download(UNIVERSE, start=START, end=END, auto_adjust=True,
                         progress=False)["Close"].dropna()[UNIVERSE]

    mu = annualized_mean(prices).values
    cov = annualized_cov(prices).values
    rets = daily_returns(prices).values

    w_market = market_weights(UNIVERSE)
    market_vol = float(np.sqrt(w_market @ cov @ w_market))
    delta = implied_risk_aversion(MARKET_SHARPE * market_vol, market_vol ** 2)
    pi = equilibrium_returns(cov, w_market, delta)

    print("in-sample portfolios...")
    portfolios = build_portfolios(mu, cov, rets, w_market)

    in_sample = []
    for name, w in portfolios.items():
        ret, vol, sharpe = portfolio_performance(w, mu, cov)
        in_sample.append({
            "name": name,
            "ret": round(ret, 4), "vol": round(vol, 4), "sharpe": round(sharpe, 3),
            "weights": [round(float(x), 4) for x in w],
            "rc": [round(float(x), 4) for x in risk_contributions(w, cov)],
            "var95": round(var_of_weights(w, rets, 0.95), 5),
            "cvar95": round(cvar_of_weights(w, rets, 0.95), 5),
        })

    print("efficient frontier...")
    ef = efficient_frontier(mu, cov, n_points=60)
    frontier = [[round(float(v), 5), round(float(r), 5)]
                for v, r in zip(ef["volatility"], ef["target_return"])]

    print("walk-forward at 2/3/5/7y...")
    lookbacks = {}
    panel_3y = None
    for lb in (2, 3, 5, 7):
        # verbose so a failed solve is named in the build log rather than
        # silently shortening a column of the published table
        panel, summary = walk_forward(prices, lookback=lb, verbose=True,
                                      w_market=w_market)
        lookbacks[lb] = [{"name": r["portfolio"], "cagr": round(r["cagr"], 4),
                          "vol": round(r["vol"], 4), "sharpe": round(r["sharpe"], 3),
                          "worst": round(r["worst"], 4),
                          "turnover": round(r["turnover"], 4),
                          "years": int(r["years"])}
                         for _, r in summary.iterrows()]
        if lb == LOOKBACK_YEARS:
            wide = panel.pivot(index="year", columns="portfolio", values="ret")
            cumulative = (1 + wide).cumprod()
            panel_3y = {
                "years": [int(y) for y in cumulative.index],
                "series": {c: [round(float(v), 4) for v in cumulative[c]]
                           for c in cumulative.columns},
            }

    print("anchor sensitivity...")
    anchors = dict(ALT_ANCHORS)
    anchors["Fund AUM (65% SPY)"] = w_market
    anchor_rows = []
    for label, anchor in anchors.items():
        _, summary = walk_forward(prices, verbose=False, w_market=anchor)
        by_name = summary.set_index("portfolio")["sharpe"]
        anchor_rows.append({
            "label": label,
            "weights": [round(float(x), 4) for x in anchor],
            "no_views": round(float(by_name.get("Black-Litterman (no views)", float("nan"))), 3),
            "momentum": round(float(by_name.get("Black-Litterman (momentum)", float("nan"))), 3),
        })

    print("50-asset risk-model study (slow)...")
    large = factor_study.load(factor_study.UNIVERSE)
    large_rets = daily_returns(large).values
    n_large = large.shape[1]

    def table_rows(table):
        return [{"name": r["estimator"],
                 "predicted": round(r["predicted vol"], 4),
                 "realized": round(r["realized vol"], 4),
                 "ratio": round(r["ratio"], 3),
                 "turnover": round(r["turnover"], 3),
                 "short": round(r["short"], 3)}
                for _, r in table.iterrows()]

    windows, tables = {}, {}
    for w in factor_study.WINDOWS:
        tables[w] = factor_study.backtest(large, w)
        windows[w] = {
            "rows": table_rows(tables[w]),
            "edge": round(marchenko_pastur_edge(n_large, w), 3),
            "k": num_significant_factors(large_rets[-w:]),
        }
    k_sweep = table_rows(factor_study.backtest(
        large, 1008, build=k_sweep_estimators([1, 2, 3, 5, 10, 20, 50])))
    long_only_table = factor_study.backtest(large, 252, long_only=True)
    long_only = table_rows(long_only_table)

    print("transaction costs and no-trade bands...")
    levels = (0,) + tuple(COST_LEVELS_BPS)

    def cost_rows(table):
        return [{"name": r["estimator"], "band": r["band"],
                 "turnover": round(r["turnover"], 4),
                 "vol": round(r["realized vol"], 4),
                 "sharpe": {str(c): round(r[f"sharpe {c}"], 3) for c in levels},
                 "drag": {str(c): round(r[f"drag {c}"], 5) for c in COST_LEVELS_BPS}}
                for _, r in table.iterrows()]

    small = factor_study.load(factor_study.SMALL_UNIVERSE)
    cost_large = cost_rows(factor_study.cost_tables(
        large, tables[252], long_only_table, small, factor_study.backtest(small, 252)))
    cost_sweep = cost_rows(factor_study.cost_study(
        large, tables[252], bands=factor_study.BAND_SWEEP, names=["Sample"]))
    cost_etf = [{"name": r["portfolio"], "band": r["band"],
                 "turnover": round(r["turnover"], 4),
                 "sharpe": {"0": round(r["gross sharpe"], 3),
                            **{str(c): round(r[f"net sharpe {c}"], 3)
                               for c in COST_LEVELS_BPS}},
                 "drag": {str(c): round(r[f"drag {c}"], 6) for c in COST_LEVELS_BPS}}
                for r in cost_section(prices, w_market, verbose=False)]

    data = {
        "meta": {"universe": UNIVERSE, "start": str(prices.index[0].date()),
                 "end": str(prices.index[-1].date()), "days": len(prices),
                 "lookback": LOOKBACK_YEARS},
        "assets": [{"ticker": t, "ret": round(float(mu[i]), 4),
                    "vol": round(float(np.sqrt(cov[i, i])), 4)}
                   for i, t in enumerate(UNIVERSE)],
        "in_sample": in_sample,
        "frontier": frontier,
        "walk_forward": lookbacks,
        "growth": panel_3y,
        "factor": {
            "n_assets": n_large, "tickers": factor_study.UNIVERSE,
            "parameters": n_large * (n_large + 1) // 2,
            "hold_days": factor_study.HOLD_DAYS,
            "windows": {str(k): v for k, v in windows.items()},
            "k_sweep": k_sweep,
            "long_only": long_only,
        },
        "costs": {
            "levels": list(COST_LEVELS_BPS), "default": DEFAULT_COST_BPS,
            "bands": factor_study.BANDS, "etf_band": ETF_BAND,
            "large": cost_large, "sweep": cost_sweep, "etf": cost_etf,
        },
        "bl": {
            "market_weights": [round(float(x), 4) for x in w_market],
            "delta": round(delta, 3),
            "market_vol": round(market_vol, 4),
            "equilibrium": [round(float(x), 4) for x in pi],
            "sample_mean": [round(float(x), 4) for x in mu],
            "anchors": anchor_rows,
        },
    }

    OUT.write_text("window.DATA = " + json.dumps(data, separators=(",", ":")) + ";\n",
                   encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1024:.0f} KB)")
    for row in lookbacks[LOOKBACK_YEARS]:
        print(f"  {row['name']:<26} OOS sharpe {row['sharpe']:>5.2f}")


if __name__ == "__main__":
    main()
