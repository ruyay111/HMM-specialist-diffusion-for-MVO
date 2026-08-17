"""This module contains the main functions for performing the portfolio evaluation experiments.
The functions are:
- portfolio_experiment: evaluates mean-variance portfolio performance over a gridof n_synth_series
- portfolio_experiment_shrinkage: same as portfolio_experiment but uses shrinkage estimator for covariance
- plot_metric_curve: plots metric curves
"""
import cvxpy as cp
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from typing import Dict, List, Optional, Sequence, Tuple



from portfolio_core import (
    pct_returns,
    mean_var_weights,
    mean_var_weights_shrinkage,
    mix_train_matrix,
    collapse_weights,
    summarize_portfolio,
    LOOKBACK_DAYS,
    INVESTMENT_WINDOW,
)




#----- portfolio evaluation --------------------------------

def portfolio_experiment(
    real_prices: pd.DataFrame,
    synth_pools: dict,
    asset_cols: List[str],
    mix_grid: Sequence[int] = (0, 1, 2, 5, 10, 15, 20),
    risk_free: float = 0.0,
    allow_short: bool = False,
    lookback_window: int = LOOKBACK_DAYS,
    investment_window: int = INVESTMENT_WINDOW,
    seed: int = 42,
) -> pd.DataFrame:

    '''This function evaluates mean variance portfolio performance over a grid of n_synth_series values.
    For each n_synth_series in mix_grid, it constructs a training matrix and computes mean-variance weights,
    and evaluates the out-of-sample performance on a test set of real returns.

    Parameters:
    a: real_prices: DataFrame of asset prices
    b: synth_pools: dict of synthetic return DataFrames for each asset
    c: asset_cols: list of asset column names to use from real_prices and synth_pools
    d: mix_grid: number of synthetic series to include in training data 
    e: risk_free: annual risk-free rate
    f: allow_short: true/false short or not 
    g: lookback_window: number of days to use for the training period 
    h: investment_window: number of days to use for the test period 
    i: seed: chosen seed
    '''
    real_rets = pct_returns(real_prices[asset_cols])

    if lookback_window + investment_window > len(real_rets):
        raise ValueError("lookback_window + investment_window exceeds data length")

    real_train = real_rets.iloc[:lookback_window].copy()
    real_test = real_rets.iloc[lookback_window: lookback_window + investment_window].copy()

    results = []
    for n_synth in mix_grid:
        use_train, n_draw = mix_train_matrix(
            real_train.values, synth_pools, asset_cols,
            n_synth_series=int(n_synth), seed=seed,
            train_index=real_train.index,
            prefer_stress=True,
        )
        w = mean_var_weights(use_train, allow_short=allow_short)
        w = collapse_weights(w, n_assets=len(asset_cols), n_draw=n_draw)

        port_ret = pd.Series(real_test.values @ w, index=real_test.index)
        row = summarize_portfolio(port_ret, rf=risk_free)
        row["n_synth_series"] = int(n_synth)
        row["max_weight_used"] = float(np.max(np.abs(w)))
        results.append(row)

    return pd.DataFrame(results)



def portfolio_experiment_shrinkage(
    real_prices: pd.DataFrame,
    synth_pools: dict,
    asset_cols: List[str],
    mix_grid: Sequence[int] = (0, 1, 2, 5, 10, 15, 20),
    risk_free: float = 0.0,
    allow_short: bool = False,
    lookback_window: int = LOOKBACK_DAYS,
    investment_window: int = INVESTMENT_WINDOW,
    seed: int = 42,
) -> pd.DataFrame:
    
    """This is the same as portfolio_experiment but uses
    mean_var_weights_shrinkage instead of mean_var_weights."""

    real_rets = pct_returns(real_prices[asset_cols])

    if lookback_window + investment_window > len(real_rets):
        raise ValueError("lookback_window + investment_window exceeds data length")

    real_train = real_rets.iloc[:lookback_window].copy()
    real_test = real_rets.iloc[lookback_window: lookback_window + investment_window].copy()

    results = []
    for n_synth in mix_grid:
        use_train, n_draw = mix_train_matrix(
            real_train.values, synth_pools, asset_cols,
            n_synth_series=int(n_synth), seed=seed,
            train_index=real_train.index,
            prefer_stress=True,
        )
        w = mean_var_weights_shrinkage(use_train, allow_short=allow_short)
        w = collapse_weights(w, n_assets=len(asset_cols), n_draw=n_draw)

        port_ret = pd.Series(real_test.values @ w, index=real_test.index)
        row = summarize_portfolio(port_ret, rf=risk_free)
        row["n_synth_series"] = int(n_synth)
        row["max_weight_used"] = float(np.max(np.abs(w)))
        results.append(row)

    return pd.DataFrame(results)

# ----- vizualisation --------------------------------


def plot_metric_curve(
    results: pd.DataFrame,
    metric: str = "sharpe",
    spy_col: str | None = None,
    title: str | None = None,
    estimator: str = "mean",          
    confidence_bands: bool = True,
    ax=None,
):
    """This function plots the mean out-of-sample metric ratio vs n_synth_series
      for calm and crisis regimes."""
    
    """Parameters:
    a: results: DataFrame with columns "regime", "n_synth_series", "sharpe"
    b: spy_col: column name for SPY Sharpe ratio in results
    c: title: plot title
    d: estimator: "mean" or "median" 
    e: confidence_bands: whether to show +-1 SE bands around the mean
    f: ax: optional matplotlib axis to plot on
    """
    

    if ax is None:
        _, ax = plt.subplots(figsize=(8, 5))

    colors = {"calm": "steelblue", "crisis": "tomato"}
    agg = (results.groupby(["regime", "n_synth_series"])[metric]
                  .agg([estimator, "sem"]).reset_index())

    for regime, grp in agg.groupby("regime"):
        grp = grp.sort_values("n_synth_series")
        c = colors.get(regime, "green")
        ax.plot(grp["n_synth_series"], grp[estimator], marker="o", color=c, label=regime)
        if confidence_bands and estimator == "mean":
            ax.fill_between(grp["n_synth_series"],
                            grp["mean"] - grp["sem"],
                            grp["mean"] + grp["sem"],
                            alpha=0.15, color=c)

    if spy_col and spy_col in results.columns:
        spy_by_regime = results.groupby("regime")[spy_col].agg(estimator)
        for regime, c in colors.items():
            if regime in spy_by_regime.index:
                v = float(spy_by_regime[regime])
                ax.axhline(v, linestyle="--", color=c, alpha=0.5,
                           label=f"SPY {regime} ({v:.2f})")

    band = " (+-1 SE)" if confidence_bands and estimator == "mean" else ""
    ax.set_xlabel("N Synthetic Series per Asset")
    ax.set_ylabel(f"{estimator.title()} Out-of-Sample {metric.title()}{band}")
    ax.set_title(title or f"{metric.title()} vs. N Synthetic Series")
    ax.legend()
    return ax




