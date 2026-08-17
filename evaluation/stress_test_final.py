from __future__ import annotations


"""This module contains the main functions for performing the stress tests on the portfolio.
The functions are:
- stress_test_by_regime: performs the regime stress test
- shock_test: performs the shock stress test
- plot_metric_curve: visualizes the results of the stress tests by plotting the metric curves
"""

import numpy as np
import pandas as pd
from scipy.stats import t
from typing import Dict, List, Optional, Sequence, Tuple

from portfolio_core import (   
    pct_returns,
    time_split,
    regime_by_vol,
    regime_by_drawdown,
    mean_var_weights,
    mix_train_matrix,     
    collapse_weights,
    summarize_portfolio,
    LOOKBACK_DAYS,       
    INVESTMENT_WINDOW    
)


# ---------------------- 2008 data ----------------------
# I will use these for the "crash_2008" shock scenario,
# it simulates a shock with similar characteristics to the S&P in the 2008 financial crisis.
_SP500_2008_RETURNS = None
_SP500_2008_GARCH_PARAMS = None

def _get_2008_returns():
    """This function downloads the S&P data for 2007-2009 and calculates the daily log returns
    """
    global _SP500_2008_RETURNS
    if _SP500_2008_RETURNS is None:
        import yfinance as yf

        sp500 = yf.download('^GSPC', start='2007-01-01', end='2009-12-31', progress=False)
        _SP500_2008_RETURNS = np.log(sp500['Close'] / sp500['Close'].shift(1)).dropna()
    return _SP500_2008_RETURNS

def _get_2008_garch_params():
    """This function fits a GARCH(1,1) model to the S&P returns during the 2008 crisis"""
    global _SP500_2008_GARCH_PARAMS
    if _SP500_2008_GARCH_PARAMS is None:
        from arch import arch_model

        returns_2008 = _get_2008_returns()
        returns_acute = returns_2008['2008-09-01':'2008-11-30']
        model = arch_model(returns_acute * 100, vol='Garch', p=1, q=1)
        result = model.fit(disp='off')
        _SP500_2008_GARCH_PARAMS = (
            result.params['omega'],
            result.params['alpha[1]'],
            result.params['beta[1]']
        )
    return _SP500_2008_GARCH_PARAMS


# ---------------------- shocks ----------------------

def apply_shock_to_returns(
    rets: pd.DataFrame,
    shock_type: str,
    seed: int = 42,
    shock_size: float = 0.10,
    vol_scale: float = 1.5,
    corr_inflate: float = 1.3,
) -> pd.DataFrame:
    
    """This function applies the different shock scenarios to the returns dataframe. 
    The shock_type parameter determines which shock is applied,
    The function returns a new DataFrame with the shocked returns.
    Important to note the type of shocks are:
    Volatility scaling, negative return shock, correlation inflation,
    whipsaw, slow-bleed and a shock simulating the 2008 financial crisis."""

    """Parameters:
    a: rets: DataFrame of returns
    b: shock_type: type of shock to apply
    c: seed: chosen seed
    d: shock_size: size of the negative return shock 
    e: vol_scale: scaling factor for volatility
    f: corr_inflate: factor to inflate correlations 
    """

    rng = np.random.default_rng(seed)
    R = rets.copy()

    if shock_type == "NEG_RETURN_SHOCK":
        t_idx = int(rng.integers(0, len(R)))
        R.iloc[t_idx, :] = R.iloc[t_idx, :] - shock_size
        return R

    if shock_type == "VOL_SCALE":
        return R * vol_scale

    if shock_type == "CORR_INFLATE":
        a = 1.0 / corr_inflate
        f = R.mean(axis=1)
        return a * R + (1 - a) * f.values.reshape(-1, 1)

    if shock_type == "WHIPSAW":
        t_idx = int(rng.integers(0, len(R) - 23))
        adj_positive = rng.uniform(0.001, 0.004, size=15)
        R.iloc[t_idx:t_idx+15, :] = R.iloc[t_idx:t_idx+15, :].to_numpy() + adj_positive[:, None]
        adj_positive = adj_positive + 1
        comp_push = adj_positive.cumprod()[-1]
        d = 0.05
        reversal_target = 1 - d
        reversal_push = reversal_target / comp_push
        k, theta_g = 2, 1
        weights = rng.gamma(shape=k, scale=theta_g, size=7)
        scaling_factor = -np.log(reversal_push) / np.sum(weights)
        log_return = -scaling_factor * weights
        adj_negative = np.exp(log_return) - 1
        R.iloc[t_idx+15:t_idx+22, :] = R.iloc[t_idx+15:t_idx+22, :].to_numpy() + adj_negative[:, None]
        return R

    if shock_type == "SLOW_BLEED":
        t_idx = int(rng.integers(0, len(R) - 121))
        slow_bleed_stress = rng.uniform(0.0005, 0.002, size=120)
        days = rng.choice(len(slow_bleed_stress), size=8, replace=False)
        slow_bleed_stress[days] = -rng.uniform(0.001, 0.005, size=8)
        R.iloc[t_idx:t_idx+120, :] = R.iloc[t_idx:t_idx+120, :].to_numpy() - slow_bleed_stress[:, None]
        return R

    if shock_type == "crash_2008":
        omega, alpha, beta = _get_2008_garch_params()
        pre_crisis_days, early_stress_days, crash_days, recovery_days = 60, 60, 60, 120
        pre_crisis = rng.normal(0.0006, 0.008, size=pre_crisis_days)
        early_stress = rng.normal(-0.002, 0.02, size=early_stress_days)
        base_vol = 0.05
        vol = base_vol
        crash_rets = []
        for _ in range(crash_days):
            cr = t.rvs(df=3, loc=-0.008, scale=vol, random_state=rng)
            vol = np.sqrt(beta * vol**2 + alpha * cr**2 + omega * base_vol**2)
            crash_rets.append(cr)
        crash_rets = np.array(crash_rets)
        recovery = rng.normal(0.002, 0.025, size=recovery_days)
        crisis_path = np.concatenate([pre_crisis, early_stress, crash_rets, recovery])
        crisis_path_simple = np.exp(crisis_path) - 1
        R.iloc[:, :] = (1 + R.to_numpy()) * (1 + crisis_path_simple[:, None]) - 1
        return R

    raise ValueError(f"Unknown shock_type: {shock_type}")




#----- actual stress tests --------------------------------
def stress_test_by_regime(
    real_prices: pd.DataFrame,
    synth_pools: dict,
    asset_cols: List[str],
    mix_grid: Sequence[int] = (0, 1, 2, 5, 10, 15, 20),
    lookback_window: int = LOOKBACK_DAYS,
    investment_window: int = INVESTMENT_WINDOW,
    allow_short: bool = False,
    seed: int = 42,
    regime_kind: str = "vol",
    rf: float = 0.0,
) -> pd.DataFrame:
    
    """This function performs the regime stress test. It takes in the real price data, 
    synthetic pools, and other parameters, and returns a DataFrame summarizing the performance
    of the portfolio across different regimes and different numbers of synthetic series
    included in the training data. """
    
    """Parameters:
    a: real_prices: DataFrame of real prices
    b: synth_pools: dictionary of synthetic returns
    c: asset_cols: list of asset column names 
    d: mix_grid: the number of synthetic series to include in training
    e: lookback_window: number of days used in the training period
    f: investment_window: number of days used in the testing period
    g: allow_short: true/false for short or not
    h: seed: chosen seed
    i: regime_kind: whether to define regimes by "vol" or "drawdown
    j: rf: annual risk-free rate
    """


    real_rets = pct_returns(real_prices[asset_cols])
    train_rets, test_rets = time_split(real_rets, lookback_window, investment_window)
    if regime_kind == "vol":
        regime = regime_by_vol(real_prices[asset_cols])
    elif regime_kind == "drawdown":
        regime = regime_by_drawdown(real_prices[asset_cols])
    else:
        raise ValueError("regime_kind must be 'vol' or 'drawdown'")
    rows: List[Dict] = []
    for pct in mix_grid:
        use_train, n_draw = mix_train_matrix(
            train_rets.values, synth_pools, asset_cols,
            n_synth_series=int(pct), seed=seed,
            train_index = train_rets.index
        )
        w = mean_var_weights(use_train, allow_short=allow_short)
        w = collapse_weights(w, n_assets=len(asset_cols), n_draw=n_draw)
        port_test = pd.Series(test_rets.values @ w, index=test_rets.index)
        base = summarize_portfolio(port_test, label="ALL", rf=rf)
        base.update({"n_synth_series": int(pct)})
        rows.append(base)
        common_idx = port_test.index.intersection(regime.index)
        if len(common_idx) == 0:
            continue
        tmp = pd.DataFrame({
            "ret": port_test.loc[common_idx],
            "bucket": regime.loc[common_idx].astype(str),
        })
        for bucket, grp in tmp.groupby("bucket"):
            d = summarize_portfolio(grp["ret"], label=bucket, rf=rf)
            d.update({"n_synth_series": int(pct)})
            rows.append(d)
    out = pd.DataFrame(rows)
    col_order = [
        "n_synth_series", "bucket", "sharpe", "calmar",
        "ann_mu", "ann_sd", "max_drawdown", "VaR_5%", "CVaR_5%", "n_obs",
    ]
    for c in col_order:
        if c not in out.columns:
            out[c] = np.nan
    return out[col_order].sort_values(["n_synth_series", "bucket"]).reset_index(drop=True)



def shock_test(
    real_prices: pd.DataFrame,
    synth_pools: dict,
    asset_cols: List[str],
    n_synth_series: int = 5,
    lookback_window: int = LOOKBACK_DAYS,
    investment_window: int = 60 * 5,
    allow_short: bool = False,
    seed: int = 42,
    rf: float = 0.0,
    scenarios: Optional[List[Tuple[str, Dict]]] = None,
) -> list:
    
    """This function performs the shock stress test. It takes in the real price data
    synthetic pools, and other parameters, and returns a list of DataFrames that summarizes the performance
    of the portfolio across different shock scenarios and different numbers of synthetic series. """

    """Parameters:
    a: real_prices: DataFrame of real prices
    b: synth_pools: dictionary of synthetic returns
    c: asset_cols: list of asset column names
    d: n_synth_series: the number of synthetic series to include in training
    e: lookback_window: number of days used in the training period
    f: investment_window: number of days used in the testing period
    g: allow_short: true/false for short or not
    h: seed: chosen seed
    i: rf: annual risk-free rate
    j: scenarios: list of shock scenarios to apply"""


    real_rets = pct_returns(real_prices[asset_cols])
    default_scenarios = [
        ("BASE",{"shock_type": "VOL_SCALE","vol_scale": 1.0}),
        ("VOL_SCALE_1.8x",{"shock_type": "VOL_SCALE", "vol_scale": 1.8}),
        ("NEG_SHOCK_-10%", {"shock_type": "NEG_RETURN_SHOCK","shock_size": 0.10}),
        ("CORR_INFLATE_1.3",{"shock_type": "CORR_INFLATE","corr_inflate": 1.3}),
        ("WHIPSAW", {"shock_type": "WHIPSAW"}),
        ("SLOW_BLEED",{"shock_type": "SLOW_BLEED"}),
        ("crash_2008", {"shock_type": "crash_2008"}),
    ]
    list_of_outs = []
    window = 21
    for i in range(0, len(real_rets) - lookback_window - investment_window, window):
        window_rets = real_rets[i: i + lookback_window + investment_window]
        train_rets, test_rets = time_split(window_rets, lookback_window, investment_window)
        use_train, n_draw = mix_train_matrix(
            train_rets.values, synth_pools, asset_cols,
            n_synth_series=n_synth_series, seed=seed,
            train_index = train_rets.index
        )
        w = mean_var_weights(use_train, allow_short=allow_short)
        w = collapse_weights(w, n_assets=len(asset_cols), n_draw=n_draw)
        used_scenarios = scenarios if scenarios is not None else default_scenarios
        rows: List[Dict] = []
        for name, kwargs in used_scenarios:
            if name == "BASE" and kwargs.get("shock_type") == "VOL_SCALE" and kwargs.get("vol_scale") == 1.0:
                shocked = test_rets
            else:
                shocked = apply_shock_to_returns(test_rets, seed=seed, **kwargs)
            port = pd.Series(shocked.values @ w, index=shocked.index)
            d = summarize_portfolio(port, label=name, rf=rf)
            d.update({"n_synth_series": int(n_synth_series)})
            rows.append(d)
        out = pd.DataFrame(rows)
        col_order = [
            "n_synth_series", "bucket", "sharpe", "calmar",
            "ann_mu", "ann_sd", "max_drawdown", "VaR_5%", "CVaR_5%", "n_obs",
        ]
        list_of_outs.append(out[col_order].sort_values(["bucket"]).reset_index(drop=True))
    return list_of_outs
