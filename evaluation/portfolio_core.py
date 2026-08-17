from __future__ import annotations

"""This module contains the core functions for portfolio construction, evaluation, and regime classification."""
"""The functions are:
- pct_returns: converts price data to returns
- time_split: splits returns into training and test sets
- sharpe_ratio: calculates the Sharpe ratio of a return series
- max_drawdown: calculates the maximum drawdown of a return series
- calmar_ratio: calculates the Calmar ratio of a return series
- var_cvar: calculates the VaR and CVaR of a return series
- summarize_portfolio: summarizes the performance of a portfolio by calculating various risk metrics
- classify_regime: classifies a test window as "crisis" or "calm" based on crisis windows that we found online
- regime_by_vol: classifies regimes based on rolling volatility
- regime_by_drawdown: classifies regimes based on rolling drawdown
- project_to_simplex: simplex projection
- mean_var_weights: calculates mean-variance weights 
- mean_var_weights_shrinkage: same as above but with Ledoit-Wolf shrinkage estimator for the covariance matrix
- mix_train_matrix: augments the real training matrix with synthetic series
- collapse_weights: collapses the weights of the synthetic series back into their respective original assets
- portfolio_experiment: runs the main portfolio experiment by varying the number of synthetic series
"""


try:
    import cvxpy as cp
except Exception:
    cp = None
import numpy as np
import pandas as pd


TRADING_DAYS = 252
LOOKBACK_DAYS = 252*5
INVESTMENT_WINDOW = 252
CRISIS_WINDOWS = [
    ("2020-02-01", "2020-06-01"),  # COVID
    ("2022-01-01", "2022-12-31"),  # 2022 rate cycle
    ("2018-10-01", "2019-01-01"),  # Q4 drawdown
] #for future, u can check if there are more crisis windows worth adding



# -- data classification ---------------------------------------
def classify_regime(test_start, test_end) -> str: 
    '''This decides if a crisis is happening during the test test window. 
    This allows us to bucket results based on market regime.'''
    """Parameters:
    a: test_start: start date of the test window
    b: test_end: end date of the test window 
    """
    for cs, ce in CRISIS_WINDOWS:
        if test_start <= pd.Timestamp(ce) and test_end >= pd.Timestamp(cs):
            return "crisis"
    return "calm"


# -- helper methods ---------------------------------------
def pct_returns(price_df: pd.DataFrame) -> pd.DataFrame:
    """This takes a dataframe of prices and returns a dataframe of pct returns, 
    with rows containing any NaN dropped."""

    """Parameters:
    a: price_df: DataFrame of asset prices
    """
    px = price_df.astype(float).ffill()
    rets = px.pct_change().dropna(how="all")
    rets = rets.dropna(axis=0, how="any")
    return rets

def time_split(rets: pd.DataFrame, lookback: int, invest: int) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """This splits the returns dataframe into a training set of length lookback and a test set of length invest, 
    which starts immediately after the training set. """

    """Parameters:
    a: rets: DataFrame of returns
    b: lookback: number of observations used for the training
    c: invest: number of observations used when testing the portfolio performance
    """

    if lookback + invest > len(rets):
        raise ValueError("lookback + invest exceeds length of returns")
    train = rets.iloc[:lookback].copy()
    test = rets.iloc[lookback: lookback + invest].copy()
    return train, test

# -- risk metrics ---------------------------------------

def sharpe_ratio(r: np.ndarray, rf: float = 0.0) -> float:
    """This calcaulates the annualised Sharpe ratio of a return series r, given a risk-free rate rf"""
    """Parameters:
    a: r: array of returns
    b: rf: annual risk-free rate 
    """

    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    excess = r - rf / TRADING_DAYS
    sd = excess.std(ddof=0)
    if np.isclose(sd, 0.0):
        return np.nan
    return float(np.sqrt(TRADING_DAYS) * excess.mean() / sd)


def max_drawdown(r: np.ndarray) -> float:
    """This calculates the maximum drawdown of a return series r"""

    """Parameters:
    a: r: array of returns
    """
    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan
    equity = np.cumprod(1.0 + r)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(dd.min())

def calmar_ratio(r: np.ndarray) -> float:
    """This calculates the Calmar ratio of a return series r,
    which is the annualised return divided by the absolute value of the maximum drawdown.
    It's a particularly good measure for evaluating performance during a stress period, 
    as it punishes drawdowns more heavily than the Sharpe ratio."""

    """Parameters:
    a: r: array of returns
    """

    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size < 2:
        return np.nan
    ann_ret = float(r.mean() * TRADING_DAYS)
    mdd = max_drawdown(r)
    if not np.isfinite(mdd) or np.isclose(mdd, 0.0):
        return np.nan
    return ann_ret / abs(mdd)

def var_cvar(r: np.ndarray, alpha: float = 0.05) -> Tuple[float, float]:
    """This calculates the conditional value at risk (CVaR) and value at risk (VaR)
    of a return series r at confidence level alpha."""

    """Parameters:
    a: r: array of returns
    b: alpha: confidence level for VaR/CVaR
    """

    r = np.asarray(r, dtype=float)
    r = r[np.isfinite(r)]
    if r.size == 0:
        return np.nan, np.nan
    q = float(np.quantile(r, alpha))
    tail = r[r <= q]
    cvar = float(tail.mean()) if tail.size else q
    return q, cvar

# -- portfolio evaluation ---------------------------------------

def summarize_portfolio(port_ret: pd.Series, label: str = "ALL", rf: float = 0.0) -> Dict:
    """This summarizes the performance of a portfolio by calling the risk metrics that are defined
    above and returning them in a dictionary"""


    """Parameters:
    a: port_ret: Series of portfolio returns
    b: label: label for the portfolio 
    c: rf: annual risk-free rate
    """

    arr = port_ret.values
    v, cv = var_cvar(arr)
    return {
        "bucket":label,
        "sharpe":sharpe_ratio(arr, rf=rf),
        "calmar":calmar_ratio(arr),
        "max_drawdown":max_drawdown(arr),
        "VaR_5%":v,
        "CVaR_5%":cv,
        "ann_mu":float(np.nanmean(arr) * TRADING_DAYS),
        "ann_sd":float(np.nanstd(arr, ddof=0) * np.sqrt(TRADING_DAYS)),
        "n_obs":int(np.isfinite(arr).sum()),
    }


# ---------------------- regimes ----------------------

def regime_by_vol(price_df: pd.DataFrame, window: int = 21,
                  q_low: float = 0.33, q_high: float = 0.67) -> pd.Series:
    
    """This builds a regime classifier based on the rolling volatility of the average return across all assets in price_df.
    It classifies each day as 'LOW_VOL', 'MID_VOL', or 'HIGH_VOL' based on whether the rolling volatility is in the bottom, middle,
    or top third of its historical distribution."""
    
    """Parameters:
    a: price_df: DataFrame of asset prices
    b: window: rolling window length
    c: q_low: lower quantile cutoff
    d: q_high: upper quantile cutoff
    """
    rets = pct_returns(price_df)
    basket = rets.mean(axis=1)
    rv = basket.rolling(window).std() * np.sqrt(TRADING_DAYS)
    rv = rv.dropna()
    lo = rv.quantile(q_low)
    hi = rv.quantile(q_high)
    reg = pd.Series(index=rv.index, dtype="object")
    reg[rv <= lo] = "LOW_VOL"
    reg[(rv > lo) & (rv < hi)] = "MID_VOL"
    reg[rv >= hi] = "HIGH_VOL"
    return reg

def regime_by_drawdown(price_df: pd.DataFrame, lookback: int = 63, dd_th: float = -0.10) -> pd.Series:
    """This builds a regime classifier based on the rolling maximum drawdown of the portfolio.
    It classifies each day as 'DRAWDOWN' or 'NORMAL' based on whether the drawdown exceeds the threshold."""

    """Parameters:
    a: price_df: DataFrame of asset prices
    b: lookback: rolling window length
    c: dd_th: drawdown cutoff 
    """
    rets = pct_returns(price_df)
    basket = rets.mean(axis=1)
    equity = (1.0 + basket).cumprod()
    peak = equity.rolling(lookback).max()
    dd = equity / peak - 1.0
    dd = dd.dropna()
    reg = pd.Series(index=dd.index, dtype="object")
    reg[dd <= dd_th] = "DRAWDOWN"
    reg[dd > dd_th] = "NORMAL"
    return reg


# ------ Portfolio fittting -------------------------------------------


def project_to_simplex(v: np.ndarray) -> np.ndarray:
    """This function uses simplex projection to find the closest point to v 
    that lies in the simplex defined by sum(w)=1 and w>=0."""

    """Parameters:
    a: v: input vector of weights (can be any real numbers)
    """

    v = np.asarray(v, dtype=float)
    n = v.size
    if n == 0:
        return v
    u = np.sort(v)[::-1]
    cssv = np.cumsum(u)
    rho = np.nonzero(u * np.arange(1, n + 1) > (cssv - 1))[0]
    if rho.size == 0:
        return np.ones(n) / n
    rho = rho[-1]
    theta = (cssv[rho] - 1) / (rho + 1)
    w = np.maximum(v - theta, 0.0)
    s = w.sum()
    return w / s if not np.isclose(s, 0.0) else np.ones(n) / n

def _closed_form_mean_var_weights(mu: np.ndarray, Sigma: np.ndarray, allow_short: bool) -> np.ndarray:
    n = Sigma.shape[0]
    invS = np.linalg.pinv(Sigma)
    ones = np.ones(n)
    a = float(ones @ invS @ mu)
    b = float(ones @ invS @ ones)
    lam = (a - 1.0) / b if not np.isclose(b, 0.0) else 0.0
    wv = invS @ (mu - lam * ones)
    if allow_short:
        s = float(wv.sum())
        return wv / s if not np.isclose(s, 0.0) else np.ones(n) / n
    return project_to_simplex(wv)


def mean_var_weights(
    R: np.ndarray,
    allow_short: bool = False,
    ridge: float = 1e-6,
    max_weight: float = None,
    n_assets_orig: int = None,
    n_draw: int = None,
) -> np.ndarray:
    """This function calculates mean-variance optimal weights based on the return matrix.
    It follows the standard markowitz formulation
    But also allows for additional optional constraints."""

    """Parameters:
    a: R: 2D array of returns
    b: allow_short: if True, allows for short positions
    c: ridge: true/false adds ridge regularisation
    d: max_weight: adds a constraint that weights must be <= max_weight
    e: n_assets_orig: number of original assets before augmentation
    f: n_draw: number of synthethic columns per asset"""


    if R.shape[0] < 2:
        raise ValueError("Need >=2 observations to fit Markowitz weights")
    mu = R.mean(axis=0)
    n_obs, n_feat = R.shape
    # Regime-matched / heavily augmented panels are often singular (T << p).
    eff_ridge = float(ridge)
    if n_obs <= n_feat:
        avg_var = float(np.mean(np.var(R, axis=0, ddof=1))) if n_obs > 1 else 1.0
        eff_ridge = max(eff_ridge, 1e-4 * max(avg_var, 1e-12))
    Sigma = np.cov(R.T, ddof=1)
    if np.ndim(Sigma) == 0:
        Sigma = np.array([[float(Sigma)]])
    Sigma = 0.5 * (Sigma + Sigma.T) + eff_ridge * np.eye(n_feat)
    n = Sigma.shape[0]
    if cp is not None:
        try:
            w = cp.Variable(n)
            # psd_wrap avoids rare ARPACK failures when certifying near-PSD covariances
            obj = cp.Maximize(mu @ w - 0.5 * cp.quad_form(w, cp.psd_wrap(Sigma)))
            cons = [cp.sum(w) == 1]
            if not allow_short:
                cons += [w >= 0]
            if max_weight is not None:
                if n_assets_orig is not None and n_draw is not None and n_draw > 0:
                    for i in range(n_assets_orig):
                        synth_start = n_assets_orig + i * n_draw
                        synth_end   = synth_start + n_draw
                        group_w = w[i] + cp.sum(w[synth_start:synth_end])
                        cons += [group_w <= max_weight]
                        if allow_short:
                            cons += [group_w >= -max_weight]
                else:
                    cons += [w <= max_weight]
                    if allow_short:
                        cons += [w >= -max_weight]
            cp.Problem(obj, cons).solve(solver="ECOS")
            wv = np.array(w.value).reshape(-1)
            if np.any(~np.isfinite(wv)):
                raise RuntimeError("Optimization failed: weights are not finite")
            s = wv.sum()
            if np.isclose(s, 0.0):
                raise RuntimeError("Optimization failed: sum(weights)=0")
            return wv / s
        except Exception:
            # Numerical PSD certification / solver failures: fall back to closed form
            return _closed_form_mean_var_weights(mu, Sigma, allow_short=allow_short)
    return _closed_form_mean_var_weights(mu, Sigma, allow_short=allow_short)


def mean_var_weights_shrinkage(
    R: np.ndarray,
    allow_short: bool = False,
    ridge: float = 1e-6,
) -> np.ndarray:
    
    """Same as above but with Ledoit-Wolf shrinkage estimator for the covariance matrix"""
    from sklearn.covariance import LedoitWolf
    if R.shape[0] < 2:
        raise ValueError("Need >=2 observations to fit Markowitz weights")
    mu = R.mean(axis=0)
    lw = LedoitWolf().fit(R)
    Sigma = 0.5 * (lw.covariance_ + lw.covariance_.T) + ridge * np.eye(R.shape[1])
    n = Sigma.shape[0]
    if cp is not None:
        try:
            w = cp.Variable(n)
            obj = cp.Maximize(mu @ w - 0.5 * cp.quad_form(w, cp.psd_wrap(Sigma)))
            cons = [cp.sum(w) == 1]
            if not allow_short:
                cons += [w >= 0]
            cp.Problem(obj, cons).solve(solver="ECOS")
            wv = np.array(w.value).reshape(-1)
            if np.any(~np.isfinite(wv)):
                raise RuntimeError("Shrinkage MVO failed: non-finite weights")
            s = wv.sum()
            if np.isclose(s, 0.0):
                raise RuntimeError("Shrinkage MVO failed: sum(weights)=0")
            return wv / s
        except Exception:
            return _closed_form_mean_var_weights(mu, Sigma, allow_short=allow_short)
    return _closed_form_mean_var_weights(mu, Sigma, allow_short=allow_short)



def synth_series_stress_scores(synth_pools: dict, asset_cols: list) -> np.ndarray:
    """Score each aligned synthetic series by equal-weight vol × (1+|max DD|)."""
    n_series = min(int(synth_pools[a].shape[1]) for a in asset_cols)
    scores = np.full(n_series, -np.inf, dtype=float)
    for j in range(n_series):
        ew = np.mean(
            [synth_pools[a].iloc[:, j].to_numpy(dtype=float) for a in asset_cols],
            axis=0,
        )
        ew = ew[np.isfinite(ew)]
        if ew.size < 2:
            continue
        wealth = np.cumprod(1.0 + ew)
        peak = np.maximum.accumulate(wealth)
        mdd = float((wealth / peak - 1.0).min())
        vol = float(ew.std(ddof=0) * np.sqrt(TRADING_DAYS))
        scores[j] = vol * (1.0 + abs(mdd))
    return scores


def choose_synth_series(
    synth_pools: dict,
    asset_cols: list,
    n_synth_series: int,
    seed: int = 42,
    prefer_stress: bool = True,
) -> np.ndarray:
    """Select synthetic series indices, optionally preferring high-stress paths."""
    n_series = min(int(synth_pools[a].shape[1]) for a in asset_cols)
    n_draw = min(int(n_synth_series), n_series)
    if n_draw <= 0:
        return np.array([], dtype=int)
    if prefer_stress:
        scores = synth_series_stress_scores(synth_pools, asset_cols)
        return np.argsort(-scores)[:n_draw]
    rng = np.random.default_rng(seed)
    return rng.choice(n_series, size=n_draw, replace=False)


def mix_train_matrix(
    real_train: np.ndarray,
    synth_pools: dict,
    asset_cols: list,
    n_synth_series: int,
    seed: int = 42,
    train_index=None,
    prefer_stress: bool = True,
) -> tuple:
    """Column-augment real training returns with synthetic series per asset."""
    if n_synth_series == 0:
        return real_train.copy(), 0
    T = real_train.shape[0]
    chosen = choose_synth_series(
        synth_pools, asset_cols, n_synth_series, seed=seed, prefer_stress=prefer_stress
    )
    n_draw = int(chosen.size)
    extra_cols = []
    for asset in asset_cols:
        pool = synth_pools[asset]
        if pool.shape[0] < T and train_index is None:
            raise ValueError(f"Synthetic pool for {asset} has {pool.shape[0]} rows but need {T}")
        if train_index is not None:
            block = pool.loc[train_index].values
        else:
            block = pool.values[:T]
        extra_cols.append(block[:, chosen])
    return np.hstack([real_train, np.hstack(extra_cols)]), n_draw


def mix_train_with_regime_paths(
    real_train: np.ndarray,
    synth_paths: np.ndarray,
    seq_len: int = 128,
    rng: np.random.Generator | None = None,
) -> tuple[np.ndarray, int]:
    """Column-stack real regime-matched days with HMM-path specialist series.

    ``synth_paths`` has shape ``(n_synth, H, n_assets)``. Mixing length is
    ``min(len(real_train), seq_len, H)``. This does not look up calendar dates.
    """
    real_train = np.asarray(real_train, dtype=float)
    if real_train.ndim != 2:
        raise ValueError(f"real_train must be 2D, got {real_train.shape}")
    n_synth = 0 if synth_paths is None else int(np.asarray(synth_paths).shape[0])
    if n_synth == 0:
        return real_train.copy(), 0
    synth_paths = np.asarray(synth_paths, dtype=float)
    if synth_paths.ndim != 3:
        raise ValueError(f"synth_paths must be (n_synth, H, n_assets), got {synth_paths.shape}")
    if synth_paths.shape[2] != real_train.shape[1]:
        raise ValueError(
            f"Asset mismatch: real {real_train.shape[1]} vs synth {synth_paths.shape[2]}"
        )
    horizon = int(synth_paths.shape[1])
    t_mix = min(int(real_train.shape[0]), int(seq_len), horizon)
    if t_mix < 2:
        raise ValueError(f"Need at least 2 mixed rows; got T={t_mix}")
    real_block = real_train[-t_mix:]
    if rng is None:
        rng = np.random.default_rng()
    extra_cols = []
    for asset_idx in range(real_train.shape[1]):
        series = np.empty((t_mix, n_synth), dtype=float)
        for path_idx in range(n_synth):
            path = synth_paths[path_idx, :, asset_idx]
            start = 0 if horizon == t_mix else int(rng.integers(0, horizon - t_mix + 1))
            series[:, path_idx] = path[start : start + t_mix]
        extra_cols.append(series)
    return np.hstack([real_block, np.hstack(extra_cols)]), n_synth


def mix_train_rows(
    real_train: np.ndarray,
    synth_pools: dict,
    asset_cols: list,
    n_synth_series: int,
    n_extra_rows: int,
    seed: int = 42,
    prefer_stress: bool = True,
) -> np.ndarray:
    """Row-append synthetic observations (keeps n_assets columns)."""
    if int(n_synth_series) <= 0 or int(n_extra_rows) <= 0:
        return real_train.copy()
    chosen = choose_synth_series(
        synth_pools, asset_cols, n_synth_series, seed=seed, prefer_stress=prefer_stress
    )
    panels = [
        np.column_stack([synth_pools[a].iloc[:, j].to_numpy(dtype=float) for a in asset_cols])
        for j in chosen
    ]
    pool_rows = np.vstack(panels)
    pool_rows = pool_rows[np.isfinite(pool_rows).all(axis=1)]
    if pool_rows.shape[0] == 0:
        return real_train.copy()
    rng = np.random.default_rng(seed)
    take = min(int(n_extra_rows), pool_rows.shape[0])
    idx = rng.choice(pool_rows.shape[0], size=take, replace=False)
    return np.vstack([real_train, pool_rows[idx]])

def collapse_weights(w: np.ndarray, n_assets: int, n_draw: int) -> np.ndarray:
    """This function takes a weight vector w of length n_assets + n_assets*n_draw
      corresponding to a portfolio that includes both synthetic series and real asset data, 
    and then it collapses the weights by summing the weights of the synthetic series back into their respective
    original assets, ultimately returning a weight vector of the original assets only."""

    """Parameters:
    a: w: weight vector of length n_assets + n_assets*n_draw
    b: n_assets: number of original assets
    c: n_draw: number of synthetic columns per original asset"""

    if n_draw == 0:
        return w[:n_assets] / w[:n_assets].sum()
    w_collapsed = w[:n_assets].copy()
    for i in range(n_assets):
        start = n_assets + i * n_draw
        end = start + n_draw
        w_collapsed[i] += w[start:end].sum()
    w_collapsed /= w_collapsed.sum()
    return w_collapsed




