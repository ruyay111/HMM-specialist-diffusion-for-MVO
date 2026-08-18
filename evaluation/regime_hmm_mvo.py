"""Regime MVO using the supervised HMM.

Regime detection and supervised Gaussian HMM logic come from ``supervised_hmm.py``.
This module handles portfolio selection, synthetic augmentation, and rolling
evaluation.

Strategy ``regime_mvo``: at each rebalance, forecast the dominant 60-day
regime and train MVO on recent historical days with that regime label.

Default mixing is row-append of calendar synth. Set
``synth_mode="hmm_path_specialist"`` to column-stack HMM-path specialist
windows instead (no date matching).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

from supervised_hmm import (
    assign_regimes_by_centers,
    fit_supervised_hmm,
    identify_regimes,
)
from portfolio_core import (
    calmar_ratio,
    classify_regime,
    collapse_weights,
    mean_var_weights,
    mix_train_rows,
    mix_train_with_regime_paths,
    project_to_simplex,
    sharpe_ratio,
    summarize_portfolio,
)
from synth_path_builder import sample_stitched_paths

try:
    import cvxpy as cp
except ImportError:  # pragma: no cover
    cp = None

STRATEGY_NAME = "regime_mvo"
SOFT_MIN_TRAIN = 30
ROWS_PER_SYNTH = 21
SYNTH_MODE_ROW_APPEND = "row_append"
SYNTH_MODE_HMM_PATH = "hmm_path_specialist"
PATH_HORIZON = 128


def market_regime_signal(real_rets: pd.DataFrame, asset_cols: Sequence[str]) -> pd.Series:
    """Use one equal-weight market return signal shared by all assets."""
    return real_rets[list(asset_cols)].mean(axis=1).rename("market_return")


def standardize_history(values: pd.Series) -> tuple[pd.Series, float, float]:
    """Standardize the market signal exactly within the available history."""
    mean = float(values.mean())
    std = float(values.std(ddof=1))
    if not np.isfinite(std) or np.isclose(std, 0.0):
        raise ValueError("Historical market return standard deviation is zero or invalid.")
    return ((values - mean) / std).rename("standardized_market_return"), mean, std


def select_regime_train(
    asset_returns: pd.DataFrame,
    labels: pd.Series,
    predicted_regime: int,
    max_train: int = 504,
) -> tuple[pd.DataFrame, dict]:
    """Select recent historical observations in the dominant forecast regime."""
    common = asset_returns.index.intersection(labels.index)
    aligned_returns = asset_returns.loc[common]
    aligned_labels = labels.loc[common]
    available = aligned_returns.loc[aligned_labels == int(predicted_regime)]
    selected = available.tail(int(max_train))
    metadata = {
        "n_available": int(len(available)),
        "n_hist_train": int(len(selected)),
        "predicted_regime": int(predicted_regime),
        "train_source": STRATEGY_NAME,
    }
    return selected, metadata


def _solve_stabilized_mvo(
    train: pd.DataFrame,
    historical_returns: pd.DataFrame,
    objective: str = "mean_variance",
    max_weight: float = 0.30,
    mean_shrink: float = 0.5,
    allow_short: bool = False,
) -> np.ndarray:
    """Ledoit-Wolf covariance, shrunk mean, and optional long-only / weight caps."""
    X = train.to_numpy(dtype=float)
    conditional_mean = X.mean(axis=0)
    unconditional_mean = historical_returns.to_numpy(dtype=float).mean(axis=0)
    mu = float(mean_shrink) * conditional_mean + (1.0 - float(mean_shrink)) * unconditional_mean
    covariance = LedoitWolf().fit(X).covariance_
    covariance = 0.5 * (covariance + covariance.T)
    n_assets = X.shape[1]
    cap = float(max_weight) if max_weight is not None else None

    if cp is not None:
        try:
            weights = cp.Variable(n_assets)
            risk = cp.quad_form(weights, cp.psd_wrap(covariance))
            if objective == "min_variance":
                problem_objective = cp.Minimize(risk)
            else:
                problem_objective = cp.Maximize(mu @ weights - 0.5 * risk)
            constraints = [cp.sum(weights) == 1]
            if not allow_short:
                constraints.append(weights >= 0)
            if cap is not None:
                constraints.append(weights <= cap)
                if allow_short:
                    constraints.append(weights >= -cap)
            cp.Problem(problem_objective, constraints).solve(solver="ECOS")
            result = np.asarray(weights.value, dtype=float).reshape(-1)
            if np.all(np.isfinite(result)) and not np.isclose(result.sum(), 0.0):
                return result / result.sum()
        except (cp.error.SolverError, TypeError, ValueError, RuntimeError):
            ...

    inverse = np.linalg.pinv(covariance)
    ones = np.ones(n_assets)
    if objective == "min_variance":
        result = inverse @ ones
    else:
        a = float(ones @ inverse @ mu)
        b = float(ones @ inverse @ ones)
        lagrange = (a - 1.0) / b if not np.isclose(b, 0.0) else 0.0
        result = inverse @ (mu - lagrange * ones)
    if allow_short:
        if cap is not None:
            result = np.clip(result, -cap, cap)
        total = result.sum()
        if np.isclose(total, 0.0):
            result = np.full(n_assets, 1.0 / n_assets)
        else:
            result = result / total
        return result
    result = project_to_simplex(result)
    if cap is None:
        return result
    for _ in range(20):
        result = np.minimum(result, cap)
        result /= result.sum()
        if result.max() <= cap + 1e-10:
            break
    return result


def _realized_test_regimes(
    standardized_history: pd.Series,
    standardized_test: pd.Series,
    regime_volatility_centers: np.ndarray,
    volatility_window: int,
) -> pd.Series:
    """Assign held-out test regimes using only centers learned on history."""
    combined = pd.concat([standardized_history, standardized_test])
    labels = assign_regimes_by_centers(
        combined.to_numpy(),
        regime_volatility_centers,
        volatility_window=volatility_window,
    )
    index = combined.index[-len(labels) :]
    labeled = pd.Series(labels, index=index, dtype=int)
    return labeled.reindex(standardized_test.index).dropna().astype(int)


def build_regime_train(
    hist_train: pd.DataFrame,
    synth_pools: dict,
    asset_cols: Sequence[str],
    n_synth: int,
    seed: int,
    min_regime_train: int = 80,
    soft_min_train: int = SOFT_MIN_TRAIN,
    rows_per_synth: int = ROWS_PER_SYNTH,
    prefer_stress: bool = True,
) -> tuple[pd.DataFrame | None, dict]:
    """Build the MVO training panel, filling scarce regimes with synthetic rows."""
    n_hist = int(len(hist_train))
    meta = {
        "n_hist_train": n_hist,
        "n_synth_rows": 0,
        "n_train": n_hist,
        "used_synth_fill": False,
    }
    if n_hist < int(soft_min_train):
        return None, meta

    train = hist_train
    if int(n_synth) > 0:
        fill_rows = max(0, int(min_regime_train) - n_hist)
        aug_rows = int(n_synth) * int(rows_per_synth)
        n_extra = fill_rows + aug_rows
        if n_extra > 0:
            arr = mix_train_rows(
                hist_train.to_numpy(dtype=float),
                synth_pools,
                list(asset_cols),
                n_synth_series=int(n_synth),
                n_extra_rows=n_extra,
                seed=int(seed),
                prefer_stress=prefer_stress,
            )
            train = pd.DataFrame(arr, columns=hist_train.columns)
            meta["n_synth_rows"] = int(len(train) - n_hist)
            meta["n_train"] = int(len(train))
            meta["used_synth_fill"] = bool(fill_rows > 0)
    return train, meta


def _fit_specialist_column_stack_weights(
    mixed: np.ndarray,
    n_assets: int,
    n_draw: int,
    objective: str,
    max_weight: float,
    allow_short: bool,
) -> np.ndarray:
    """Data_5-style column-stack MVO, then collapse synthetic columns."""
    if objective == "min_variance":
        mu = np.zeros(mixed.shape[1], dtype=float)
        ridge = 1e-6
        n_obs, n_feat = mixed.shape
        sigma = np.cov(mixed.T, ddof=1)
        if np.ndim(sigma) == 0:
            sigma = np.array([[float(sigma)]])
        if n_obs <= n_feat:
            avg_var = float(np.mean(np.var(mixed, axis=0, ddof=1)))
            ridge = max(ridge, 1e-4 * max(avg_var, 1e-12))
        sigma = 0.5 * (sigma + sigma.T) + ridge * np.eye(n_feat)
        if cp is not None:
            weights = cp.Variable(n_feat)
            problem = cp.Problem(
                cp.Minimize(cp.quad_form(weights, cp.psd_wrap(sigma))),
                [cp.sum(weights) == 1] + ([] if allow_short else [weights >= 0]),
            )
            problem.solve(solver="ECOS")
            raw = np.asarray(weights.value, dtype=float).reshape(-1)
            raw = raw / raw.sum()
        else:
            ones = np.ones(n_feat)
            raw = np.linalg.solve(sigma, ones)
            raw = raw / raw.sum()
    else:
        raw = mean_var_weights(
            mixed,
            allow_short=allow_short,
            max_weight=max_weight,
            n_assets_orig=n_assets,
            n_draw=n_draw,
        )
    return collapse_weights(raw, n_assets, n_draw)


def _fit_weights(
    train: pd.DataFrame,
    history: pd.DataFrame,
    objective: str,
    max_weight: float,
    mean_shrink: float,
    allow_short: bool = False,
) -> np.ndarray:
    return _solve_stabilized_mvo(
        train,
        historical_returns=history,
        objective=objective,
        max_weight=max_weight,
        mean_shrink=mean_shrink,
        allow_short=allow_short,
    )


def fit_regime_context(
    real_rets: pd.DataFrame,
    asset_cols: Sequence[str],
    history_end: int,
    invest: int = 60,
    n_regimes: int = 5,
    volatility_window: int = 20,
    min_segment_length: int = 32,
    max_regime_train: int = 504,
    random_state: int = 42,
    **_ignored,
) -> dict:
    """Fit the causal HMM at ``history_end`` and select regime-matched history."""
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols]
    history = returns.iloc[: int(history_end)]
    market = market_regime_signal(returns, asset_cols)
    market_history = market.iloc[: int(history_end)]
    standardized_history, _, _ = standardize_history(market_history)

    regime_labels_array, regime_diagnostics = identify_regimes(
        standardized_history.to_numpy(),
        n_regimes=n_regimes,
        volatility_window=volatility_window,
        min_segment_length=min_segment_length,
        random_state=random_state,
    )
    regime_labels = pd.Series(
        regime_labels_array,
        index=standardized_history.index,
        name="regime",
        dtype=int,
    )
    hmm_model = fit_supervised_hmm(
        standardized_history.to_numpy(),
        regime_labels_array,
        n_regimes=n_regimes,
    )
    _, forecast_occupancy, predicted_regime = hmm_model.forecast_occupancy(
        standardized_history.to_numpy(),
        horizon=int(invest),
    )
    hist_train, train_metadata = select_regime_train(
        history,
        regime_labels,
        predicted_regime,
        max_train=max_regime_train,
    )
    return {
        "history": history,
        "hist_train": hist_train,
        "train_metadata": train_metadata,
        "predicted_regime": int(predicted_regime),
        "forecast_occupancy": forecast_occupancy,
        "regime_diagnostics": regime_diagnostics,
        "hmm_model": hmm_model,
    }


def run_regime_mvo_experiment(
    real_rets: pd.DataFrame,
    synth_pools: dict,
    asset_cols: Sequence[str],
    mix_grid: Sequence[int] = (0, 1, 2, 5, 10, 20),
    invest: int = 60,
    step: int = 60,
    min_history: int = 1260,
    n_regimes: int = 5,
    volatility_window: int = 20,
    min_segment_length: int = 32,
    min_regime_train: int = 80,
    soft_min_train: int = SOFT_MIN_TRAIN,
    max_regime_train: int = 504,
    max_weight: float = 0.30,
    mean_shrink: float = 0.5,
    allow_short: bool = False,
    objectives: Sequence[str] = ("mean_variance", "min_variance"),
    seed: int = 42,
    random_state: int = 42,
    synth_mode: str = SYNTH_MODE_ROW_APPEND,
    specialist_pools: dict | None = None,
    path_horizon: int = PATH_HORIZON,
) -> pd.DataFrame:
    """Run regime MVO on non-overlapping out-of-sample windows."""
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols].dropna(how="any").copy()
    market = market_regime_signal(returns, asset_cols)
    rows = []
    if synth_mode not in {SYNTH_MODE_ROW_APPEND, SYNTH_MODE_HMM_PATH}:
        raise ValueError(f"Unknown synth_mode={synth_mode!r}")
    if synth_mode == SYNTH_MODE_HMM_PATH and specialist_pools is None:
        if any(int(n) > 0 for n in mix_grid):
            raise ValueError("hmm_path_specialist requires specialist_pools")

    for t in range(int(min_history), len(returns) - int(invest) + 1, int(step)):
        history = returns.iloc[:t]
        test = returns.iloc[t : t + int(invest)]
        market_history = market.iloc[:t]
        market_test = market.iloc[t : t + int(invest)]
        standardized_history, history_mean, history_std = standardize_history(market_history)
        standardized_test = ((market_test - history_mean) / history_std).rename(
            "standardized_market_return"
        )

        regime_labels_array, regime_diagnostics = identify_regimes(
            standardized_history.to_numpy(),
            n_regimes=n_regimes,
            volatility_window=volatility_window,
            min_segment_length=min_segment_length,
            random_state=random_state,
        )
        regime_labels = pd.Series(
            regime_labels_array,
            index=standardized_history.index,
            name="regime",
            dtype=int,
        )
        hmm_model = fit_supervised_hmm(
            standardized_history.to_numpy(),
            regime_labels_array,
            n_regimes=n_regimes,
        )
        filtered = hmm_model.predict_proba(
            standardized_history.to_numpy(),
            smoothed=False,
        )
        alpha_t = filtered[-1].copy()
        _, forecast_occupancy, predicted_regime = hmm_model.forecast_occupancy(
            standardized_history.to_numpy(),
            horizon=invest,
        )

        test_regimes = _realized_test_regimes(
            standardized_history,
            standardized_test,
            regime_diagnostics["regime_mean_volatility"],
            volatility_window,
        )
        realized_occupancy = (
            test_regimes.value_counts(normalize=True)
            .reindex(range(n_regimes), fill_value=0.0)
            .to_numpy(dtype=float)
        )
        realized_dominant = int(np.argmax(realized_occupancy))
        brier_score = float(np.sum((forecast_occupancy - realized_occupancy) ** 2))

        hist_train, train_metadata = select_regime_train(
            history,
            regime_labels,
            predicted_regime,
            max_train=max_regime_train,
        )

        shared = {
            "seed": int(seed),
            "strategy": STRATEGY_NAME,
            "allow_short": bool(allow_short),
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "predicted_regime": int(predicted_regime),
            "realized_dominant_regime": realized_dominant,
            "dominant_regime_accuracy": float(predicted_regime == realized_dominant),
            "forecast_brier": brier_score,
            "forecast_occupancy": forecast_occupancy.tolist(),
            "realized_occupancy": realized_occupancy.tolist(),
            "transition_matrix": hmm_model.transmat_.tolist(),
            "emission_means": hmm_model.means_.tolist(),
            "emission_variances": hmm_model.variances_.tolist(),
            "n_available_regime": train_metadata.get("n_available"),
            "n_hist_train": train_metadata.get("n_hist_train"),
            "train_source": train_metadata.get("train_source"),
            "synth_mode": synth_mode,
        }

        for objective in objectives:
            for n_synth in mix_grid:
                if objective == "min_variance" and int(n_synth) != 0:
                    continue
                base = {
                    **shared,
                    "objective": objective,
                    "n_synth_series": int(n_synth),
                    "n_synth_rows": 0,
                    "used_synth_fill": False,
                    "path_match_rate": np.nan,
                    "sampled_path_occupancy": None,
                }
                if int(len(hist_train)) < int(soft_min_train):
                    rows.append(
                        {
                            **base,
                            "n_train": int(len(hist_train)),
                            "status": "skipped_insufficient_regime_sample",
                        }
                    )
                    continue

                rng = np.random.default_rng(int(seed) + t + int(n_synth))
                if synth_mode == SYNTH_MODE_HMM_PATH and int(n_synth) > 0:
                    synth_arr, regime_paths = sample_stitched_paths(
                        alpha_t,
                        hmm_model.transmat_,
                        specialist_pools,
                        n_paths=int(n_synth),
                        horizon=int(path_horizon),
                        rng=rng,
                    )
                    mixed, n_draw = mix_train_with_regime_paths(
                        hist_train.to_numpy(dtype=float),
                        synth_arr,
                        seq_len=int(path_horizon),
                        rng=rng,
                    )
                    weights = _fit_specialist_column_stack_weights(
                        mixed,
                        n_assets=len(asset_cols),
                        n_draw=n_draw,
                        objective=objective,
                        max_weight=max_weight,
                        allow_short=allow_short,
                    )
                    path_match = float(np.mean(regime_paths == int(predicted_regime)))
                    occupancy = (
                        np.bincount(regime_paths.reshape(-1), minlength=n_regimes)
                        / regime_paths.size
                    )
                    base.update(
                        {
                            "n_train": int(mixed.shape[0]),
                            "n_synth_rows": 0,
                            "used_synth_fill": False,
                            "path_match_rate": path_match,
                            "sampled_path_occupancy": occupancy.tolist(),
                        }
                    )
                else:
                    train, build_meta = build_regime_train(
                        hist_train,
                        synth_pools,
                        asset_cols,
                        n_synth=int(n_synth) if synth_mode == SYNTH_MODE_ROW_APPEND else 0,
                        seed=int(seed) + t + int(n_synth),
                        min_regime_train=min_regime_train,
                        soft_min_train=soft_min_train,
                    )
                    base.update(
                        {
                            "n_train": build_meta["n_train"],
                            "n_synth_rows": build_meta["n_synth_rows"],
                            "used_synth_fill": build_meta["used_synth_fill"],
                        }
                    )
                    if train is None:
                        rows.append({**base, "status": "skipped_insufficient_regime_sample"})
                        continue
                    weights = _fit_weights(
                        train=train,
                        history=history,
                        objective=objective,
                        max_weight=max_weight,
                        mean_shrink=mean_shrink,
                        allow_short=allow_short,
                    )

                portfolio_returns = pd.Series(test.to_numpy() @ weights, index=test.index)
                metrics = summarize_portfolio(
                    portfolio_returns,
                    label=classify_regime(test.index[0], test.index[-1]),
                )
                status = "ok_synth_fill" if base.get("used_synth_fill") else "ok"
                rows.append(
                    {
                        **base,
                        **metrics,
                        "status": status,
                        "max_weight_used": float(np.max(np.abs(weights))),
                        "spy_sharpe": sharpe_ratio(test[asset_cols[0]].to_numpy()),
                        "spy_calmar": calmar_ratio(test[asset_cols[0]].to_numpy()),
                    }
                )

    return pd.DataFrame(rows)


def delta_vs_no_synth(results: pd.DataFrame) -> pd.DataFrame:
    """Per-seed, per-window change relative to n_synth_series=0."""
    valid = results.loc[results["status"].astype(str).str.startswith("ok")].copy()
    keys = ["seed", "test_start", "objective", "strategy"]
    if "allow_short" in valid.columns:
        keys = keys + ["allow_short"]
    metric_cols = [c for c in ["sharpe", "calmar", "max_drawdown", "CVaR_5%"] if c in valid.columns]
    baseline = valid.loc[valid["n_synth_series"] == 0, keys + metric_cols].rename(
        columns={c: f"{c}_n0" for c in metric_cols}
    )
    augmented = valid.loc[valid["n_synth_series"] > 0].copy()
    merged = augmented.merge(baseline, on=keys, how="inner")
    for c in metric_cols:
        merged[f"delta_{c}"] = merged[c] - merged[f"{c}_n0"]
    return merged


def summarize_synth_deltas(delta: pd.DataFrame, by: Sequence[str] = ("bucket", "n_synth_series")) -> pd.DataFrame:
    """Mean paired deltas and win rates, optionally by calm/crisis."""
    rows = []
    delta_cols = [c for c in delta.columns if c.startswith("delta_")]
    group_cols = [c for c in by if c in delta.columns]
    grouped = delta.groupby(group_cols) if group_cols else [((), delta)]
    for key, grp in grouped:
        key = key if isinstance(key, tuple) else (key,)
        row = dict(zip(group_cols, key))
        row["n_windows"] = int(len(grp))
        for c in delta_cols:
            vals = grp[c].dropna()
            row[f"{c}_mean"] = float(vals.mean()) if len(vals) else np.nan
            row[f"{c}_winrate"] = float((vals > 0).mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def _vol_bucket(predicted_regime: int, n_regimes: int) -> str:
    if int(predicted_regime) >= int(n_regimes) - 2:
        return "high"
    if int(predicted_regime) <= 1:
        return "low"
    return "mid"


def _apply_shock_for_horizon(
    test_rets: pd.DataFrame,
    name: str,
    kwargs: dict,
    seed: int,
    hist_train: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Apply shocks, adapting long scenarios to the HMM forecast horizon."""
    from scipy.stats import t as student_t
    from stress_test_final import apply_shock_to_returns, _get_2008_garch_params

    shock_type = kwargs.get("shock_type")
    if name == "BASE" or (shock_type == "VOL_SCALE" and float(kwargs.get("vol_scale", 1.0)) == 1.0):
        return test_rets
    if shock_type == "REGIME_VOL":
        if hist_train is None or len(hist_train) < 2:
            return test_rets
        target = float(hist_train.mean(axis=1).std(ddof=0))
        current = float(test_rets.mean(axis=1).std(ddof=0))
        scale = 1.0 if current == 0.0 else target / current
        return test_rets * scale
    if shock_type == "crash_2008" and len(test_rets) != 300:
        rng = np.random.default_rng(seed)
        omega, alpha, beta = _get_2008_garch_params()
        vol = 0.05
        crash_rets = []
        for _ in range(len(test_rets)):
            cr = student_t.rvs(df=3, loc=-0.008, scale=vol, random_state=rng)
            vol = np.sqrt(beta * vol**2 + alpha * cr**2 + omega * (0.05**2))
            crash_rets.append(cr)
        path = np.exp(np.asarray(crash_rets)) - 1.0
        shocked = (1.0 + test_rets.to_numpy()) * (1.0 + path[:, None]) - 1.0
        return pd.DataFrame(shocked, index=test_rets.index, columns=test_rets.columns)
    if shock_type == "SLOW_BLEED" and len(test_rets) < 121:
        rng = np.random.default_rng(seed)
        n = len(test_rets)
        bleed = rng.uniform(0.0005, 0.002, size=n)
        n_hits = min(8, n)
        bleed[rng.choice(n, size=n_hits, replace=False)] = -rng.uniform(0.001, 0.005, size=n_hits)
        out = test_rets.copy()
        out.iloc[:, :] = out.to_numpy() - bleed[:, None]
        return out
    return apply_shock_to_returns(test_rets, seed=seed, **kwargs)


def shock_test_regime_mvo(
    real_rets: pd.DataFrame,
    synth_pools: dict,
    asset_cols: Sequence[str],
    n_synth_series: int = 0,
    investment_window: int = 60,
    window_step: int = 21,
    mixed_lookback: int = 252,
    min_history: int = 1260,
    invest_forecast: int | None = None,
    n_regimes: int = 5,
    volatility_window: int = 20,
    min_segment_length: int = 32,
    min_regime_train: int = 80,
    soft_min_train: int = SOFT_MIN_TRAIN,
    max_regime_train: int = 504,
    max_weight: float = 0.30,
    mean_shrink: float = 0.5,
    allow_short: bool = False,
    seed: int = 42,
    random_state: int = 42,
    scenarios: Sequence[tuple] | None = None,
) -> pd.DataFrame:
    """HMM-aligned shock test with a mixed-history MVO baseline.

    At each date the HMM forecasts the next ``investment_window`` days
    (default 60, the same horizon used for training). Regime MVO trains
    on that label. Mixed MVO trains on the last ``mixed_lookback`` days.
    Both portfolios are held over the same shocked test window.

    ``REGIME_VOL`` rescales the test window to the forecasted regime's
    historical volatility — the shock the HMM is actually preparing for.
    """
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols].dropna(how="any").copy()
    horizon = int(invest_forecast) if invest_forecast is not None else int(investment_window)
    default_scenarios = [
        ("BASE", {"shock_type": "VOL_SCALE", "vol_scale": 1.0}),
        ("REGIME_VOL", {"shock_type": "REGIME_VOL"}),
        ("VOL_SCALE_1.8x", {"shock_type": "VOL_SCALE", "vol_scale": 1.8}),
        ("NEG_SHOCK_-10%", {"shock_type": "NEG_RETURN_SHOCK", "shock_size": 0.10}),
        ("CORR_INFLATE_1.3", {"shock_type": "CORR_INFLATE", "corr_inflate": 1.3}),
        ("WHIPSAW", {"shock_type": "WHIPSAW"}),
        ("SLOW_BLEED", {"shock_type": "SLOW_BLEED"}),
        ("CRASH_2008", {"shock_type": "crash_2008"}),
    ]
    used_scenarios = list(scenarios) if scenarios is not None else default_scenarios
    rows: list[dict] = []

    for t in range(
        int(min_history),
        len(returns) - int(investment_window) + 1,
        int(window_step),
    ):
        context = fit_regime_context(
            returns,
            asset_cols,
            history_end=t,
            invest=horizon,
            n_regimes=n_regimes,
            volatility_window=volatility_window,
            min_segment_length=min_segment_length,
            max_regime_train=max_regime_train,
            random_state=random_state,
        )
        predicted_regime = int(context["predicted_regime"])
        vol_bucket = _vol_bucket(predicted_regime, n_regimes)
        test_rets = returns.iloc[t : t + int(investment_window)]
        history = context["history"]

        regime_train, build_meta = build_regime_train(
            context["hist_train"],
            synth_pools,
            asset_cols,
            n_synth=int(n_synth_series),
            seed=int(seed) + t + int(n_synth_series),
            min_regime_train=min_regime_train,
            soft_min_train=soft_min_train,
            prefer_stress=(vol_bucket == "high"),
        )
        mixed_hist = history.tail(int(mixed_lookback))
        if int(n_synth_series) > 0:
            mixed_arr = mix_train_rows(
                mixed_hist.to_numpy(dtype=float),
                synth_pools,
                asset_cols,
                n_synth_series=int(n_synth_series),
                n_extra_rows=int(n_synth_series) * ROWS_PER_SYNTH,
                seed=int(seed) + t + int(n_synth_series) + 17,
                prefer_stress=(vol_bucket == "high"),
            )
            mixed_train = pd.DataFrame(mixed_arr, columns=mixed_hist.columns)
        else:
            mixed_train = mixed_hist

        methods: dict[str, pd.DataFrame | None] = {
            "regime": regime_train,
            "mixed": mixed_train if len(mixed_train) >= int(soft_min_train) else None,
        }
        for method, train in methods.items():
            if train is None:
                continue
            weights = _fit_weights(
                train=train,
                history=history,
                objective="mean_variance",
                max_weight=max_weight,
                mean_shrink=mean_shrink,
                allow_short=allow_short,
            )
            for name, kwargs in used_scenarios:
                shocked = _apply_shock_for_horizon(
                    test_rets,
                    name,
                    kwargs,
                    seed=seed,
                    hist_train=context["hist_train"],
                )
                port = pd.Series(shocked.to_numpy() @ weights, index=shocked.index)
                metrics = summarize_portfolio(port, label=name)
                metrics.update(
                    {
                        "method": method,
                        "n_synth_series": int(n_synth_series),
                        "predicted_regime": predicted_regime,
                        "vol_bucket": vol_bucket,
                        "used_synth_fill": build_meta["used_synth_fill"] if method == "regime" else False,
                        "test_start": test_rets.index[0],
                    }
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def run_regime_mvo_path(
    real_rets: pd.DataFrame,
    synth_pools: dict,
    asset_cols: Sequence[str],
    n_synth_list: Sequence[int] = (0, 1, 2, 5, 20),
    test_start: int | None = None,
    test_end: int | None = None,
    min_history: int = 1260,
    rebalance_step: int = 1,
    invest_forecast: int = 60,
    n_regimes: int = 5,
    volatility_window: int = 20,
    min_segment_length: int = 32,
    min_regime_train: int = 80,
    soft_min_train: int = SOFT_MIN_TRAIN,
    max_regime_train: int = 504,
    max_weight: float = 0.30,
    mean_shrink: float = 0.5,
    allow_short: bool = False,
    seed: int = 42,
    random_state: int = 42,
    static: bool = False,
) -> dict[int, pd.Series]:
    """Build daily portfolio return paths under regime MVO."""
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols].dropna(how="any").copy()
    start = int(test_start) if test_start is not None else int(min_history)
    end = int(test_end) if test_end is not None else len(returns)
    if start <= 0 or end <= start:
        raise ValueError("Need test_start < test_end with positive history.")

    port_rets = {int(n): [] for n in n_synth_list}
    dates = []
    weights_by_n = {
        int(n): np.full(len(asset_cols), 1.0 / len(asset_cols)) for n in n_synth_list
    }

    def _resolve_weights(history_end: int) -> None:
        context = fit_regime_context(
            returns,
            asset_cols,
            history_end=history_end,
            invest=invest_forecast,
            n_regimes=n_regimes,
            volatility_window=volatility_window,
            min_segment_length=min_segment_length,
            max_regime_train=max_regime_train,
            random_state=random_state,
        )
        for n in n_synth_list:
            train, _ = build_regime_train(
                context["hist_train"],
                synth_pools,
                asset_cols,
                n_synth=int(n),
                seed=int(seed) + history_end + int(n),
                min_regime_train=min_regime_train,
                soft_min_train=soft_min_train,
            )
            if train is None:
                continue
            weights_by_n[int(n)] = _fit_weights(
                train=train,
                history=context["history"],
                objective="mean_variance",
                max_weight=max_weight,
                mean_shrink=mean_shrink,
                allow_short=allow_short,
            )

    if static:
        _resolve_weights(start)
        for t in range(start, end):
            day = returns.iloc[t]
            dates.append(returns.index[t])
            for n in n_synth_list:
                port_rets[int(n)].append(float(day.to_numpy() @ weights_by_n[int(n)]))
    else:
        for t in range(start, end):
            if (t - start) % int(rebalance_step) == 0:
                _resolve_weights(t)
            day = returns.iloc[t]
            dates.append(returns.index[t])
            for n in n_synth_list:
                port_rets[int(n)].append(float(day.to_numpy() @ weights_by_n[int(n)]))

    return {
        int(n): pd.Series(port_rets[int(n)], index=dates, name=f"n_synth={n}")
        for n in n_synth_list
    }
