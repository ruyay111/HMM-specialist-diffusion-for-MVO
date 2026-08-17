"""Matched mixed vs regime MVO experiment for paper figures.

Both methods use the same test windows and the Data_5 column-stack
optimizer: synthetic series are extra columns, Markowitz is solved on the
expanded matrix, then weights are collapsed back to the original 10 assets.
The only intended difference is the historical training sample: last 252
mixed days vs HMM-label-matched days.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from portfolio_core import (
    classify_regime,
    collapse_weights,
    mean_var_weights,
    mix_train_matrix,
    summarize_portfolio,
)
from regime_hmm_mvo import (
    SOFT_MIN_TRAIN,
    _apply_shock_for_horizon,
    _realized_test_regimes,
    _vol_bucket,
    fit_regime_context,
    market_regime_signal,
    select_regime_train,
    standardize_history,
)
from paper_hmm import fit_paper_supervised_hmm, identify_regimes_paper


def _column_stack_weights(
    real_train: pd.DataFrame,
    synth_pools: dict,
    asset_cols: Sequence[str],
    n_synth: int,
    seed: int,
    prefer_stress: bool,
    allow_short: bool = False,
) -> tuple[np.ndarray, int, int, int]:
    """Fit Data_5-style column-stack MVO and return collapsed 10-asset weights."""
    asset_cols = list(asset_cols)
    idx = real_train.index
    if int(n_synth) > 0:
        for asset in asset_cols:
            idx = idx.intersection(synth_pools[asset].index)
    aligned = real_train.loc[idx]
    expanded, n_draw = mix_train_matrix(
        aligned.to_numpy(dtype=float),
        synth_pools,
        asset_cols,
        n_synth_series=int(n_synth),
        seed=int(seed),
        train_index=aligned.index,
        prefer_stress=prefer_stress,
    )
    finite = np.isfinite(expanded).all(axis=1)
    expanded = expanded[finite]
    if expanded.shape[0] < 2:
        raise ValueError("Need at least 2 finite training rows after alignment.")
    raw = mean_var_weights(
        expanded,
        allow_short=allow_short,
        n_assets_orig=len(asset_cols),
        n_draw=int(n_draw),
    )
    weights = collapse_weights(raw, n_assets=len(asset_cols), n_draw=int(n_draw))
    return weights, int(n_draw), int(expanded.shape[0]), int(expanded.shape[1])


def run_matched_experiment(
    real_rets: pd.DataFrame,
    synth_pools: dict,
    asset_cols: Sequence[str],
    mix_grid: Sequence[int] = (0, 1, 5, 20),
    invest: int = 60,
    step: int = 60,
    mixed_lookback: int = 252,
    min_history: int = 1260,
    n_regimes: int = 5,
    volatility_window: int = 20,
    min_segment_length: int = 32,
    min_regime_train: int = 80,
    soft_min_train: int = SOFT_MIN_TRAIN,
    max_regime_train: int = 504,
    allow_short: bool = False,
    prefer_stress: bool = True,
    seed: int = 42,
    random_state: int = 42,
    **_ignored,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Run mixed and regime MVO on identical out-of-sample windows.

    Returns
    -------
    windows:
        One row per (test window, method, n_synth).
    panel:
        Daily portfolio returns used to stitch wealth paths.
    """
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols].dropna(how="any").copy()
    market = market_regime_signal(returns, asset_cols)
    ranking = "stress" if prefer_stress else "random"
    window_rows: list[dict] = []
    panel_rows: list[dict] = []

    for t in range(int(min_history), len(returns) - int(invest) + 1, int(step)):
        history = returns.iloc[:t]
        test = returns.iloc[t : t + int(invest)]
        market_history = market.iloc[:t]
        market_test = market.iloc[t : t + int(invest)]
        standardized_history, history_mean, history_std = standardize_history(market_history)
        standardized_test = ((market_test - history_mean) / history_std).rename(
            "standardized_market_return"
        )

        regime_labels_array, regime_diagnostics = identify_regimes_paper(
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
        hmm_model = fit_paper_supervised_hmm(
            standardized_history.to_numpy(),
            regime_labels_array,
            n_regimes=n_regimes,
        )
        _, forecast_occupancy, predicted_regime = hmm_model.forecast_occupancy(
            standardized_history.to_numpy(),
            horizon=int(invest),
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
        vol_bucket = _vol_bucket(int(predicted_regime), n_regimes)
        calendar_bucket = classify_regime(test.index[0], test.index[-1])
        hist_train, train_metadata = select_regime_train(
            history,
            regime_labels,
            predicted_regime,
            max_train=max_regime_train,
        )
        mixed_hist = history.tail(int(mixed_lookback))

        shared = {
            "seed": int(seed),
            "test_start": test.index[0],
            "test_end": test.index[-1],
            "predicted_regime": int(predicted_regime),
            "realized_dominant_regime": realized_dominant,
            "forecast_correct": int(predicted_regime == realized_dominant),
            "forecast_brier": float(np.sum((forecast_occupancy - realized_occupancy) ** 2)),
            "vol_bucket": vol_bucket,
            "bucket": calendar_bucket,
            "n_available_regime": train_metadata.get("n_available"),
            "n_hist_regime": train_metadata.get("n_hist_train"),
            "prefer_stress": bool(prefer_stress),
            "ranking": ranking,
        }

        trains = {
            "regime": hist_train if len(hist_train) >= int(soft_min_train) else None,
            "mixed": mixed_hist if len(mixed_hist) >= int(soft_min_train) else None,
        }
        for n_synth in mix_grid:
            n_synth = int(n_synth)
            for method, real_train in trains.items():
                base = {
                    **shared,
                    "method": method,
                    "n_synth_series": n_synth,
                    "n_train": 0 if real_train is None else int(len(real_train)),
                    "n_draw": 0,
                    "n_features": 0,
                    "used_synth_fill": False,
                }
                if real_train is None:
                    window_rows.append({**base, "status": "skipped_insufficient_sample"})
                    continue
                weights, n_draw, n_obs, n_feat = _column_stack_weights(
                    real_train,
                    synth_pools,
                    asset_cols,
                    n_synth=n_synth,
                    seed=int(seed) + t + n_synth + (0 if method == "regime" else 17),
                    prefer_stress=prefer_stress,
                    allow_short=allow_short,
                )
                port = pd.Series(test.to_numpy() @ weights, index=test.index)
                metrics = summarize_portfolio(port, label=calendar_bucket)
                window_rows.append(
                    {
                        **base,
                        **metrics,
                        "n_train": n_obs,
                        "n_draw": n_draw,
                        "n_features": n_feat,
                        "status": "ok",
                        "max_weight_used": float(np.max(np.abs(weights))),
                        "hhi": float(np.sum(np.square(weights))),
                    }
                )
                for dt, ret in port.items():
                    panel_rows.append(
                        {
                            "date": dt,
                            "method": method,
                            "n_synth_series": n_synth,
                            "prefer_stress": bool(prefer_stress),
                            "ranking": ranking,
                            "port_ret": float(ret),
                            "predicted_regime": int(predicted_regime),
                            "vol_bucket": vol_bucket,
                            "bucket": calendar_bucket,
                        }
                    )

    return pd.DataFrame(window_rows), pd.DataFrame(panel_rows)


def shock_test_matched(
    real_rets: pd.DataFrame,
    synth_pools: dict,
    asset_cols: Sequence[str],
    n_synth_series: int = 0,
    investment_window: int = 60,
    window_step: int = 60,
    mixed_lookback: int = 252,
    min_history: int = 1260,
    n_regimes: int = 5,
    volatility_window: int = 20,
    min_segment_length: int = 32,
    soft_min_train: int = SOFT_MIN_TRAIN,
    max_regime_train: int = 504,
    allow_short: bool = False,
    prefer_stress: bool = True,
    seed: int = 42,
    random_state: int = 42,
    scenarios: Sequence[tuple] | None = None,
) -> pd.DataFrame:
    """HMM-aligned shock test using the column-stack optimizer for both methods."""
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols].dropna(how="any").copy()
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
            invest=int(investment_window),
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
        mixed_hist = history.tail(int(mixed_lookback))
        trains = {
            "regime": context["hist_train"] if len(context["hist_train"]) >= int(soft_min_train) else None,
            "mixed": mixed_hist if len(mixed_hist) >= int(soft_min_train) else None,
        }
        for method, real_train in trains.items():
            if real_train is None:
                continue
            weights, _, _, _ = _column_stack_weights(
                real_train,
                synth_pools,
                asset_cols,
                n_synth=int(n_synth_series),
                seed=int(seed) + t + int(n_synth_series) + (0 if method == "regime" else 17),
                prefer_stress=prefer_stress,
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
                        "test_start": test_rets.index[0],
                    }
                )
                rows.append(metrics)
    return pd.DataFrame(rows)


def valid_windows(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df["status"].astype(str).str.startswith("ok")].copy()


def both_methods(df: pd.DataFrame) -> pd.DataFrame:
    """Keep dates where mixed and regime both produced a valid result for that n."""
    ok = valid_windows(df)
    keys = ["test_start", "n_synth_series", "prefer_stress"]
    counts = ok.groupby(keys)["method"].nunique()
    keep = counts[counts == 2].reset_index()[keys]
    return ok.merge(keep, on=keys, how="inner")


def delta_vs_n0(df: pd.DataFrame) -> pd.DataFrame:
    """Paired change vs n_synth=0 within the same method and window."""
    ok = valid_windows(df)
    keys = ["method", "test_start", "prefer_stress"]
    metrics = [c for c in ["sharpe", "calmar", "max_drawdown", "CVaR_5%", "hhi"] if c in ok.columns]
    base = ok.loc[ok["n_synth_series"] == 0, keys + metrics].rename(
        columns={c: f"{c}_n0" for c in metrics}
    )
    aug = ok.loc[ok["n_synth_series"] > 0].copy()
    merged = aug.merge(base, on=keys, how="inner")
    for c in metrics:
        merged[f"delta_{c}"] = merged[c] - merged[f"{c}_n0"]
    return merged


def regime_minus_mixed(df: pd.DataFrame) -> pd.DataFrame:
    """Paired regime − mixed on the same window and n_synth."""
    ok = both_methods(df)
    keys = ["test_start", "n_synth_series", "prefer_stress"]
    metrics = [c for c in ["sharpe", "calmar", "max_drawdown", "CVaR_5%", "hhi"] if c in ok.columns]
    extra = [
        c
        for c in ["bucket", "vol_bucket", "forecast_correct", "predicted_regime", "realized_dominant_regime"]
        if c in ok.columns
    ]
    reg = ok.loc[ok["method"].eq("regime"), keys + extra + metrics].rename(
        columns={c: f"{c}_regime" for c in metrics}
    )
    mix = ok.loc[ok["method"].eq("mixed"), keys + metrics].rename(
        columns={c: f"{c}_mixed" for c in metrics}
    )
    merged = reg.merge(mix, on=keys, how="inner")
    for c in metrics:
        merged[f"delta_{c}"] = merged[f"{c}_regime"] - merged[f"{c}_mixed"]
    return merged


def stitch_wealth(panel: pd.DataFrame) -> pd.DataFrame:
    """Turn non-overlapping window returns into cumulative wealth by method and n."""
    rows = []
    group_cols = ["method", "n_synth_series", "prefer_stress"]
    for key, grp in panel.groupby(group_cols):
        key = key if isinstance(key, tuple) else (key,)
        meta = dict(zip(group_cols, key))
        series = grp.sort_values("date").drop_duplicates("date").set_index("date")["port_ret"]
        wealth = (1.0 + series).cumprod()
        peak = wealth.cummax()
        dd = wealth / peak - 1.0
        part = pd.DataFrame(
            {
                **meta,
                "date": wealth.index,
                "port_ret": series.reindex(wealth.index).to_numpy(),
                "wealth": wealth.to_numpy(),
                "drawdown": dd.to_numpy(),
            }
        )
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def summarize_winrates(delta: pd.DataFrame, by: Sequence[str]) -> pd.DataFrame:
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
            row[f"{c}_median"] = float(vals.median()) if len(vals) else np.nan
            row[f"{c}_winrate"] = float((vals > 0).mean()) if len(vals) else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
