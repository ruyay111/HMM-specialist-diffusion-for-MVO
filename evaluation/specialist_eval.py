"""Data_5-style experiments using HMM-path specialist pools (no date matching)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np
import pandas as pd

from portfolio_core import (
    calmar_ratio,
    classify_regime,
    collapse_weights,
    mean_var_weights,
    mix_train_with_regime_paths,
    sharpe_ratio,
    summarize_portfolio,
)
from regime_hmm_mvo import fit_regime_context, standardize_history
from stress_test_final import apply_shock_to_returns
from synth_path_builder import load_specialist_pools, sample_stitched_paths

ASSET_COLS = [
    "A001",
    "A004",
    "A006",
    "A008",
    "A009",
    "A011",
    "A012",
    "A013",
    "A014",
    "A015",
]
PATH_HORIZON = 128
DEFAULT_SCENARIOS = [
    ("BASE", {"shock_type": "VOL_SCALE", "vol_scale": 1.0}),
    ("VOL_SCALE_1.8x", {"shock_type": "VOL_SCALE", "vol_scale": 1.8}),
    ("NEG_SHOCK_-10%", {"shock_type": "NEG_RETURN_SHOCK", "shock_size": 0.10}),
    ("CORR_INFLATE_1.3", {"shock_type": "CORR_INFLATE", "corr_inflate": 1.3}),
    ("WHIPSAW", {"shock_type": "WHIPSAW"}),
    ("SLOW_BLEED", {"shock_type": "SLOW_BLEED"}),
    ("CRASH_2008", {"shock_type": "crash_2008"}),
]


def load_regime_pools(
    project_root: Path,
    n_regimes: int = 5,
) -> tuple[dict[int, np.ndarray], Path, str]:
    """Prefer generated specialist pools; fall back to training windows."""
    pool_root = project_root / "train" / "pools"
    window_root = project_root / "train" / "data" / "regime_windows"
    generated = all(
        (pool_root / f"regime_k{k}" / "windows.npy").exists() for k in range(n_regimes)
    )
    if generated:
        return load_specialist_pools(pool_root, n_regimes=n_regimes), pool_root, "generated"
    pools: dict[int, np.ndarray] = {}
    for k in range(n_regimes):
        path = window_root / f"regime_{k}.npy"
        if not path.exists():
            raise FileNotFoundError(
                f"Missing {path} and generated pools under {pool_root}"
            )
        windows = np.load(path)
        if windows.ndim != 3 or windows.shape[0] == 0:
            raise ValueError(f"Empty or invalid window file {path}")
        pools[k] = windows
    return pools, window_root, "training_windows"


def _hmm_context(real_rets: pd.DataFrame, asset_cols: Sequence[str], history_end: int, invest: int, random_state: int) -> dict:
    context = fit_regime_context(
        real_rets,
        list(asset_cols),
        history_end=int(history_end),
        invest=int(invest),
        random_state=int(random_state),
    )
    standardized, _, _ = standardize_history(context["history"].mean(axis=1))
    context["standardized"] = standardized.to_numpy(dtype=float)
    return context


def specialist_weights(
    real_train: pd.DataFrame,
    context: dict,
    pools: dict[int, np.ndarray],
    n_synth: int,
    seed: int,
    allow_short: bool = False,
    max_weight: float | None = None,
    path_horizon: int = PATH_HORIZON,
) -> np.ndarray:
    """Column-stack last real days with HMM-path specialist series, then collapse."""
    n_assets = int(real_train.shape[1])
    rng = np.random.default_rng(int(seed))
    if int(n_synth) <= 0:
        mixed = real_train.to_numpy(dtype=float)
        n_draw = 0
    else:
        alpha = context["hmm_model"].predict_proba(
            context["standardized"],
            smoothed=False,
        )[-1]
        paths, _ = sample_stitched_paths(
            alpha,
            context["hmm_model"].transmat_,
            pools,
            n_paths=int(n_synth),
            horizon=int(path_horizon),
            rng=rng,
        )
        mixed, n_draw = mix_train_with_regime_paths(
            real_train.to_numpy(dtype=float),
            paths,
            seq_len=int(path_horizon),
            rng=rng,
        )
    weights = mean_var_weights(
        mixed,
        allow_short=allow_short,
        max_weight=max_weight,
        n_assets_orig=n_assets,
        n_draw=n_draw,
    )
    return collapse_weights(weights, n_assets, n_draw)


def rolling_experiment(
    real_rets: pd.DataFrame,
    pools: dict[int, np.ndarray],
    asset_cols: Sequence[str] = ASSET_COLS,
    mix_grid: Sequence[int] = (0, 1, 2, 5, 10, 15, 20),
    lookback: int = 252,
    invest: int = 60,
    step: int = 21,
    seeds: Sequence[int] = (42,),
    allow_short: bool = False,
    max_weight: float | None = None,
    random_state: int = 42,
) -> pd.DataFrame:
    """Rolling mixed-lookback MVO with specialist-path synth instead of calendar dates."""
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols].dropna(how="any")
    rows = []
    for start in range(0, len(returns) - lookback - invest + 1, step):
        history_end = start + lookback
        test = returns.iloc[history_end : history_end + invest]
        real_train = returns.iloc[start:history_end]
        regime = classify_regime(test.index[0], test.index[-1])
        spy = test[asset_cols[0]].to_numpy()
        context = _hmm_context(returns, asset_cols, history_end, invest, random_state)
        for seed in seeds:
            for n_synth in mix_grid:
                weights = specialist_weights(
                    real_train,
                    context,
                    pools,
                    n_synth=int(n_synth),
                    seed=int(seed) + start + int(n_synth),
                    allow_short=allow_short,
                    max_weight=max_weight,
                )
                port = pd.Series(test.to_numpy() @ weights, index=test.index)
                row = summarize_portfolio(port, label=regime)
                row.update(
                    {
                        "regime": regime,
                        "n_synth_series": int(n_synth),
                        "test_start": test.index[0],
                        "seed": int(seed),
                        "max_weight_used": float(np.max(np.abs(weights))),
                        "spy_sharpe": sharpe_ratio(spy),
                        "spy_calmar": calmar_ratio(spy),
                    }
                )
                rows.append(row)
    return pd.DataFrame(rows)


def paired_deltas(results: pd.DataFrame) -> pd.DataFrame:
    keys = ["test_start", "regime", "seed"] if "seed" in results.columns else ["test_start", "regime"]
    keys = [k for k in keys if k in results.columns]
    base = results.loc[results["n_synth_series"] == 0, keys + ["sharpe", "calmar", "max_drawdown"]].rename(
        columns={"sharpe": "sharpe_n0", "calmar": "calmar_n0", "max_drawdown": "max_drawdown_n0"}
    )
    aug = results.loc[results["n_synth_series"] > 0].merge(base, on=keys, how="inner")
    aug["delta_sharpe"] = aug["sharpe"] - aug["sharpe_n0"]
    aug["delta_calmar"] = aug["calmar"] - aug["calmar_n0"]
    aug["delta_max_drawdown"] = aug["max_drawdown"] - aug["max_drawdown_n0"]
    return (
        aug.groupby(["regime", "n_synth_series"])
        .agg(
            n_windows=("delta_sharpe", "size"),
            delta_sharpe_mean=("delta_sharpe", "mean"),
            delta_sharpe_winrate=("delta_sharpe", lambda s: float((s > 0).mean())),
            delta_calmar_mean=("delta_calmar", "mean"),
            delta_calmar_winrate=("delta_calmar", lambda s: float((s > 0).mean())),
            delta_max_dd_mean=("delta_max_drawdown", "mean"),
        )
        .round(4)
    )


def shock_experiment(
    real_rets: pd.DataFrame,
    pools: dict[int, np.ndarray],
    asset_cols: Sequence[str] = ASSET_COLS,
    mix_grid: Sequence[int] = (0, 1, 2, 5, 10, 15, 20),
    lookback: int = 252,
    invest: int = 300,
    step: int = 21,
    seed: int = 42,
    random_state: int = 42,
    scenarios: Sequence[tuple] | None = None,
) -> pd.DataFrame:
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols].dropna(how="any")
    used = list(scenarios) if scenarios is not None else DEFAULT_SCENARIOS
    rows = []
    for start in range(0, len(returns) - lookback - invest + 1, step):
        history_end = start + lookback
        real_train = returns.iloc[start:history_end]
        test = returns.iloc[history_end : history_end + invest]
        context = _hmm_context(returns, asset_cols, history_end, invest=60, random_state=random_state)
        for n_synth in mix_grid:
            weights = specialist_weights(
                real_train,
                context,
                pools,
                n_synth=int(n_synth),
                seed=int(seed) + start + int(n_synth),
            )
            for name, kwargs in used:
                if name == "BASE":
                    shocked = test
                else:
                    shocked = apply_shock_to_returns(test, seed=seed, **kwargs)
                port = pd.Series(shocked.to_numpy() @ weights, index=shocked.index)
                row = summarize_portfolio(port, label=name)
                row["n_synth_series"] = int(n_synth)
                row["test_start"] = test.index[0]
                rows.append(row)
    return pd.DataFrame(rows)


def daily_paths(
    real_rets: pd.DataFrame,
    pools: dict[int, np.ndarray],
    asset_cols: Sequence[str] = ASSET_COLS,
    n_synth_list: Sequence[int] = (0, 1, 2, 5, 20),
    lookback: int = 252,
    test_start: int | None = None,
    test_end: int | None = None,
    hmm_step: int = 21,
    seed: int = 42,
    allow_short: bool = False,
    max_weight: float | None = None,
    random_state: int = 42,
) -> dict[int, pd.Series]:
    """Daily holdings; HMM and weights refresh every ``hmm_step`` days."""
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols].dropna(how="any")
    start = int(lookback if test_start is None else test_start)
    end = int(len(returns) if test_end is None else test_end)
    port = {int(n): [] for n in n_synth_list}
    dates = []
    weights = {}
    context = None
    for t in range(start, end):
        if context is None or (t - start) % int(hmm_step) == 0:
            context = _hmm_context(returns, asset_cols, t, invest=60, random_state=random_state)
            real_train = returns.iloc[t - lookback : t]
            for n in n_synth_list:
                weights[int(n)] = specialist_weights(
                    real_train,
                    context,
                    pools,
                    n_synth=int(n),
                    seed=int(seed) + t + int(n),
                    allow_short=allow_short,
                    max_weight=max_weight,
                )
        day = returns.iloc[t]
        dates.append(returns.index[t])
        for n in n_synth_list:
            port[int(n)].append(float(day.to_numpy() @ weights[int(n)]))
    return {
        int(n): pd.Series(port[int(n)], index=dates, name=f"n_synth={n}")
        for n in n_synth_list
    }


def static_paths(
    real_rets: pd.DataFrame,
    pools: dict[int, np.ndarray],
    asset_cols: Sequence[str] = ASSET_COLS,
    n_synth_list: Sequence[int] = (0, 1, 2, 5, 20),
    test_start: int = 0,
    seed: int = 42,
    allow_short: bool = False,
    max_weight: float | None = None,
    random_state: int = 42,
) -> dict[int, pd.Series]:
    asset_cols = list(asset_cols)
    returns = real_rets[asset_cols].dropna(how="any")
    train = returns.iloc[: int(test_start)]
    test = returns.iloc[int(test_start) :]
    context = _hmm_context(returns, asset_cols, int(test_start), invest=60, random_state=random_state)
    out = {}
    for n in n_synth_list:
        weights = specialist_weights(
            train,
            context,
            pools,
            n_synth=int(n),
            seed=int(seed) + int(n),
            allow_short=allow_short,
            max_weight=max_weight,
        )
        out[int(n)] = pd.Series(test.to_numpy() @ weights, index=test.index, name=f"n_synth={n}")
    return out


def path_metrics(paths: dict[int, pd.Series], benchmark: pd.Series, benchmark_label: str) -> pd.DataFrame:
    rows = [summarize_portfolio(series, label=f"n_synth={n}") for n, series in paths.items()]
    rows.append(summarize_portfolio(benchmark, label=benchmark_label))
    return pd.DataFrame(rows).set_index("bucket")


def specialist_corr_matrix(pools: dict[int, np.ndarray], rng_seed: int = 42) -> np.ndarray:
    """One representative panel: one window from each regime, concatenated."""
    rng = np.random.default_rng(rng_seed)
    blocks = []
    for regime in sorted(pools):
        pool = pools[regime]
        blocks.append(pool[int(rng.integers(0, pool.shape[0]))])
    return np.corrcoef(np.concatenate(blocks, axis=0).T)
