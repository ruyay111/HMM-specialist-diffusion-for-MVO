"""HMM-path specialist synthesis: sample a regime path and stitch specialist windows.

This is not calendar date matching. Generated windows are drawn from offline
per-regime pools and concatenated along a sampled HMM path.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def sample_hmm_path(
    filtered_state: np.ndarray,
    transmat: np.ndarray,
    horizon: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Sample a length-``horizon`` regime path from the causal filtered state."""
    current = np.asarray(filtered_state, dtype=float).reshape(-1)
    transmat = np.asarray(transmat, dtype=float)
    n_regimes = current.shape[0]
    if transmat.shape != (n_regimes, n_regimes):
        raise ValueError(
            f"transmat shape {transmat.shape} does not match filtered_state length {n_regimes}"
        )
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    current = current / current.sum()
    path = np.empty(int(horizon), dtype=int)
    path[0] = int(rng.choice(n_regimes, p=current))
    for t in range(1, int(horizon)):
        row = transmat[path[t - 1]]
        row = row / row.sum()
        path[t] = int(rng.choice(n_regimes, p=row))
    return path


def _contiguous_slice(
    window: np.ndarray,
    length: int,
    rng: np.random.Generator,
) -> np.ndarray:
    seq_len = int(window.shape[0])
    if length > seq_len:
        raise ValueError(f"Need {length} days from a window of length {seq_len}")
    start = 0 if length == seq_len else int(rng.integers(0, seq_len - length + 1))
    return window[start : start + length]


def stitch_specialist_path(
    regime_path: np.ndarray,
    pools: dict[int, np.ndarray],
    rng: np.random.Generator,
) -> np.ndarray:
    """Stitch specialist windows along a sampled regime path.

    Consecutive days with the same regime are filled from one or more windows
    from that regime's pool. Different regimes never share a window.
    """
    regime_path = np.asarray(regime_path, dtype=int).reshape(-1)
    if regime_path.size == 0:
        raise ValueError("regime_path is empty")
    sample_pool = next(iter(pools.values()))
    n_assets = int(sample_pool.shape[-1])
    seq_len = int(sample_pool.shape[1])
    out = np.zeros((len(regime_path), n_assets), dtype=float)

    t = 0
    while t < len(regime_path):
        regime = int(regime_path[t])
        end = t + 1
        while end < len(regime_path) and int(regime_path[end]) == regime:
            end += 1
        need = end - t
        if regime not in pools:
            raise KeyError(f"No specialist pool for regime {regime}")
        pool = np.asarray(pools[regime])
        if pool.ndim != 3 or pool.shape[0] == 0:
            raise ValueError(f"Specialist pool for regime {regime} is empty")
        filled = 0
        while filled < need:
            window = pool[int(rng.integers(0, pool.shape[0]))]
            take = min(int(seq_len), need - filled)
            out[t + filled : t + filled + take] = _contiguous_slice(window, take, rng)
            filled += take
        t = end
    return out


def sample_stitched_paths(
    filtered_state: np.ndarray,
    transmat: np.ndarray,
    pools: dict[int, np.ndarray],
    n_paths: int,
    horizon: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(n_paths, horizon, n_assets)`` stitched paths and the regime paths."""
    if int(n_paths) <= 0:
        n_assets = int(next(iter(pools.values())).shape[-1])
        return (
            np.empty((0, int(horizon), n_assets), dtype=float),
            np.empty((0, int(horizon)), dtype=int),
        )
    paths = []
    regimes = []
    for _ in range(int(n_paths)):
        regime_path = sample_hmm_path(filtered_state, transmat, horizon, rng)
        paths.append(stitch_specialist_path(regime_path, pools, rng))
        regimes.append(regime_path)
    return np.stack(paths, axis=0), np.stack(regimes, axis=0)


def load_specialist_pools(
    pool_root: str | Path,
    n_regimes: int = 5,
) -> dict[int, np.ndarray]:
    """Load ``regime_k{k}/windows.npy`` arrays from an offline pool directory."""
    pool_root = Path(pool_root)
    pools: dict[int, np.ndarray] = {}
    for regime in range(int(n_regimes)):
        path = pool_root / f"regime_k{regime}" / "windows.npy"
        if not path.exists():
            raise FileNotFoundError(f"Missing specialist pool {path}")
        windows = np.load(path)
        if windows.ndim != 3 or windows.shape[0] == 0:
            raise ValueError(f"Invalid pool at {path}: shape={windows.shape}")
        pools[regime] = windows
    return pools


def pool_meta(pool_root: str | Path, regime: int) -> dict:
    meta_path = Path(pool_root) / f"regime_k{regime}" / "meta.json"
    if not meta_path.exists():
        return {}
    return json.loads(meta_path.read_text(encoding="utf-8"))
