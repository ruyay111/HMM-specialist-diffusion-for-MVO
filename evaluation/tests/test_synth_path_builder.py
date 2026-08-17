"""Unit tests for HMM-path specialist stitching (no trained diffusion required)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

EVAL_DIR = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(EVAL_DIR))

from portfolio_core import mix_train_with_regime_paths
from synth_path_builder import (  # noqa: E402
    sample_hmm_path,
    sample_stitched_paths,
    stitch_specialist_path,
)


def _fake_pools(n_regimes: int = 5, n_windows: int = 6, seq_len: int = 128, n_assets: int = 10):
    pools = {}
    for regime in range(n_regimes):
        windows = np.full((n_windows, seq_len, n_assets), float(regime), dtype=float)
        windows += np.linspace(0.0, 0.01, seq_len)[None, :, None]
        pools[regime] = windows
    return pools


def test_sample_hmm_path_length_and_support():
    rng = np.random.default_rng(0)
    transmat = np.eye(3)
    path = sample_hmm_path(np.array([0.0, 1.0, 0.0]), transmat, horizon=12, rng=rng)
    assert path.shape == (12,)
    assert set(path.tolist()) <= {0, 1, 2}
    assert np.all(path == 1)


def test_stitch_uses_matching_regime_pool():
    rng = np.random.default_rng(1)
    pools = _fake_pools()
    path = np.array([0] * 40 + [4] * 20 + [2] * 68, dtype=int)
    stitched = stitch_specialist_path(path, pools, rng)
    assert stitched.shape == (128, 10)
    assert np.allclose(np.round(stitched[:40, 0]), 0)
    assert np.allclose(np.round(stitched[40:60, 0]), 4)
    assert np.allclose(np.round(stitched[60:, 0]), 2)


def test_long_run_is_filled_from_multiple_windows():
    rng = np.random.default_rng(2)
    pools = _fake_pools(seq_len=128)
    path = np.zeros(200, dtype=int)
    stitched = stitch_specialist_path(path, pools, rng)
    assert stitched.shape == (200, 10)
    assert np.allclose(np.round(stitched[:, 0]), 0)


def test_sample_stitched_paths_batch_shape():
    rng = np.random.default_rng(3)
    pools = _fake_pools()
    transmat = np.full((5, 5), 0.05)
    np.fill_diagonal(transmat, 0.8)
    transmat = transmat / transmat.sum(axis=1, keepdims=True)
    alpha = np.array([0.1, 0.1, 0.6, 0.1, 0.1])
    windows, regimes = sample_stitched_paths(
        alpha, transmat, pools, n_paths=4, horizon=128, rng=rng
    )
    assert windows.shape == (4, 128, 10)
    assert regimes.shape == (4, 128)
    assert set(np.unique(regimes).tolist()) <= set(range(5))


def test_mix_train_with_regime_paths_column_stack_no_dates():
    rng = np.random.default_rng(4)
    real = np.ones((80, 10), dtype=float)
    synth = np.full((3, 128, 10), 2.0, dtype=float)
    mixed, n_draw = mix_train_with_regime_paths(real, synth, seq_len=128, rng=rng)
    assert n_draw == 3
    assert mixed.shape == (80, 10 + 10 * 3)
    assert np.allclose(mixed[:, :10], 1.0)
    assert np.allclose(mixed[:, 10:], 2.0)


if __name__ == "__main__":
    test_sample_hmm_path_length_and_support()
    test_stitch_uses_matching_regime_pool()
    test_long_run_is_filled_from_multiple_windows()
    test_sample_stitched_paths_batch_shape()
    test_mix_train_with_regime_paths_column_stack_no_dates()
    print("ok")
