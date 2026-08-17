#!/usr/bin/env python3
"""Phase 2: extract contiguous per-regime windows for specialist training.

Input log returns: train/data/benchmark_data_log_ret_10.csv
Input labels: train/data/regime_labels_paper_5.csv
Output: train/data/regime_windows/regime_{k}.npy of shape (N_k, seq_len, 10)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--returns-csv",
        default=str(REPO_ROOT / "train" / "data" / "benchmark_data_log_ret_10.csv"),
    )
    parser.add_argument(
        "--labels-csv",
        default=str(REPO_ROOT / "train" / "data" / "regime_labels_paper_5.csv"),
    )
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--stride", type=int, default=1)
    parser.add_argument("--n-regimes", type=int, default=5)
    parser.add_argument(
        "--short-policy",
        choices=("skip", "tile"),
        default="tile",
        help=(
            "skip: drop segments shorter than seq_len (plan v1 purity). "
            "tile: cyclically repeat short same-regime segments to seq_len so "
            "high-vol specialists (k=3,4) can train."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "train" / "data" / "regime_windows"),
    )
    return parser.parse_args()


def contiguous_runs(labels: np.ndarray) -> list[tuple[int, int, int]]:
    """Return (start, end, regime) half-open index runs."""
    if labels.size == 0:
        return []
    runs: list[tuple[int, int, int]] = []
    start = 0
    current = int(labels[0])
    for idx, value in enumerate(labels[1:], start=1):
        value = int(value)
        if value != current:
            runs.append((start, idx, current))
            start = idx
            current = value
    runs.append((start, len(labels), current))
    return runs


def sliding_windows(block: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    n_days, n_assets = block.shape
    if n_days < seq_len:
        return np.empty((0, seq_len, n_assets), dtype=float)
    starts = range(0, n_days - seq_len + 1, stride)
    return np.stack([block[s : s + seq_len] for s in starts], axis=0)


def tiled_windows(block: np.ndarray, seq_len: int, stride: int) -> np.ndarray:
    """Build seq_len windows by cyclic tiling of a short contiguous segment."""
    n_days, n_assets = block.shape
    if n_days == 0:
        return np.empty((0, seq_len, n_assets), dtype=float)
    if n_days >= seq_len:
        return sliding_windows(block, seq_len, stride)
    windows = []
    for offset in range(0, n_days, max(1, stride)):
        rotated = np.concatenate([block[offset:], block[:offset]], axis=0)
        reps = int(np.ceil(seq_len / n_days))
        tiled = np.concatenate([rotated] * reps, axis=0)[:seq_len]
        windows.append(tiled)
    return np.stack(windows, axis=0)


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = pd.read_csv(args.returns_csv, parse_dates=["date"]).set_index("date")
    labels = pd.read_csv(args.labels_csv, parse_dates=["date"]).set_index("date")
    missing = [col for col in ASSET_COLS if col not in returns.columns]
    if missing:
        raise ValueError(f"Missing return columns: {missing}")

    common = returns.index.intersection(labels.index)
    returns = returns.loc[common, ASSET_COLS]
    labels = labels.loc[common, "regime"].astype(int)
    if len(returns) == 0:
        raise ValueError("No overlapping dates between returns and labels.")

    seq_len = int(args.seq_len)
    stride = int(args.stride)
    n_regimes = int(args.n_regimes)
    windows_by_regime: dict[int, list[np.ndarray]] = {k: [] for k in range(n_regimes)}
    segments_kept = {k: 0 for k in range(n_regimes)}
    segments_skipped = {k: 0 for k in range(n_regimes)}

    for start, end, regime in contiguous_runs(labels.to_numpy()):
        if regime < 0 or regime >= n_regimes:
            raise ValueError(f"Unexpected regime label {regime}")
        length = end - start
        block = returns.iloc[start:end].to_numpy(dtype=float)
        if length < seq_len:
            if args.short_policy == "skip":
                segments_skipped[regime] += 1
                continue
            windows = tiled_windows(block, seq_len=seq_len, stride=stride)
        else:
            windows = sliding_windows(block, seq_len=seq_len, stride=stride)
        if windows.shape[0] == 0:
            segments_skipped[regime] += 1
            continue
        windows_by_regime[regime].append(windows)
        segments_kept[regime] += 1

    manifest: dict = {
        "returns_csv": str(Path(args.returns_csv).expanduser().resolve()),
        "labels_csv": str(Path(args.labels_csv).expanduser().resolve()),
        "seq_len": seq_len,
        "stride": stride,
        "short_policy": args.short_policy,
        "n_regimes": n_regimes,
        "n_assets": len(ASSET_COLS),
        "assets": ASSET_COLS,
        "aligned_start": str(returns.index[0].date()),
        "aligned_end": str(returns.index[-1].date()),
        "n_aligned_days": int(len(returns)),
        "regimes": {},
    }

    for regime in range(n_regimes):
        if windows_by_regime[regime]:
            stacked = np.concatenate(windows_by_regime[regime], axis=0)
        else:
            stacked = np.empty((0, seq_len, len(ASSET_COLS)), dtype=float)
        out_path = output_dir / f"regime_{regime}.npy"
        np.save(out_path, stacked)
        manifest["regimes"][str(regime)] = {
            "n_windows": int(stacked.shape[0]),
            "n_segments_kept": int(segments_kept[regime]),
            "n_segments_skipped_short": int(segments_skipped[regime]),
            "path": str(out_path),
        }
        print(
            f"[INFO] regime {regime}: {stacked.shape[0]} windows "
            f"from {segments_kept[regime]} segments "
            f"({segments_skipped[regime]} short segments skipped)"
        )
        if stacked.shape[0] == 0:
            print(
                f"[WARN] regime {regime} has no {seq_len}-day contiguous "
                "windows; specialist training for this regime will be skipped."
            )

    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(f"[OK] wrote {manifest_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
