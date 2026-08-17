#!/usr/bin/env python3
"""Convert diffusion test generated_data.csv into Notebook_Data_5 data5 format."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


ASSET_COLS_10 = [
    "A001", "A004", "A006", "A008", "A009",
    "A011", "A012", "A013", "A014", "A015",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--generated-csv",
        required=True,
        help="Path to generated_data.csv from exp.test(save_data=True).",
    )
    parser.add_argument(
        "--benchmark-csv",
        default="./data/benchmark_data.csv",
        help="Benchmark price CSV used to build the evaluation date index.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory for per-asset synthetic CSV files.",
    )
    parser.add_argument("--num-sequences", type=int, default=20)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--date-start", default="2013-01-02")
    parser.add_argument("--date-end", default="2022-08-31")
    return parser.parse_args()


def load_windows(generated_csv: Path, seq_len: int, n_assets: int) -> np.ndarray:
    flat = pd.read_csv(generated_csv).to_numpy(dtype=float)
    if flat.shape[1] != n_assets:
        raise ValueError(f"Expected {n_assets} asset columns, got {flat.shape[1]}")
    if flat.shape[0] % seq_len != 0:
        raise ValueError(
            f"Row count {flat.shape[0]} is not divisible by seq_len={seq_len}"
        )
    n_windows = flat.shape[0] // seq_len
    return flat.reshape(n_windows, seq_len, n_assets)


def load_dates(benchmark_csv: Path, date_start: str, date_end: str) -> np.ndarray:
    benchmark = pd.read_csv(benchmark_csv)
    benchmark = benchmark.rename(columns={"as_of": "date"})
    benchmark["date"] = pd.to_datetime(benchmark["date"])
    dates = benchmark.loc[
        (benchmark["date"] >= date_start) & (benchmark["date"] <= date_end),
        "date",
    ].to_numpy()
    if len(dates) == 0:
        raise ValueError("No benchmark dates found in the requested range.")
    return dates


def bootstrap_sequence(
    windows: np.ndarray,
    target_len: int,
    rng: np.random.Generator,
) -> np.ndarray:
    chunks: list[np.ndarray] = []
    total = 0
    while total < target_len:
        idx = int(rng.integers(0, windows.shape[0]))
        chunk = windows[idx]
        chunks.append(chunk)
        total += chunk.shape[0]
    path = np.concatenate(chunks, axis=0)
    return path[:target_len]


def write_asset_csvs(
    output_dir: Path,
    dates: np.ndarray,
    assets: list[str],
    sequences: np.ndarray,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    columns = [f"Sequence_{idx + 1}" for idx in range(sequences.shape[0])]
    target_len = len(dates)

    for asset_idx, asset in enumerate(assets):
        values = sequences[:, :target_len, asset_idx].T
        out = pd.DataFrame(values, columns=columns)
        out.insert(0, "date", pd.to_datetime(dates).strftime("%Y-%m-%d"))
        out_path = output_dir / f"{asset}_synthetic_log_returns_2013_2022_{len(columns)}seq.csv"
        out.to_csv(out_path, index=False)


def main() -> int:
    args = parse_args()
    generated_csv = Path(args.generated_csv).expanduser().resolve()
    benchmark_csv = Path(args.benchmark_csv).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    windows = load_windows(generated_csv, args.seq_len, len(ASSET_COLS_10))
    dates = load_dates(benchmark_csv, args.date_start, args.date_end)

    # Match the existing data5 convention: 2521 business-day rows from 2013-01-02.
    if len(dates) > 2521:
        dates = dates[1:2522]
    target_len = len(dates)

    rng = np.random.default_rng(args.seed)
    sequences = np.zeros((args.num_sequences, target_len, len(ASSET_COLS_10)), dtype=float)
    for seq_idx in range(args.num_sequences):
        sequences[seq_idx] = bootstrap_sequence(windows, target_len, rng)

    write_asset_csvs(output_dir, dates, ASSET_COLS_10, sequences)

    metadata = {
        "source_generated_csv": str(generated_csv),
        "benchmark_csv": str(benchmark_csv),
        "assets": ASSET_COLS_10,
        "num_sequences": args.num_sequences,
        "num_windows": int(windows.shape[0]),
        "window_len": args.seq_len,
        "target_len": target_len,
        "date_start": str(pd.Timestamp(dates[0]).date()),
        "date_end": str(pd.Timestamp(dates[-1]).date()),
        "bootstrap_seed": args.seed,
        "method": "multivariate_window_bootstrap",
    }
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"[OK] Wrote {len(ASSET_COLS_10)} asset files to {output_dir}")
    print(f"[INFO] windows={windows.shape[0]} x {args.seq_len}, paths={args.num_sequences} x {target_len}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
