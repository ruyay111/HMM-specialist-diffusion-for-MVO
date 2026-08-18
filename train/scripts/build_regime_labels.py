#!/usr/bin/env python3
"""Phase 1: 5-regime labels for specialist training.

Default range matches the calendar UniTST_MP sample: 2001-01-01 → 2022-08-31
(no evaluation-window date filter). The MVO backtest still refits the HMM
causally at each rebalance and does not reuse these labels at test time.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
EVAL_DIR = REPO_ROOT / "evaluation"
sys.path.insert(0, str(EVAL_DIR))

from supervised_hmm import identify_regimes  # noqa: E402
from portfolio_core import pct_returns  # noqa: E402
from regime_hmm_mvo import market_regime_signal, standardize_history  # noqa: E402

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
        "--prices-csv",
        default=str(EVAL_DIR / "data" / "benchmark" / "benchmark_data.csv"),
    )
    parser.add_argument(
        "--start-date",
        default="2001-01-01",
        help="Inclusive start date for offline specialist labels (calendar UniTST_MP sample).",
    )
    parser.add_argument("--end-date", default="2022-08-31")
    parser.add_argument("--n-regimes", type=int, default=5)
    parser.add_argument("--vol-window", type=int, default=20)
    parser.add_argument("--min-segment", type=int, default=32)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--output-dir",
        default=str(REPO_ROOT / "train" / "data"),
    )
    return parser.parse_args()


def load_evaluation_returns(
    prices_csv: Path,
    start_date: str,
    end_date: str,
    asset_cols: list[str],
) -> pd.DataFrame:
    prices = pd.read_csv(prices_csv)
    prices = prices.rename(columns={"as_of": "date"})
    prices["date"] = pd.to_datetime(prices["date"])
    prices = prices.loc[
        (prices["date"] >= start_date) & (prices["date"] <= end_date),
        ["date", *asset_cols],
    ].set_index("date")
    missing = [col for col in asset_cols if col not in prices.columns]
    if missing:
        raise ValueError(f"Missing asset columns: {missing}")
    returns = pct_returns(prices[asset_cols])
    if returns.empty:
        raise ValueError("No returns in the requested date range.")
    return returns


def main() -> int:
    args = parse_args()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    returns = load_evaluation_returns(
        Path(args.prices_csv).expanduser().resolve(),
        args.start_date,
        args.end_date,
        ASSET_COLS,
    )
    market = market_regime_signal(returns, ASSET_COLS)
    standardized, market_mean, market_std = standardize_history(market)
    labels, diagnostics = identify_regimes(
        standardized.to_numpy(),
        n_regimes=args.n_regimes,
        volatility_window=args.vol_window,
        min_segment_length=args.min_segment,
        random_state=args.random_state,
    )
    centers = np.asarray(diagnostics["regime_mean_volatility"], dtype=float)

    label_frame = pd.DataFrame(
        {
            "date": returns.index,
            "regime": labels.astype(int),
            "market_return": market.to_numpy(dtype=float),
            "standardized_market_return": standardized.to_numpy(dtype=float),
        }
    )
    label_path = output_dir / "regime_labels_5.csv"
    center_path = output_dir / "regime_vol_centers.npy"
    meta_path = output_dir / "regime_labels_5.meta.json"

    label_frame.to_csv(label_path, index=False)
    np.save(center_path, centers)

    counts = label_frame["regime"].value_counts().sort_index()
    metadata = {
        "prices_csv": str(Path(args.prices_csv).expanduser().resolve()),
        "start_date": args.start_date,
        "end_date": args.end_date,
        "return_start": str(returns.index[0].date()),
        "return_end": str(returns.index[-1].date()),
        "n_days": int(len(returns)),
        "n_regimes": int(args.n_regimes),
        "vol_window": int(args.vol_window),
        "min_segment": int(args.min_segment),
        "random_state": int(args.random_state),
        "assets": ASSET_COLS,
        "market_mean": float(market_mean),
        "market_std": float(market_std),
        "regime_mean_volatility": centers.tolist(),
        "n_change_points": int(len(diagnostics["change_points"])),
        "regime_counts": {int(k): int(v) for k, v in counts.items()},
        "note": (
            "Offline labels on the full specialist sample (default 2001-01-01 to "
            "2022-08-31), matching calendar UniTST_MP. Rolling MVO still refits "
            "the HMM causally and does not reuse these labels at test time."
        ),
    }
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[OK] labels {label_path}")
    print(f"[OK] centers {center_path}")
    print(
        f"[INFO] {returns.index[0].date()} -> {returns.index[-1].date()} "
        f"n={len(returns)}"
    )
    for regime, count in counts.items():
        print(f"[INFO] regime {int(regime)}: {int(count)} days")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
