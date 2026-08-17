#!/usr/bin/env python3
"""Phase 4: sample offline specialist pools from trained UniTST_MP checkpoints.

Saves train/pools/regime_k{k}/windows.npy with shape (N_pool, 128, 10) in
log-return space, plus meta.json. Does not date-match generated windows.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from argparse import Namespace
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

TRAIN_DIR = Path(__file__).resolve().parents[1]
os.chdir(TRAIN_DIR)
sys.path.insert(0, str(TRAIN_DIR))

from src.exp.exp_diffusion_denoised_x import Exp_Diffusion_Denoised_X  # noqa: E402
from src.utils.utils import process_model_dict  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-regimes", type=int, default=5)
    parser.add_argument("--n-pool", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--n-assets", type=int, default=10)
    parser.add_argument("--sample-step", type=int, default=450)
    parser.add_argument("--temperature", type=float, default=0.7473231454237225)
    parser.add_argument("--checkpoints-root", default="./checkpoints")
    parser.add_argument("--output-dir", default="./pools")
    parser.add_argument(
        "--description-template",
        default="specialist_regime_{k}",
        help="Must match --description used during specialist training.",
    )
    return parser.parse_args()


def _load_args_json(checkpoint_dir: Path) -> Namespace:
    args_path = checkpoint_dir / "args.json"
    if not args_path.exists():
        raise FileNotFoundError(f"Missing {args_path}")
    payload = json.loads(args_path.read_text(encoding="utf-8"))
    return Namespace(**payload)


def _find_checkpoint_dir(checkpoints_root: Path, description: str) -> Path:
    matches = [
        path
        for path in checkpoints_root.glob("*")
        if path.is_dir() and path.name.endswith(f"_{description}")
    ]
    if not matches:
        raise FileNotFoundError(
            f"No checkpoint directory ending with _{description} under {checkpoints_root}"
        )
    if len(matches) > 1:
        matches = sorted(matches, key=lambda p: p.stat().st_mtime, reverse=True)
    return matches[0]


def generate_regime_pool(
    checkpoint_dir: Path,
    n_pool: int,
    sample_step: int,
    temperature: float,
    seq_len: int,
    n_assets: int,
) -> np.ndarray:
    args = _load_args_json(checkpoint_dir)
    if not hasattr(args, "compile"):
        args.compile = False
    exp = Exp_Diffusion_Denoised_X(args)
    ckpt = checkpoint_dir / "checkpoint.pth"
    if not ckpt.exists():
        raise FileNotFoundError(f"Missing {ckpt}")
    exp.model.load_state_dict(process_model_dict(str(ckpt)))
    dataset, _ = exp._get_data()
    generated = exp.generate_data(
        size=n_pool,
        sample_step=sample_step,
        dataset=dataset,
        model=exp.model,
        sampler="DDPM",
        n_steps=20,
        ddim_discretize="uniform",
        ddim_eta=0.0,
        method="discrete",
        overlap_ratio=0.25,
        temperature=temperature,
    )
    flat = generated.to_numpy(dtype=float)
    expected = n_pool * seq_len
    if flat.shape != (expected, n_assets):
        raise ValueError(
            f"Generated shape {flat.shape} != ({expected}, {n_assets})"
        )
    return flat.reshape(n_pool, seq_len, n_assets)


def main() -> int:
    args = parse_args()
    checkpoints_root = Path(args.checkpoints_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    for regime in range(int(args.n_regimes)):
        description = args.description_template.format(k=regime)
        try:
            checkpoint_dir = _find_checkpoint_dir(checkpoints_root, description)
        except FileNotFoundError as exc:
            print(f"[SKIP] regime {regime}: {exc}")
            continue
        print(f"[GEN] regime {regime} from {checkpoint_dir.name}")
        windows = generate_regime_pool(
            checkpoint_dir=checkpoint_dir,
            n_pool=int(args.n_pool),
            sample_step=int(args.sample_step),
            temperature=float(args.temperature),
            seq_len=int(args.seq_len),
            n_assets=int(args.n_assets),
        )
        regime_dir = output_dir / f"regime_k{regime}"
        regime_dir.mkdir(parents=True, exist_ok=True)
        np.save(regime_dir / "windows.npy", windows)
        meta = {
            "regime": int(regime),
            "n_pool": int(windows.shape[0]),
            "seq_len": int(windows.shape[1]),
            "n_assets": int(windows.shape[2]),
            "sample_step": int(args.sample_step),
            "temperature": float(args.temperature),
            "method": "discrete",
            "sampler": "DDPM",
            "checkpoint_dir": str(checkpoint_dir),
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "units": "log_returns",
            "date_matched": False,
        }
        (regime_dir / "meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
        print(f"[OK] {regime_dir / 'windows.npy'} shape={windows.shape}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
