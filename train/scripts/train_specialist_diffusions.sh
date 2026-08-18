#!/usr/bin/env bash
# Train one unconditional UniTST_MP specialist per regime.
# Run from Synthetic-data-tests/train/ or from this script's directory.
set -euo pipefail
cd "$(dirname "$0")/.."

resolve_python() {
  if [[ -n "${PYTHON:-}" ]]; then
    echo "${PYTHON}"
    return
  fi
  if [[ -n "${CONDA_PREFIX:-}" ]]; then
    if [[ -f "${CONDA_PREFIX}/python.exe" ]]; then
      echo "${CONDA_PREFIX}/python.exe"
      return
    fi
    if [[ -x "${CONDA_PREFIX}/bin/python" ]]; then
      echo "${CONDA_PREFIX}/bin/python"
      return
    fi
  fi
  if command -v python.exe >/dev/null 2>&1; then
    command -v python.exe
    return
  fi
  if command -v python >/dev/null 2>&1; then
    command -v python
    return
  fi
  echo "[ERROR] python not found. Activate the conda env or set PYTHON=/path/to/python.exe" >&2
  exit 1
}

PYTHON="$(resolve_python)"
echo "[INFO] using PYTHON=${PYTHON}"

N_REGIMES="${N_REGIMES:-5}"
WINDOWS_DIR="${WINDOWS_DIR:-./data/regime_windows}"

for k in $(seq 0 $((N_REGIMES - 1))); do
  data_path="${WINDOWS_DIR}/regime_${k}.npy"
  if [[ ! -f "${data_path}" ]]; then
    echo "[SKIP] missing ${data_path}"
    continue
  fi
  n_windows="$("${PYTHON}" -c "import numpy as np; print(int(np.load(r'''${data_path}''').shape[0]))")"
  n_windows="${n_windows//$'\r'/}"
  if [[ "${n_windows}" -lt 8 ]]; then
    echo "[SKIP] regime ${k} has ${n_windows} windows (< 8)"
    continue
  fi
  has_ckpt="$("${PYTHON}" -c "from pathlib import Path; npy=Path(r'''${data_path}'''); ckpts=list(Path('checkpoints').glob('*_specialist_regime_${k}/checkpoint.pth')); print(any(p.exists() and p.stat().st_mtime>=npy.stat().st_mtime for p in ckpts))")"
  has_ckpt="${has_ckpt//$'\r'/}"
  if [[ "${has_ckpt}" == "True" ]]; then
    echo "[SKIP] specialist_regime_${k} checkpoint is newer than ${data_path}"
    continue
  fi
  echo "[TRAIN] specialist_regime_${k} n_windows=${n_windows}"
  "${PYTHON}" run.py \
    --task_name diffusion_denoised_x \
    --model UniTST_MP \
    --data RegimeWindows \
    --data_path "${data_path}" \
    --seq_len 128 \
    --enc_in 10 \
    --batch_size 32 \
    --sample_multiplier 8 \
    --train_epochs 50 \
    --causal_mask \
    --ind_proj \
    --RoPE \
    --channel_embed \
    --learning_rate 0.0027983303288563873 \
    --lr_decay_rounds 10 \
    --loss "1.0-KL2_N+1.0-Corr+1.0-FFT" \
    --use_gpu True \
    --gpu 0 \
    --device cuda \
    --description "specialist_regime_${k}"
done
