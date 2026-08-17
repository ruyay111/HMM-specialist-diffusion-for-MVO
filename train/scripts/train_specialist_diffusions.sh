#!/usr/bin/env bash
# Train one unconditional UniTST_MP specialist per paper regime.
# Run from Synthetic-data-tests/train/ or from this script's directory.
set -euo pipefail
cd "$(dirname "$0")/.."

N_REGIMES="${N_REGIMES:-5}"
WINDOWS_DIR="${WINDOWS_DIR:-./data/regime_windows}"

for k in $(seq 0 $((N_REGIMES - 1))); do
  data_path="${WINDOWS_DIR}/regime_${k}.npy"
  if [[ ! -f "${data_path}" ]]; then
    echo "[SKIP] missing ${data_path}"
    continue
  fi
  n_windows="$(python - << PY
import numpy as np
print(int(np.load("${data_path}").shape[0]))
PY
)"
  if [[ "${n_windows}" -lt 8 ]]; then
    echo "[SKIP] regime ${k} has ${n_windows} windows (< 8)"
    continue
  fi
  echo "[TRAIN] specialist_regime_${k} n_windows=${n_windows}"
  python run.py \
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
    --description "specialist_regime_${k}"
done
