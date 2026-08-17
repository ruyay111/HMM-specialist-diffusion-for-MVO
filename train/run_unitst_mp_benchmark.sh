#!/usr/bin/env bash
# Train + generate UniTST_MP Benchmark synthetic data (matches the notebook synth pipeline).
# Run from Synthetic-data-tests/train/
set -euo pipefail
cd "$(dirname "$0")"

python run.py \
  --task_name diffusion_denoised_x \
  --model UniTST_MP \
  --data Benchmark \
  --data_path ./data/benchmark_data_log_ret_10.csv \
  --sample_multiplier 8 \
  --train_epochs 50 \
  --causal_mask \
  --ind_proj \
  --RoPE \
  --channel_embed \
  --learning_rate 0.0027983303288563873 \
  --lr_decay_rounds 10 \
  --enc_in 10 \
  --loss "1.0-KL2_N+1.0-Corr+1.0-FFT" \
  --description unitst_mp_benchmark
