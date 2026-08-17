# Plan: HMM-path specialist diffusion for regime MVO

Replace calendar date-matched synthetic augmentation with regime-conditional synthetic paths built like HMMGAN, but using UniTST_MP diffusion specialists instead of GANs.

## Locked decisions

| Item | Choice |
|------|--------|
| Location | `/Users/mohanyang/Desktop/ruya/Synthetic-data-tests` |
| Architecture | Specialists: one unconditional UniTST_MP per regime |
| Regimes | Paper 5 regimes via `identify_regimes_paper` |
| Generation | Full HMM regime path (not dominant-regime only) |
| Diffusion window | `seq_len = 128` |
| Synth storage | Offline per-regime pools |
| Real MVO train | Historical days labeled with dominant forecast regime |
| Synth MVO train | Paths stitched from specialist pools along the HMM path |
| Date matching | Removed for this strategy |
| Model / task | `UniTST_MP` + `diffusion_denoised_x` |
| Mixing length (v1) | `min(len(real_train), 128)` |

## End-to-end picture

```text
TODAY
  unconditional diffusion → date-indexed panels
  → mix_train_matrix(..., train_index=dates)
  → MVO on real regime-k* days + date-matched synth

HMMGAN
  label regimes → train GAN_k → fit HMM
  → sample regime path r_1..T → x_t = G_{r_t}(noise)

THIS PLAN
  label regimes (paper) → train Diffusion_k on regime-k windows
  → offline pool_k of 128×10 paths
  → at rebalance: sample path r_1..H from paper HMM (causal filtered start)
  → stitch synth path from pool_{r_t}
  → MVO on real regime-k* days + stitched synth (no dates)
```

## Target layout

```text
Synthetic-data-tests/
├── README.md
├── requirements.txt
├── docs/
│   └── PLAN_HMM_PATH_SPECIALIST_DIFFUSION.md   # this file
├── train/
│   ├── run.py
│   ├── src/
│   ├── data/
│   │   ├── benchmark_data_log_ret_10.csv
│   │   ├── benchmark_data.csv
│   │   ├── regime_labels_paper_5.csv            # NEW
│   │   └── regime_windows/                      # NEW
│   │       ├── regime_0.npy
│   │       ├── ...
│   │       ├── regime_4.npy
│   │       └── manifest.json
│   ├── scripts/
│   │   ├── convert_diffusion_generated_to_data5.py
│   │   ├── build_paper_regime_labels.py         # NEW
│   │   ├── build_regime_window_datasets.py      # NEW
│   │   ├── train_specialist_diffusions.sh       # NEW
│   │   └── generate_specialist_pools.py         # NEW
│   ├── checkpoints/…_specialist_regime_k/
│   ├── pools/                                   # NEW offline pools
│   │   ├── regime_k0/windows.npy
│   │   ├── ...
│   │   └── regime_k4/windows.npy
│   └── test_results/
├── evaluation/
│   ├── Notebook_Regime_HMM_MVO.ipynb
│   ├── Notebook_Data_5.ipynb
│   ├── paper_hmm.py
│   ├── regime_hmm_mvo.py                        # extend
│   ├── synth_path_builder.py                    # NEW
│   ├── data_utils.py
│   ├── portfolio_core.py
│   ├── portfolio_eval_final.py
│   ├── stress_test_final.py
│   └── data/benchmark/benchmark_data.csv
└── results/
    └── 450_discrete_DDPM_0.7473/                # legacy calendar synth (baseline)
```

## Phase 0 — Prerequisites

Already present under `Synthetic-data-tests/train/`:

- `run.py`
- `src/` (UniTST_MP, `exp_diffusion_denoised_x`, samplers, data providers)
- `data/benchmark_data_log_ret_10.csv`
- `data/benchmark_data.csv`

Already present under `evaluation/`:

- `paper_hmm.py` (`identify_regimes_paper`, `fit_paper_supervised_hmm`, …)
- `regime_hmm_mvo.py` (causal regime MVO runner)
- portfolio helpers

## Phase 1 — Build paper regime labels

**Script:** `train/scripts/build_paper_regime_labels.py`

**Reuse:** `evaluation/paper_hmm.py` → `identify_regimes_paper`

**Steps:**

1. Load prices from `train/data/benchmark_data.csv` (or `evaluation/data/benchmark/benchmark_data.csv`).
2. Restrict to the notebook asset set and date range used in evaluation (2013-01-01 → 2022-08-31).
3. Compute pct returns and equal-weight market return (same signal as regime MVO).
4. Standardize the market series; run `identify_regimes_paper` with:
   - `n_regimes=5`
   - `volatility_window=20`
   - `min_segment_length=32`
5. Save:
   - `train/data/regime_labels_paper_5.csv` with columns `date,regime`
   - `train/data/regime_vol_centers.npy`
   - optional diagnostics plot / print of counts and mean vol by regime

**Note:** Offline labeling may use the full sample for pool construction. The MVO backtest still refits regimes causally on history only.

## Phase 2 — Build per-regime diffusion windows

**Script:** `train/scripts/build_regime_window_datasets.py`

**Inputs:**

- `train/data/benchmark_data_log_ret_10.csv` (10-asset log returns, diffusion scale)
- `train/data/regime_labels_paper_5.csv`

**Window rule:**

1. Align dates between log-return panel and regime labels.
2. Extract contiguous segments where `regime == k`.
3. Inside each segment of length `L`:
   - if `L >= 128`: extract sliding windows of length 128 (stride 1 or 5)
   - if `L < 128`: skip (keep regime purity for v1)
4. Each window shape: `(128, 10)`.

**Outputs:**

```text
train/data/regime_windows/regime_{k}.npy    # (N_k, 128, 10)
train/data/regime_windows/manifest.json     # counts, stride, date range
```

Log per-regime counts. Expect regimes 0–1 to be plentiful and 3–4 scarce. Do not merge rare regimes by default.

## Phase 3 — Train five specialist UniTST_MP models

**Reuse:**

- `train/run.py`
- `src/exp/exp_diffusion_denoised_x.py`
- `src/models/UniTST_MultiPatch.py`

**Small data adapter (preferred):**

Add `Dataset_RegimeWindows` in `src/data_provider/data_loader.py` and register `--data RegimeWindows` in `data_factory.py`.

- Loads `regime_{k}.npy`
- Yields windows of shape `(128, 10)` with the same scaling interface expected by the diffusion exp

**Wrapper:** `train/scripts/train_specialist_diffusions.sh`

For each `k in 0..4`:

```bash
python run.py \
  --task_name diffusion_denoised_x \
  --model UniTST_MP \
  --data RegimeWindows \
  --data_path ./data/regime_windows/regime_${k}.npy \
  --enc_in 10 \
  --seq_len 128 \
  --sample_multiplier 8 \
  --train_epochs 50 \
  --causal_mask \
  --ind_proj \
  --RoPE \
  --channel_embed \
  --learning_rate 0.0027983303288563873 \
  --lr_decay_rounds 10 \
  --loss "1.0-KL2_N+1.0-Corr+1.0-FFT" \
  --description specialist_regime_${k}
```

**Checkpoints:** under `train/checkpoints/` with setting names containing `specialist_regime_{k}`.

## Phase 4 — Offline generate specialist pools

**Script:** `train/scripts/generate_specialist_pools.py`

**Reuse:**

- Checkpoint loading and `generate_data` path from `Exp_Basic_Diffusion` / `Exp_Diffusion_Denoised_X`
- Sampler settings aligned with the existing notebook synth: discrete DDPM, temperature `0.7473`, step e.g. `450`

**For each regime k:**

1. Load specialist checkpoint `k`.
2. Generate `N_pool` samples with shape `(N_pool, 128, 10)`.
   - Suggested starting value: `N_pool = 200`–`500`.
3. Inverse-scale to the return units expected by evaluation (same convention as current generated panels).
4. Save:

```text
train/pools/regime_k{k}/windows.npy
train/pools/regime_k{k}/meta.json
```

Pools are regime-keyed bags of windows. No calendar index.

## Phase 5 — Causal HMM path sampling at evaluation time

**New module:** `evaluation/synth_path_builder.py`

**Reuse:** `paper_hmm.fit_paper_supervised_hmm`, filtered-state APIs already used by `forecast_occupancy`.

At each MVO rebalance date `t` (history only):

1. Fit paper regimes on standardized history (existing `regime_hmm_mvo` logic).
2. Fit supervised Gaussian HMM on `(history_returns, regime_labels)`.
3. Obtain causal filtered state `alpha` at the last history day.
4. Sample a regime path of length `H = 128`:

```text
r_0 ~ Categorical(alpha)
for h = 1 .. H-1:
    r_h ~ Categorical(A[r_{h-1}, :])
```

This mirrors HMMGAN’s `hmm_model.sample(n_steps)`, but starts from the causal filtered state rather than a generic HMM start.

Also keep:

- forecast occupancy `q_bar`
- dominant regime `k* = argmax q_bar` for real-train selection

## Phase 6 — Stitch synthetic paths from offline pools

**Function:** `build_synth_paths_from_hmm(regime_path, pools, n_paths, seed)` in `synth_path_builder.py`

**Algorithm (block-wise):**

1. Compress `regime_path` into contiguous runs:
   `[(regime=1, len=40), (regime=0, len=55), ...]`.
2. For each synthetic series `j = 1 .. n_synth`:
   - For each run `(k, L)`:
     - Draw a window `W` from `pools[k]` with shape `(128, 10)`.
     - If `L <= 128`: take a random contiguous slice of length `L`.
     - If `L > 128`: concatenate successive draws until length `L`.
   - Concatenate runs → path `X_j` with shape `(H, 10)`.
3. Return array `(n_synth, H, 10)`.

This improves on HMMGAN’s one-scalar-per-step generation while keeping specialist selection by discrete regime ID.

## Phase 7 — Wire into regime MVO (replace date matching)

**Extend:** `evaluation/regime_hmm_mvo.py`

### Real train (unchanged)

```text
train_real = select_regime_train(history, labels, k_star, max_train=504)
```

### Synth train (new)

```text
regime_path = sample_hmm_path(hmm, filtered_alpha, H=128)
synth_paths = build_synth_paths_from_hmm(regime_path, pools, n_synth, seed)
```

### Mixing (new; no `train_index`)

Do **not** call `mix_train_matrix(..., train_index=dates)` for this strategy.

Add `mix_train_with_regime_paths(real_train, synth_paths)`:

1. Let `T = min(len(real_train), 128)`.
2. Take the last `T` rows of `real_train` (or a deterministic subsample).
3. From each synth path, take a length-`T` block (e.g. first `T` or a random contiguous block).
4. Column-stack like the current mixer: `[real | synth_1 | ...]` then reuse `collapse_weights`.

### Experiment flag

```python
synth_mode = "hmm_path_specialist"  # new default for this plan
# legacy baseline remains available as "calendar" if needed
```

## Phase 8 — Notebook updates

**Notebook:** `evaluation/Notebook_Regime_HMM_MVO.ipynb`

1. Load offline pools from `train/pools/regime_k*/windows.npy`.
2. Pass pools + `synth_mode="hmm_path_specialist"` into the experiment runner.
3. Keep `n_synth = 0` as the real-only baseline.
4. Add diagnostics:
   - dominant-regime accuracy and Brier (existing)
   - mean sampled-path occupancy vs forecast occupancy
   - fraction of path days equal to `k*`
   - train sizes / skip rates

`Notebook_Data_5.ipynb` can keep the legacy calendar synth under `results/450_discrete_DDPM_0.7473/` for comparison.

The last section of `Notebook_Regime_HMM_MVO.ipynb` loads specialist pools when they exist and runs `synth_mode="hmm_path_specialist"`.


## Phase 9 — Validation checklist

1. Phase 1 regime counts look sensible; mean vol increases with regime ID.
2. Phase 2 manifest shows usable `N_k` for each specialist (especially k=0,1).
3. Each specialist trains; loss decreases; checkpoints save.
4. Generated regime-4 windows have higher volatility than regime-0 windows.
5. Stitched-path regime histogram matches the sampled HMM path histogram.
6. `n_synth=0` matches the previous hard-regime / regime_mvo real-only baseline.
7. `n_synth>0` runs without requiring overlapping calendar dates.
8. Smoke one rebalance window end-to-end using prebuilt pools.

## Reuse vs new work

| Reuse | New |
|------|-----|
| `run.py`, `diffusion_denoised_x`, `UniTST_MP` | `build_paper_regime_labels.py` |
| `Exp_Basic_Diffusion.generate_data` | `Dataset_RegimeWindows` + factory registration |
| `paper_hmm.identify_regimes_paper` / supervised HMM | `build_regime_window_datasets.py` |
| Ledoit–Wolf MVO, `collapse_weights` | `generate_specialist_pools.py` |
| Notebook structure, seeds, mix grid | `synth_path_builder.py` |
| | `mix_train_with_regime_paths` |
| | `synth_mode` plumbing in `regime_hmm_mvo.py` |

| Do not use for this design | Why |
|------|-----|
| `Benchmark_Cond` / `*_conditional` tasks | Different architecture (single conditional net) |
| `mix_train_matrix(..., train_index=dates)` | Date matching |
| UniTST conditional CFG path | Not needed for specialists |

## Implementation order

1. Write this plan (done).
2. Phase 1: paper regime label CSV.
3. Phase 2: regime window `.npy` datasets + manifest.
4. Phase 3: `Dataset_RegimeWindows` + train specialists.
5. Phase 4: offline pools.
6. Phase 5–6: `synth_path_builder.py` with unit tests on fake pools.
7. Phase 7: hook into `regime_hmm_mvo.py` behind `synth_mode`.
8. Phase 8–9: notebook run and validation vs calendar baseline.

## Open follow-ups (not blocking v1)

- Whether scarce regimes (3/4) need more epochs, lower LR, or data augmentation.
- Whether evaluation should also report a dominant-only synth ablation beside the full HMM path.
- Whether to later add online sampling (call diffusion at each rebalance) in addition to offline pools.
