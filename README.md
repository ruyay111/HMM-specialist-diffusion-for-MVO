# Synthetic-data-tests

Self-contained package for:

1. Training / generating diffusion synthetic returns (`UniTST_MP`)
2. Evaluating synthetic data in portfolio notebooks (`Notebook_Data_5`, `Notebook_Regime_HMM_MVO`)

## Layout

```
Synthetic-data-tests/
├── README.md
├── requirements.txt
├── train/                          # diffusion train + generate
│   ├── run.py
│   ├── run_unitst_mp_benchmark.sh  # recommended CLI wrapper
│   ├── src/                        # UniTST_MP, exp, data loaders, samplers
│   ├── data/
│   │   ├── benchmark_data_log_ret_10.csv   # training inputs (10 assets)
│   │   └── benchmark_data.csv
│   ├── scripts/convert_diffusion_generated_to_data5.py
│   ├── checkpoints/…best_guess_optuna/     # pretrained weights for this run
│   └── test_results/…/450_…/               # matching generated CSV from that setting
├── evaluation/                     # notebooks + portfolio / HMM helpers
│   ├── Notebook_Data_5.ipynb
│   ├── Notebook_Regime_HMM_MVO.ipynb
│   ├── data_utils.py
│   ├── portfolio_core.py
│   ├── portfolio_eval_final.py
│   ├── stress_test_final.py
│   ├── regime_hmm_mvo.py
│   ├── supervised_hmm.py                # supervised HMM (no torch GAN deps)
│   └── data/benchmark/benchmark_data.csv
└── results/
    └── 450_discrete_DDPM_0.7473/   # synth used by the notebooks
        ├── generated_data.csv
        └── notebook_data5/         # per-asset 20-sequence CSVs
```

## Setup

```bash
cd /Users/mohanyang/Desktop/ruya/Synthetic-data-tests
pip install -r requirements.txt
```

Use the same conda env you already use for Diffusion (`diff`) if preferred.
A full frozen export of the original train env is in `train/requirements-full-freeze.txt`.

## Train + generate synthetic data

From `train/`:

```bash
cd train
./run_unitst_mp_benchmark.sh
```

Equivalent command:

```bash
cd train
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
```

Notes:

- `--data_path` is required; the default `warehouse/...` path does not exist.
- `--channel_embed` matches the saved Optuna/best-guess setting used by the notebooks.
- Outputs land under `train/test_results/{setting}/…/generated_data.csv`.
- Sampling temperature `0.7473` is set inside `run.py` (not a CLI flag).

To feed a newly generated CSV into the notebooks:

```bash
python train/scripts/convert_diffusion_generated_to_data5.py \
  --generated-csv path/to/generated_data.csv \
  --benchmark-csv evaluation/data/benchmark/benchmark_data.csv \
  --output-dir results/450_discrete_DDPM_0.7473/notebook_data5
```

Then point the notebooks’ `TEST_RUN_DIR` at that results folder (already the default).

## Run evaluation notebooks

Open notebooks from `evaluation/` (kernel working directory should be `evaluation/`):

- `Notebook_Data_5.ipynb` — synthetic-data portfolio stress / MVO experiments
- `Notebook_Regime_HMM_MVO.ipynb` — supervised HMM + regime MVO

They load:

- Real prices: `evaluation/data/benchmark/benchmark_data.csv`
- Synthetic pools: `results/450_discrete_DDPM_0.7473/notebook_data5/`

## Specialist HMM-path diffusion

See [`docs/PLAN_HMM_PATH_SPECIALIST_DIFFUSION.md`](docs/PLAN_HMM_PATH_SPECIALIST_DIFFUSION.md).

The existing calendar UniTST_MP model (`best_guess_optuna`) was trained on
`train/data/benchmark_data_log_ret_10.csv` with **no date filter**:
**2001-01-01 → 2022-08-31**. Evaluation notebooks then reindex generated windows
onto **2013-01-02 → 2022-08-31**.

The specialist pipeline labels and trains on the same calendar sample
(`2001-01-01` → `2022-08-31`). Evaluation notebooks still restrict the
MVO backtest to **2013-01-02 → 2022-08-31**.

```bash
# Phase 1: 5-regime labels (2001-01-01 to 2022-08-31)
python train/scripts/build_regime_labels.py

# Phase 2: contiguous 128-day windows per regime
python train/scripts/build_regime_window_datasets.py

# Phase 3: train one UniTST_MP per regime (GPU, long)
cd train && bash scripts/train_specialist_diffusions.sh

# Phase 4: sample offline pools
python train/scripts/generate_specialist_pools.py --n-pool 256
```

Then in `Notebook_Regime_HMM_MVO.ipynb`, run the HMM-path specialist section
(`synth_mode="hmm_path_specialist"`). `Notebook_Data_5.ipynb` stays on the
legacy calendar synth.
