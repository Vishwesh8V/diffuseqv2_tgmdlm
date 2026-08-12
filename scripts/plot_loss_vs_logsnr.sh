#!/usr/bin/env bash
set -euo pipefail

# ============ CONFIG ============
CUDA_DEVICE=0
MODEL_PATH="diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/ema_0.9999_200000.pt"
TIME_SCHEDULE_PATH="diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/alpha_cumprod_step_120000.npy"
SPLIT="test"
BATCH_SIZE=64
NUM_T_GRID=50
SOLVER_STEPS=20
SOLVER_ORDER=2
SOLVER_SKIP_TYPE="logSNR"
N_NOISE_SEEDS=3
MIN_EXAMPLES_PER_POSITION=5
MAX_POSITIONS=0
OUT_DIR="loss_logsnr_plots/$(basename "$(dirname "$MODEL_PATH")")"
SEED=102
# =================================

CUDA_VISIBLE_DEVICES=$CUDA_DEVICE python -u scripts/plot_loss_vs_logsnr.py \
    --model_path "$MODEL_PATH" \
    --time_schedule_path "$TIME_SCHEDULE_PATH" \
    --split "$SPLIT" \
    --batch_size "$BATCH_SIZE" \
    --num_t_grid "$NUM_T_GRID" \
    --solver_steps "$SOLVER_STEPS" \
    --solver_order "$SOLVER_ORDER" \
    --solver_skip_type "$SOLVER_SKIP_TYPE" \
    --n_noise_seeds "$N_NOISE_SEEDS" \
    --min_examples_per_position "$MIN_EXAMPLES_PER_POSITION" \
    --max_positions "$MAX_POSITIONS" \
    --out_dir "$OUT_DIR" \
    --seed "$SEED"
