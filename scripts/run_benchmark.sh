#!/bin/bash
# run_benchmark.sh
#
# Edit the variables below, then run:
#   bash run_benchmark.sh
#
# Runs the normal-inference baseline + the DPM-Solver step sweep, then
# immediately analyzes the results into a summary table + two plots.

set -e  # stop the script if any command fails, instead of silently continuing

# ---- EDIT THESE ----------------------------------------------------------

# Uncomment and edit if you need to activate a conda/venv environment first
# source activate your_env_name

MODEL_DIR="diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50"          # folder containing exactly ONE checkpoint you want timed
TIME_SCHEDULE_PATH="diffusion_models/diffuseq_iwslt14_mol_h128_lr0.0001_t2000_sqrt_lossaware_seed102_seq128_adan_0707_ckpt109k20260707-12:57:50/alpha_cumprod_step_120000.npy"           # your --time_schedule_path .npy file

NUM_REPEATS=0                                       # real (non-warmup) repeats per configuration
DPM_STEPS="2"                                 # space-separated list of DPM-Solver step counts to sweep
NORMAL_STEPS=2000                                    # full-schedule step count for the normal baseline

BSZ=50
SPLIT="test"
SEED=123

TIMING_CSV="benchmark_timing.csv"                    # shared CSV all runs append into
OUT_PREFIX="benchmark"                               # prefix for _summary.csv / _bar.png / _line.png

SKIP_NORMAL=false                                    # set true if the normal baseline is already logged in TIMING_CSV
SKIP_DPM=false                                       # set true to only (re)run the normal baseline

# ---------------------------------------------------------------------------

EXTRA_FLAGS=""
if [ "$SKIP_NORMAL" = true ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip_normal"
fi
if [ "$SKIP_DPM" = true ]; then
    EXTRA_FLAGS="$EXTRA_FLAGS --skip_dpm"
fi

echo "### Running benchmark sweep..."
python run_benchmark.py \
    --model_dir "$MODEL_DIR" \
    --time_schedule_path "$TIME_SCHEDULE_PATH" \
    --num_repeats "$NUM_REPEATS" \
    --dpm_steps $DPM_STEPS \
    --normal_steps "$NORMAL_STEPS" \
    --bsz "$BSZ" \
    --split "$SPLIT" \
    --seed "$SEED" \
    --timing_csv "$TIMING_CSV" \
    --top_p 0.9 \
    --rejection_rate 0.1 \
    $EXTRA_FLAGS

echo ""
echo "### Analyzing results..."
python analyze_benchmark.py \
    --timing_csv "$TIMING_CSV" \
    --out_prefix "$OUT_PREFIX"

echo ""
echo "### Done. See ${OUT_PREFIX}_summary.csv, ${OUT_PREFIX}_bar.png, ${OUT_PREFIX}_line.png"