#!/bin/bash
set -e
source /opt/miniforge3/etc/profile.d/conda.sh
conda activate env1

SCRIPTS=/root/25S151115/project3/scripts
OUT=$SCRIPTS/outputs/automated_experiments
mkdir -p "$OUT"
TS=$(date +%Y%m%d_%H%M%S)

echo "===== BATCH 1: abn512 (20 tasks) =====" | tee "$OUT/sweep_all_${TS}.log"
python3 "$SCRIPTS/run_modelb_lr_sched_clip_sweep.py" \
  --task_filter best2_lr \
  --adapter_bottlenecks 512 \
  --gpu_ids 0 4 6 \
  --per_gpu_concurrency 4 \
  2>&1 | tee -a "$OUT/modelb_abn512_${TS}.log" "$OUT/sweep_all_${TS}.log"

echo "===== BATCH 2: sigma sweep on abn256 (60 tasks, sigma=0.01 SKIP_DONE) =====" | tee -a "$OUT/sweep_all_${TS}.log"
python3 "$SCRIPTS/run_modelb_lr_sched_clip_sweep.py" \
  --task_filter best2_lr \
  --adapter_bottlenecks 256 \
  --sigmas 0.001 0.01 0.1 \
  --gpu_ids 0 4 6 \
  --per_gpu_concurrency 4 \
  2>&1 | tee -a "$OUT/modelb_sigma_abn256_${TS}.log" "$OUT/sweep_all_${TS}.log"

echo "===== BATCH 3: lr_endpt on abn256 (60 tasks, 1e-5 SKIP_DONE) =====" | tee -a "$OUT/sweep_all_${TS}.log"
python3 "$SCRIPTS/run_modelb_lr_sched_clip_sweep.py" \
  --task_filter lr_endpt \
  --adapter_bottlenecks 256 \
  --gpu_ids 0 4 6 \
  --per_gpu_concurrency 4 \
  2>&1 | tee -a "$OUT/modelb_lr_endpt_abn256_${TS}.log" "$OUT/sweep_all_${TS}.log"

echo "===== ALL DONE =====" | tee -a "$OUT/sweep_all_${TS}.log"
