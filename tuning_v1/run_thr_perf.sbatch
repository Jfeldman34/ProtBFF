#!/bin/bash
#SBATCH -t 0-08:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/thrperf_%x_%j.out
#SBATCH -e slurm_logs/thrperf_%x_%j.err
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
FOLDS_ROOT="${FOLDS_ROOT:-data/cross_validation_folds_mva}"
THRS="${THRS:-30,40,50,60,80,90}"
TAG="${TAG:-mva}"
/n/home02/wangdz/.conda/envs/esm3/bin/python tuning_v1/threshold_perf_sweep.py \
  --cache "$CACHE" --embed_dim "$ED" \
  --folds_root "$FOLDS_ROOT" \
  --thresholds "$THRS" --seeds 5 \
  --out tuning_v1/out/thr_perf_${TAG}_${ENC}.json
