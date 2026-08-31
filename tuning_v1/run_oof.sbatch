#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-04:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/oof_%j.out
#SBATCH -e slurm_logs/oof_%j.err
#SBATCH -J oof_dual
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
$PY tuning_v1/oof_preds.py \
  --cache model_benchmarking/score_caches/skempi_score_cache.npz \
  --folds_dir data/cross_validation_folds_mva/60_percent \
  --clusters data/cross_validation_folds_mva/60_percent/clusters.tsv \
  --readout dual --embed_dim 768 --seeds 5 \
  --out tuning_v1/out/oof_dual_prosst.csv
