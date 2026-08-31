#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-08:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/esm3_arch_%j.out
#SBATCH -e slurm_logs/esm3_arch_%j.err
#SBATCH -J esm3_arch
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
# ESM3 cache: 5 scores x 1536 = 7680 cols -> embed_dim 1536
$PY tuning_v1/protbff_arch.py \
  --cache model_benchmarking/score_caches/skempi_esm3_score_cache.npz \
  --folds_dir data/cross_validation_folds_mva/60_percent \
  --clusters data/cross_validation_folds_mva/60_percent/clusters.tsv \
  --variants antisym,dual --embed_dim 1536 --seeds 5 \
  --out tuning_v1/out/arch_esm3.json
