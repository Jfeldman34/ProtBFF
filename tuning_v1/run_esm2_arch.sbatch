#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-08:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/esm2_arch_%j.out
#SBATCH -e slurm_logs/esm2_arch_%j.err
#SBATCH -J esm2_arch
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
$PY tuning_v1/protbff_arch.py \
  --cache model_benchmarking/score_caches/skempi_esm2_score_cache.npz \
  --folds_dir data/cross_validation_folds_mva/60_percent \
  --clusters data/cross_validation_folds_mva/60_percent/clusters.tsv \
  --variants antisym,dual --embed_dim 1280 --seeds 5 \
  --out tuning_v1/out/arch_esm2.json
