#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-11:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/loss_prosst_%j.out
#SBATCH -e slurm_logs/loss_prosst_%j.err
#SBATCH -J loss_prosst
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python tuning_v1/loss_experiments.py \
  --cache model_benchmarking/score_caches/skempi_score_cache.npz \
  --folds_dir data/cross_validation_folds_mva/60_percent --clusters data/cross_validation_folds_mva/60_percent/clusters.tsv --embed_dim 768 --seeds 5 \
  --out tuning_v1/out/loss_prosst.json
