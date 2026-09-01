#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-11:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/abl_base_esmc_%j.out
#SBATCH -e slurm_logs/abl_base_esmc_%j.err
#SBATCH -J abl_base_esmc
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python tuning_v1/ablation_baselines.py \
  --cache model_benchmarking/score_caches/skempi_esmc_score_cache.npz \
  --merged_dir data/merged_output_esmc \
  --folds_dir data/cross_validation_folds_mva/60_percent \
  --clusters data/cross_validation_folds_mva/60_percent/clusters.tsv \
  --embed_dim 1152 --readout dual --seeds 5 \
  --out tuning_v1/out/ablation_baselines_esmc.json
