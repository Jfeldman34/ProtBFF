#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-06:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/lomo_%j.out
#SBATCH -e slurm_logs/lomo_%j.err
#SBATCH -J lomo_esmc
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python tuning_v1/lomo.py \
  --cache model_benchmarking/score_caches/skempi_esmc_score_cache.npz \
  --embed_dim 1152 --seeds 2 \
  --out tuning_v1/out/lomo_esmc.json
