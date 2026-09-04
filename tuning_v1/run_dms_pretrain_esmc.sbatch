#!/bin/bash
#SBATCH -t 0-06:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/dms_pretrain_esmc_%j.out
#SBATCH -e slurm_logs/dms_pretrain_esmc_%j.err
#SBATCH -J pretrain_esmc
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python -u tuning_v1/dms_pretrain.py \
  --cache model_benchmarking/score_caches/skempi_esmc_score_cache.npz \
  --merged_dir data/merged_output_esmc --embed_dim 1152 \
  --out_ca tuning_v1/out/dms_bases/esmc_ca_base.pth \
  --out_simple tuning_v1/out/dms_bases/esmc_simple_base.pth
