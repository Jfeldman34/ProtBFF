#!/bin/bash
#SBATCH -t 0-10:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/dms_ft_%x_%j.out
#SBATCH -e slurm_logs/dms_ft_%x_%j.err
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python -u tuning_v1/dms_finetune.py \
  --ca_base tuning_v1/out/dms_bases/esmc_ca_base.pth \
  --simple_base tuning_v1/out/dms_bases/esmc_simple_base.pth \
  --cache model_benchmarking/score_caches/${CX}_esmc_score_cache.npz \
  --merged data/dms_esmc/${CX}_merged_esmc --embed_dim 1152 \
  --out tuning_v1/out/dms_ft/${CX}_esmc
