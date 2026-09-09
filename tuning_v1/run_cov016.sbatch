#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-10:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/cov016_%j.out
#SBATCH -e slurm_logs/cov016_%j.err
#SBATCH -J cov016
set -e
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
echo "=== 1. build cov016 merged dir (WT scores + ESM-C embeddings) ==="
$PY tuning_v1/build_cov016_esmc.py
echo "=== 2. build scaled ProtBFF cache ==="
$PY tuning_v1/build_dms_scaled_cache.py --merged_dir data/dms_esmc/cov016_merged_esmc \
  --out data/dms_cov016/cov016_esmc_score_cache.npz
echo "=== 3. fine-tune ProtBFF vs bare (ESM-C SKEMPI bases) ==="
$PY tuning_v1/dms_finetune.py \
  --ca_base tuning_v1/out/dms_bases/esmc_ca_base.pth \
  --simple_base tuning_v1/out/dms_bases/esmc_simple_base.pth \
  --cache data/dms_cov016/cov016_esmc_score_cache.npz \
  --merged data/dms_esmc/cov016_merged_esmc \
  --out tuning_v1/out/dms_ft/cov016_esmc --embed_dim 1152
echo "=== DONE ==="
