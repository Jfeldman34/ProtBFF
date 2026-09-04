#!/bin/bash
#SBATCH -t 0-10:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/dms_ft_%x_%j.out
#SBATCH -e slurm_logs/dms_ft_%x_%j.err
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
JF=/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main
AF=/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/AF3Complex
[ "$CX" = "ace2" ] && MERGED=$JF/merged_scores || MERGED=$JF/bloom_antibodies/${CX}_merged
/n/home02/wangdz/.conda/envs/esm3/bin/python -u tuning_v1/dms_finetune.py \
  --ca_base $AF/ProSST_Max_loss_all_splits/60_percent/fold_0_full_model.pth \
  --simple_base $AF/prosst_no_scores/fold_0_model.pth \
  --cache model_benchmarking/score_caches/${CX}_score_cache.npz \
  --merged $MERGED --embed_dim 768 \
  --out tuning_v1/out/dms_ft/${CX}_prosst
