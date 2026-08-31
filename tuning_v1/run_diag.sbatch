#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-02:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -o slurm_logs/diag_%j.out
#SBATCH -e slurm_logs/diag_%j.err
#SBATCH -J protbff_diag
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
$PY tuning_v1/protbff_train.py \
  --cache model_benchmarking/score_caches/skempi_score_cache.npz \
  --folds_dir data/cross_validation_folds_mva/60_percent \
  --clusters data/cross_validation_folds_mva/60_percent/clusters.tsv \
  --mode diag --out tuning_v1/out/diag.json
echo "=== DIAG DONE ==="
