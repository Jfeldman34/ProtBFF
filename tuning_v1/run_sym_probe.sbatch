#!/bin/bash
#SBATCH -p shakhnovich
#SBATCH -t 0-03:00
#SBATCH --mem=16G
#SBATCH -c 4
#SBATCH -o slurm_logs/sym_probe_%j.out
#SBATCH -e slurm_logs/sym_probe_%j.err
#SBATCH -J sym_probe
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python tuning_v1/sym_probe.py --cache model_benchmarking/score_caches/skempi_score_cache.npz \
  --folds_dir data/cross_validation_folds_mva/60_percent --clusters data/cross_validation_folds_mva/60_percent/clusters.tsv --tag ProSST --out tuning_v1/out/sym_probe_prosst.json
/n/home02/wangdz/.conda/envs/esm3/bin/python tuning_v1/sym_probe.py --cache model_benchmarking/score_caches/skempi_esmc_score_cache.npz \
  --folds_dir data/cross_validation_folds_mva/60_percent --clusters data/cross_validation_folds_mva/60_percent/clusters.tsv --tag ESM-C --out tuning_v1/out/sym_probe_esmc.json
