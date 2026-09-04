#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-08:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/dms_fewshot_esmc_%j.out
#SBATCH -e slurm_logs/dms_fewshot_esmc_%j.err
#SBATCH -J dms_esmc_fs
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
mkdir -p tuning_v1/out/dms_esmc
for cx in ace2 9lyp 7kmg; do
  $PY tuning_v1/dms_fewshot.py --merged_dir data/dms_esmc/${cx}_merged_esmc \
    --tag ${cx}_esmc --out tuning_v1/out/dms_esmc/${cx}_esmc.json --seeds 3
done
echo "ALL ESM-C DMS FEWSHOT DONE"
