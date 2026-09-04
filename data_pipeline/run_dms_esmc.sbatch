#!/bin/bash
#SBATCH -t 0-04:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/dms_esmc_%x_%j.out
#SBATCH -e slurm_logs/dms_esmc_%x_%j.err
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python data_pipeline/dms_esmc_rebuild.py \
  --complex "$CX" --out "data/dms_esmc/${CX}_merged_esmc"
