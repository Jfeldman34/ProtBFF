#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-02:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/dms_diag_%j.out
#SBATCH -e slurm_logs/dms_diag_%j.err
#SBATCH -J dms_diag
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python tuning_v1/dms_diag.py
