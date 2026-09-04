#!/bin/bash
#SBATCH -t 0-06:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/decay_enc_%j.out
#SBATCH -e slurm_logs/decay_enc_%j.err
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python tuning_v1/decay_encoders.py
