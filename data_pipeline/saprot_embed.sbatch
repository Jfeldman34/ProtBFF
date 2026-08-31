#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-11:00
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/saprot_%j.out
#SBATCH -e slurm_logs/saprot_%j.err
#SBATCH -J saprot_embed
set -uo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
/n/home02/wangdz/.conda/envs/esm3/bin/python data_pipeline/saprot_embed_pdb.py \
  --pdb-dir "$1" --output-dir "$2" ${3:+--limit $3}
