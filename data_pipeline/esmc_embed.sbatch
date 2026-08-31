#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-11:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/esmc_%j.out
#SBATCH -e slurm_logs/esmc_%j.err
#SBATCH -J esmc_embed
set -uo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
$PY data_pipeline/esmc_embed_pdb.py --pdb-dir "$1" --output-dir "$2" ${3:+--limit $3}
