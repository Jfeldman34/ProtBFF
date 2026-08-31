#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-04:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/esm3_embed_%A_%a.out
#SBATCH -e slurm_logs/esm3_embed_%A_%a.err
#SBATCH -J esm3_embed

# usage: sbatch --array=0-3 esm3_embed.sbatch <pdb-dir> <out-dir> [extra args]
PDB_DIR=$1
OUT_DIR=$2
shift 2

PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF

NSHARD=${SLURM_ARRAY_TASK_COUNT:-1}
SHARD=${SLURM_ARRAY_TASK_ID:-0}

$PY data_pipeline/esm3_embed_pdb.py \
    --pdb-dir "$PDB_DIR" \
    --output-dir "$OUT_DIR" \
    --shard "$SHARD" --num-shards "$NSHARD" "$@"
