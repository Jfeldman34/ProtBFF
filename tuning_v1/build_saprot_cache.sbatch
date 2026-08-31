#!/bin/bash
#SBATCH -p shakhnovich
#SBATCH -t 0-04:00
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH -o slurm_logs/saprot_cache_%j.out
#SBATCH -e slurm_logs/saprot_cache_%j.err
#SBATCH -J saprot_cache
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
SRC=/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/merged_output
$PY data_pipeline/rebuild_merged_esm3.py --merged_dir "$SRC" \
  --wt_embedding_dir data/wildtype_embeddings_saprot \
  --opt_embedding_dir data/optimized_embeddings_saprot \
  --output_dir data/merged_output_saprot
$PY data_pipeline/merge_scores.py --merged_dir data/merged_output_saprot \
  --output_cache model_benchmarking/score_caches/skempi_saprot_score_cache.npz
$PY - <<'PYEOF'
import numpy as np
d=np.load('model_benchmarking/score_caches/skempi_saprot_score_cache.npz',allow_pickle=True)
print("Xf",d['Xf'].shape,"cols expect 6400:",d['Xf'].shape[1]==5*1280,"rows",d['Xf'].shape[0])
PYEOF
