#!/bin/bash
#SBATCH -p shakhnovich
#SBATCH -t 0-04:00
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH -o slurm_logs/esm3_cache_%j.out
#SBATCH -e slurm_logs/esm3_cache_%j.err
#SBATCH -J esm3_cache
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
SRC=/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/data/merged_output

echo "[rebuild ESM3 merged] $(date)"
$PY data_pipeline/rebuild_merged_esm3.py \
  --merged_dir "$SRC" \
  --wt_embedding_dir data/wildtype_embeddings_esm3 \
  --opt_embedding_dir data/optimized_embeddings_esm3 \
  --output_dir data/merged_output_esm3

echo "[merge_scores -> cache] $(date)"
$PY data_pipeline/merge_scores.py \
  --merged_dir data/merged_output_esm3 \
  --output_cache model_benchmarking/score_caches/skempi_esm3_score_cache.npz

echo "[verify] $(date)"
$PY - <<'PYEOF'
import numpy as np
d = np.load('model_benchmarking/score_caches/skempi_esm3_score_cache.npz', allow_pickle=True)
print("Xf", d['Xf'].shape, "Xr", d['Xr'].shape, "y", d['y'].shape)
assert d['Xf'].shape[1] == 5*1536, f"expected 7680 cols, got {d['Xf'].shape[1]}"
print("finite Xf:", bool(np.isfinite(d['Xf']).all()), "finite Xr:", bool(np.isfinite(d['Xr']).all()))
print("rows:", d['Xf'].shape[0])
PYEOF
echo "[done] $(date)"
