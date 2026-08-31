#!/bin/bash
#SBATCH -p shakhnovich
#SBATCH -t 0-04:00
#SBATCH --mem=8G
#SBATCH -c 4
#SBATCH -o slurm_logs/esm2_cache_%j.out
#SBATCH -e slurm_logs/esm2_cache_%j.err
#SBATCH -J esm2_cache
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
JF=/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman
SRC=$JF/test_protbff/ProtBFF/data/merged_output
echo "[rebuild ESM2 merged] $(date)"
$PY data_pipeline/rebuild_merged_esm2.py \
  --merged_dir "$SRC" \
  --wt_embedding_dir $JF/esm2_embeddings/wildtype \
  --opt_embedding_dir $JF/esm2_embeddings/mutant \
  --output_dir data/merged_output_esm2
echo "[merge_scores -> cache] $(date)"
$PY data_pipeline/merge_scores.py \
  --merged_dir data/merged_output_esm2 \
  --output_cache model_benchmarking/score_caches/skempi_esm2_score_cache.npz
echo "[verify] $(date)"
$PY - <<'PYEOF'
import numpy as np
d=np.load('model_benchmarking/score_caches/skempi_esm2_score_cache.npz', allow_pickle=True)
print("Xf",d['Xf'].shape,"Xr",d['Xr'].shape,"y",d['y'].shape)
assert d['Xf'].shape[1]==5*1280, f"expected 6400 cols got {d['Xf'].shape[1]}"
print("finite:", bool(np.isfinite(d['Xf']).all()), "rows:", d['Xf'].shape[0])
PYEOF
echo "[done] $(date)"
