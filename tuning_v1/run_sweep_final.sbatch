#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0-08:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -c 8
#SBATCH -o slurm_logs/tune_%j.out
#SBATCH -e slurm_logs/tune_%j.err
#SBATCH -J protbff_tune

PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
CACHE=model_benchmarking/score_caches/skempi_score_cache.npz
FOLDS=data/cross_validation_folds_mva/60_percent
CLUST=data/cross_validation_folds_mva/60_percent/clusters.tsv
OUT=tuning_v1/out

mkdir -p $OUT
echo "=== STAGE 1: SWEEP ==="
$PY tuning_v1/protbff_train.py --cache $CACHE --folds_dir $FOLDS --clusters $CLUST \
    --mode sweep --n_configs 24 --sweep_folds 0,1,4 --out $OUT/sweep.json

echo "=== SELECT BEST (by mean val Pearson) ==="
$PY - <<'PYEOF'
import json
r = json.load(open("tuning_v1/out/sweep.json"))
keys = ['lr','weight_decay','batch_size','num_hidden','ilddt_weight','loss','dropout_scale']
best = {k: r[0][k] for k in keys}
json.dump(best, open("tuning_v1/out/best_config.json","w"), indent=2)
print("BEST:", best, " valP=%.3f"%r[0]['mean_val_pearson'])
PYEOF

echo "=== STAGE 2: FINAL (10 folds, refit + 5-seed ensemble + calibration) ==="
$PY tuning_v1/protbff_train.py --cache $CACHE --folds_dir $FOLDS --clusters $CLUST \
    --mode final --seeds 5 --config_json $OUT/best_config.json --out $OUT/final.json
echo "=== DONE ==="
