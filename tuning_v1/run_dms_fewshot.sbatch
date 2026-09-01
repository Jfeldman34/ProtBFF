#!/bin/bash
#SBATCH -p gpu_test
#SBATCH -t 0-11:00
#SBATCH --mem=24G
#SBATCH --gres=gpu:1
#SBATCH -c 4
#SBATCH -o slurm_logs/dms_fewshot_%j.out
#SBATCH -e slurm_logs/dms_fewshot_%j.err
#SBATCH -J dms_fewshot
set -euo pipefail
cd /n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF
PY=/n/home02/wangdz/.conda/envs/esm3/bin/python
JF=/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main
mkdir -p tuning_v1/out/dms

# complex/encoder -> merged dir (validated by ddG-overlap: ace2=merged_scores)
$PY tuning_v1/dms_fewshot.py --merged_dir $JF/merged_scores                     --tag ace2_prosst  --out tuning_v1/out/dms/ace2_prosst.json  --seeds 3
$PY tuning_v1/dms_fewshot.py --merged_dir $JF/bloom_esm2_embeddings_merged      --tag ace2_esm2    --out tuning_v1/out/dms/ace2_esm2.json    --seeds 3
$PY tuning_v1/dms_fewshot.py --merged_dir $JF/bloom_antibodies/9lyp_merged      --tag 9lyp_prosst  --out tuning_v1/out/dms/9lyp_prosst.json  --seeds 3
$PY tuning_v1/dms_fewshot.py --merged_dir $JF/bloom_antibodies/9lyp_esm2_merged --tag 9lyp_esm2    --out tuning_v1/out/dms/9lyp_esm2.json    --seeds 3
$PY tuning_v1/dms_fewshot.py --merged_dir $JF/bloom_antibodies/7kmg_merged      --tag 7kmg_prosst  --out tuning_v1/out/dms/7kmg_prosst.json  --seeds 3
echo "ALL DMS COMBOS DONE"
