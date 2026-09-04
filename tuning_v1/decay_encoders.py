#!/usr/bin/env python
"""Consistent leaky->honest decay for every ProtBFF encoder we have:
CD-HIT-60 (original/leaky) and MVA-60 (honest), dual readout, 5 seeds, mean-of-folds +/- SEM."""
import json, sys, os
import numpy as np
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_cache, load_clusters, load_folds
from protbff_arch import run_variant, BASE

ENCODERS = [
    ("esmc",   "model_benchmarking/score_caches/skempi_esmc_score_cache.npz",  1152),
    ("prosst", "model_benchmarking/score_caches/skempi_score_cache.npz",        768),
    ("esm2",   "model_benchmarking/score_caches/skempi_esm2_score_cache.npz",  1280),
    ("esm3",   "model_benchmarking/score_caches/skempi_esm3_score_cache.npz",  1536),
    ("saprot", "model_benchmarking/score_caches/skempi_saprot_score_cache.npz", 1280),
]
# leaky = random complex split, no homology clustering ("no sequence alignment", original);
# honest = MVA-60 (our method)
SPLITS = [("nosplit", "data/cross_validation_folds_mva/100_percent"),
          ("mva60",   "data/cross_validation_folds_mva/60_percent")]


def sem(a):
    a = np.asarray(a); return float(np.std(a) / np.sqrt(len(a)))


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    out = {}
    for enc, cache, ed in ENCODERS:
        if not os.path.exists(cache):
            print(f"skip {enc}: no cache", flush=True); continue
        Xf, Xr, y, il, ids = load_cache(cache)
        out[enc] = {}
        for split, fdir in SPLITS:
            folds = load_folds(fdir, ids)
            ctsv = f"{fdir}/clusters.tsv"
            cmap = load_clusters(ctsv) if os.path.exists(ctsv) else {c: c for c in ids}
            c = {**BASE, 'readout': 'dual', 'embed_dim': ed, 'num_scores': 5}
            r = run_variant(c, folds, Xf, Xr, y, il, ids, cmap, device, seeds=5)
            out[enc][split] = dict(protbff_r=float(r['mean_r']), protbff_s=float(r['mean_s']),
                                   protbff_r_sem=sem(r['fold_r']), protbff_s_sem=sem(r['fold_s']))
            print(f"  {enc} {split}: P={r['mean_r']:.3f}+-{sem(r['fold_r']):.3f}  "
                  f"S={r['mean_s']:.3f}+-{sem(r['fold_s']):.3f}", flush=True)
        json.dump(out, open('tuning_v1/out/decay_encoders.json', 'w'), indent=2)
    print("-> tuning_v1/out/decay_encoders.json")


if __name__ == '__main__':
    main()
