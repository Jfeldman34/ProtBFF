#!/usr/bin/env python
"""Figure 2A (original format): model performance vs sequence-identity threshold.

For each encoder and each homology threshold, run 10-fold CV and report mean-of-folds
Pearson/Spearman for ProtBFF (dual readout) and the bare encoder (ridge[D|S]). Tests how
performance changes as the train/test homology constraint is tightened -- the original
Figure 2A demonstration. Works on CD-HIT (cross_validation_folds_final) or MVA folds;
if a threshold dir has no clusters.tsv, a trivial per-complex cmap gives a random val split.
"""
import argparse, json, os, sys, glob
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
import torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_cache, load_clusters, load_folds
from protbff_arch import run_variant, cluster_aware_val, BASE


def sem(a):
    a = np.asarray(a); return float(np.std(a) / np.sqrt(len(a)))


def ridge_bare(folds, Xf, Xr, y, ids, cmap):
    D = Xf - Xr; S = (Xf + Xr) / 2.0; X = np.hstack([D, S]).astype(np.float64)
    fr, fs, pt, pp = [], [], [], []
    for fi, (tr_all, te) in enumerate(folds):
        tr, val = cluster_aware_val(tr_all, ids, cmap, 0.10, 42 + fi)
        best_a, best_v = 1.0, -np.inf
        for a in (100, 1000, 10000):
            m = Ridge(alpha=a).fit(X[tr], y[tr]); vp = m.predict(X[val])
            v = pearsonr(y[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
            if v > best_v: best_v, best_a = v, a
        both = np.concatenate([tr, val]); m = Ridge(alpha=best_a).fit(X[both], y[both]); p = m.predict(X[te])
        fr.append(pearsonr(y[te], p)[0]); fs.append(spearmanr(y[te], p)[0]); pt += list(y[te]); pp += list(p)
    return (float(np.mean(fr)), float(np.mean(fs)), float(pearsonr(pt, pp)[0]), float(spearmanr(pt, pp)[0]),
            sem(fr), sem(fs))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--embed_dim', type=int, required=True)
    ap.add_argument('--folds_root', required=True, help='dir holding <thr>_percent subdirs')
    ap.add_argument('--thresholds', required=True, help='comma list, e.g. 60,80,95,99,100')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=3)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xf, Xr, y, il, ids = load_cache(args.cache)
    print(f"N={len(y)} embed_dim={args.embed_dim} device={device}", flush=True)

    res = {}
    for thr in args.thresholds.split(','):
        fd = f"{args.folds_root}/{thr}_percent"
        folds = load_folds(fd, ids)
        ctsv = f"{fd}/clusters.tsv"
        cmap = load_clusters(ctsv) if os.path.exists(ctsv) else {c: c for c in ids}
        c = {**BASE, 'readout': 'dual', 'embed_dim': args.embed_dim, 'num_scores': 5}
        r = run_variant(c, folds, Xf, Xr, y, il, ids, cmap, device, seeds=args.seeds)
        br, bs, bpr, bps, br_sem, bs_sem = ridge_bare(folds, Xf, Xr, y, ids, cmap)
        res[thr] = dict(protbff_r=float(r['mean_r']), protbff_s=float(r['mean_s']),
                        protbff_pr=float(r['pooled_r']), protbff_ps=float(r['pooled_s']),
                        protbff_r_sem=sem(r['fold_r']), protbff_s_sem=sem(r['fold_s']),
                        bare_r=br, bare_s=bs, bare_pr=bpr, bare_ps=bps,
                        bare_r_sem=br_sem, bare_s_sem=bs_sem)
        print(f"  thr {thr}%: ProtBFF meanP={r['mean_r']:.3f}±{sem(r['fold_r']):.3f} | bare meanP={br:.3f}±{br_sem:.3f}", flush=True)
        json.dump(res, open(args.out, 'w'), indent=2)
    print(f"-> {args.out}")


if __name__ == '__main__':
    main()
