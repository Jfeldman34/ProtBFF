#!/usr/bin/env python
"""Bare-encoder baseline on MVA-60: no biophysical scores, no ProtBFF attention.

Reads a merged_output dir (per-residue embedding diffs Xf/Xr), max-pools over
residues to a single per-mutation vector (this is the 'scores=1' / no-ProtBFF
encoder representation), and fits ridge on [D|S] per MVA fold with alpha
val-selected on the same cluster-aware val used everywhere else. This is the
honest-split analogue of the paper's encoder 'Baseline' column.

Row order in merged_* is the SKEMPI/master order (verified for ProSST), so ids
match the fold files directly.
"""
import argparse, glob
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_clusters, load_folds, cluster_aware_val


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged_dir', required=True)
    ap.add_argument('--folds_dir', default='data/cross_validation_folds_mva/60_percent')
    ap.add_argument('--clusters', default='data/cross_validation_folds_mva/60_percent/clusters.tsv')
    ap.add_argument('--label', default='baseline')
    args = ap.parse_args()

    files = sorted(glob.glob(args.merged_dir + '/merged_*.npz'),
                   key=lambda p: int(Path(p).stem.split('_')[1]))
    Xf, Xr, y, ids = [], [], [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if 'Xf' not in d or 'Xr' not in d:
            continue
        # same sign convention as merge_scores: diff_fwd = Xr_emb, diff_rev = Xf_emb
        Xf.append(d['Xr'].max(axis=0))   # max-pool over residues -> (D,)
        Xr.append(d['Xf'].max(axis=0))
        y.append(float(d['ddG']))
        pid = str(d['protein_id'])
        ids.append(pid.split('_', 1)[1] if '_' in pid else pid)
    Xf = np.vstack(Xf).astype(np.float64); Xr = np.vstack(Xr).astype(np.float64)
    y = np.array(y); ids = np.array(ids)
    print(f"{args.label}: N={len(y)} D={Xf.shape[1]}", flush=True)

    D = Xf - Xr; S = (Xf + Xr) / 2.0; X = np.hstack([D, S])
    cmap = load_clusters(args.clusters); folds = load_folds(args.folds_dir, ids)
    frs, fss, fas, pt, pp = [], [], [], [], []
    for fi, (tr_all, te) in enumerate(folds):
        tr, val = cluster_aware_val(tr_all, ids, cmap, 0.10, 42 + fi)
        best_a, best_v = 1000.0, -np.inf
        for a in (100, 300, 1000, 3000, 10000):
            m = Ridge(alpha=a).fit(X[tr], y[tr]); vp = m.predict(X[val])
            v = pearsonr(y[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
            if v > best_v: best_v, best_a = v, a
        m = Ridge(alpha=best_a).fit(X[np.concatenate([tr, val])], y[np.concatenate([tr, val])])
        p = m.predict(X[te])
        frs.append(pearsonr(y[te], p)[0]); fss.append(spearmanr(y[te], p)[0])
        lab = (y[te] > 0).astype(int)
        fas.append(roc_auc_score(lab, p) if len(np.unique(lab)) > 1 else np.nan)
        pt += list(y[te]); pp += list(p)
    print(f"{args.label} MVA-60 (ridge [D|S] baseline):")
    print(f"  mean-of-folds  P={np.mean(frs):.4f}  S={np.mean(fss):.4f}  AUROC={np.nanmean(fas):.4f}")
    print(f"  pooled         P={pearsonr(pt,pp)[0]:.4f}  S={spearmanr(pt,pp)[0]:.4f}")


if __name__ == '__main__':
    main()
