#!/usr/bin/env python
"""Model-free evidence for the symmetric-aware readout (SI).

Splits the paired ProtBFF features into the antisymmetric part D = Xf - Xr (the only
thing the paper's readout can express) and the symmetric part S = (Xf+Xr)/2 (which it
discards), and fits a plain ridge regression on each, on the honest MVA-60 folds with
alpha selected per fold on a cluster-aware validation split (refit on train+val). No
neural network: if S beats D here, the antisymmetric readout is provably discarding the
larger half of the signal. Also probes the ilDDT target (symmetric quantity) to show it
is only learnable from S.
"""
import argparse, json, os, sys
import numpy as np
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_cache, load_clusters, load_folds
from protbff_arch import cluster_aware_val


def valselect(X, t, folds, ids, cmap, alphas=(30, 100, 300, 1000, 3000, 10000)):
    fr, fs, pt, pp = [], [], [], []
    for fi, (tr_all, te) in enumerate(folds):
        tr, val = cluster_aware_val(tr_all, ids, cmap, 0.10, 42 + fi)
        best_a, best_v = alphas[0], -np.inf
        for a in alphas:
            m = Ridge(alpha=a).fit(X[tr], t[tr]); vp = m.predict(X[val])
            v = pearsonr(t[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
            if v > best_v: best_v, best_a = v, a
        both = np.concatenate([tr, val])
        m = Ridge(alpha=best_a).fit(X[both], t[both]); p = m.predict(X[te])
        fr.append(pearsonr(t[te], p)[0]); fs.append(spearmanr(t[te], p)[0])
        pt += list(t[te]); pp += list(p)
    return dict(mean_r=float(np.mean(fr)), sd_r=float(np.std(fr)),
                mean_s=float(np.mean(fs)), sd_s=float(np.std(fs)),
                pooled_r=float(pearsonr(pt, pp)[0]), pooled_s=float(spearmanr(pt, pp)[0]),
                fold_r=[float(x) for x in fr])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--folds_dir', required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--tag', default='ProSST')
    args = ap.parse_args()

    Xf, Xr, y, il, ids = load_cache(args.cache)
    Xf = Xf.astype(np.float64); Xr = Xr.astype(np.float64)
    y = y.astype(np.float64); il = il.astype(np.float64)
    cmap = load_clusters(args.clusters); folds = load_folds(args.folds_dir, ids)
    D = Xf - Xr; S = (Xf + Xr) / 2.0; DS = np.hstack([D, S])
    print(f"[{args.tag}] N={len(y)} dim(D)={D.shape[1]}", flush=True)

    res = {'tag': args.tag}
    for tgt_name, t in [('ddG', y), ('ilDDT', il)]:
        for feat_name, X in [('D_antisym', D), ('S_sym', S), ('DS_both', DS)]:
            r = valselect(X, t, folds, ids, cmap)
            res[f'{tgt_name}__{feat_name}'] = r
            print(f"  {tgt_name:5s} <- {feat_name:9s}  meanP={r['mean_r']:.3f}±{r['sd_r']:.3f}  "
                  f"meanS={r['mean_s']:.3f}  pooledP={r['pooled_r']:.3f}", flush=True)
        json.dump(res, open(args.out, 'w'), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == '__main__':
    main()
