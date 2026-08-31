#!/usr/bin/env python
"""Out-of-fold test predictions for one readout variant (default: dual).

Re-runs the clean protocol (condition B: no target std, ilddt_weight=0,
cluster-aware val, epoch on val Pearson), 5 seeds/fold, and saves the
seed-ensembled test prediction for every entry. The cache is in exact
SKEMPI/master row order (verified), so `idx` == master row == folds CSV row,
which is what ensemble_rde.py joins on.

Output CSV columns: idx, fold, y, pred
"""
import argparse, time
import numpy as np
from scipy.stats import pearsonr, spearmanr

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_cache, load_clusters, load_folds, cluster_aware_val
from protbff_arch import train_once, BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--folds_dir', required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--readout', default='dual')
    ap.add_argument('--embed_dim', type=int, default=768)
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    import torch
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xf, Xr, y, il, ids = load_cache(args.cache)
    cmap = load_clusters(args.clusters); folds = load_folds(args.folds_dir, ids)
    c = {**BASE, 'readout': args.readout, 'embed_dim': args.embed_dim}
    print(f"device={device} N={len(y)} folds={len(folds)} seeds={args.seeds} readout={args.readout}", flush=True)

    rows = []
    frs, fss = [], []
    for fi, (tr_all, te) in enumerate(folds):
        t0 = time.time()
        tr, val = cluster_aware_val(tr_all, ids, cmap, c['val_frac'], c['random_state'] + fi)
        seed_preds = []
        for s in range(args.seeds):
            seed_preds.append(train_once(c, device, Xf, Xr, y, il, tr, val, te,
                                         seed=c['random_state'] + fi + 1000 * s))
        ens = np.array(seed_preds).mean(0); tt = y[te]
        fr, fs = pearsonr(tt, ens)[0], spearmanr(tt, ens)[0]
        frs.append(fr); fss.append(fs)
        for j, idx in enumerate(te):
            rows.append((int(idx), fi, float(y[idx]), float(ens[j])))
        print(f"  fold {fi}: n={len(te)} r={fr:.4f} s={fs:.4f} ({time.time()-t0:.0f}s)", flush=True)

    import csv
    with open(args.out, 'w', newline='') as fh:
        w = csv.writer(fh); w.writerow(['idx', 'fold', 'y', 'pred']); w.writerows(rows)
    print(f"\nwrote {len(rows)} rows -> {args.out}", flush=True)
    print(f"mean-of-folds r={np.mean(frs):.4f}  s={np.mean(fss):.4f}  "
          f"pooled r={pearsonr([r[2] for r in rows],[r[3] for r in rows])[0]:.4f}", flush=True)


if __name__ == '__main__':
    main()
