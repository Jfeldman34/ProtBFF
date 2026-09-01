#!/usr/bin/env python
"""Neutralized ablation rows for Table 1: all-scores=1 and embeddings=1 (ESM-C).

Both are built from the merged_output and ALIGNED TO THE CACHE BY protein_id (the ESM-C
cache is lexicographically ordered, NOT numeric merged order -> positional joins are wrong).

  all-scores=1 : biophysical scaling off = the encoder embedding through the SAME model.
                 Correct build is num_scores=1 on the unscaled pooled embedding U=maxpool(diff)
                 (NOT 5 identical blocks, which jams the cross-attention -> the old 0.031).
  embeddings=1 : keep scores, drop embeddings -> ridge on the 5 aggregate score scalars
                 (max over residues of interface/burial/(1-lddt)/SASA/dihedral) per mutation.
Runs `full` too (num_scores=5, scaled cache) as a sanity check (must reproduce 0.469).
"""
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np, torch
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_cache, load_clusters, load_folds
from protbff_arch import run_variant, cluster_aware_val, BASE

SCORE_BLOCKS = ['interface', 'burial', 'lDDT', 'SASA', 'dihedral']


def build_blocks(merged_dir):
    """-> dict {protein_id: (U_fwd(D,), U_rev(D,), Sagg(5,))}."""
    out = {}
    for f in glob.glob(merged_dir + '/merged_*.npz'):
        d = np.load(f, allow_pickle=True)
        if 'Xf' not in d or 'Xr' not in d or 'protein_id' not in d:
            continue
        pid = str(d['protein_id'])
        Uf = d['Xr'].max(axis=0).astype(np.float32)   # maxpool(diff_fwd)=maxpool(Xr_emb)
        Ur = d['Xf'].max(axis=0).astype(np.float32)
        sc = {'interface': d['interface'], 'burial': d['burial'], 'lDDT': 1.0 - d['lddt'],
              'SASA': d['sasa'], 'dihedral': d['dihedral']}
        Sagg = np.array([float(np.max(sc[s])) for s in SCORE_BLOCKS], np.float32)
        out[pid] = (Uf, Ur, Sagg)
    return out


def ridge_folds(X, y, folds, ids, cmap):
    fr, fs, pt, pp = [], [], [], []
    for fi, (tr_all, te) in enumerate(folds):
        tr, val = cluster_aware_val(tr_all, ids, cmap, 0.10, 42 + fi)
        best_a, best_v = 1.0, -np.inf
        for a in (0.01, 0.1, 1, 10, 100):
            m = Ridge(alpha=a).fit(X[tr], y[tr]); vp = m.predict(X[val])
            v = pearsonr(y[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
            if v > best_v: best_v, best_a = v, a
        both = np.concatenate([tr, val])
        m = Ridge(alpha=best_a).fit(X[both], y[both]); p = m.predict(X[te])
        fr.append(pearsonr(y[te], p)[0]); fs.append(spearmanr(y[te], p)[0]); pt += list(y[te]); pp += list(p)
    return dict(mean_r=float(np.mean(fr)), sd_r=float(np.std(fr)), mean_s=float(np.mean(fs)),
                sd_s=float(np.std(fs)), pooled_r=float(pearsonr(pt, pp)[0]), pooled_s=float(spearmanr(pt, pp)[0]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--merged_dir', required=True)
    ap.add_argument('--folds_dir', required=True); ap.add_argument('--clusters', required=True)
    ap.add_argument('--embed_dim', type=int, required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--readout', default='dual'); ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xf, Xr, y, il, ids = load_cache(args.cache)
    raw_ids = [str(s) for s in np.load(args.cache, allow_pickle=True)['ids']]
    cmap = load_clusters(args.clusters); folds = load_folds(args.folds_dir, ids)
    D = args.embed_dim
    blk = build_blocks(args.merged_dir)
    miss = [r for r in raw_ids if r not in blk]
    assert not miss, f"{len(miss)} cache ids missing from merged (e.g. {miss[:3]})"
    Uf = np.nan_to_num(np.vstack([blk[r][0] for r in raw_ids]))
    Ur = np.nan_to_num(np.vstack([blk[r][1] for r in raw_ids]))
    Sagg = np.nan_to_num(np.vstack([blk[r][2] for r in raw_ids]))
    print(f"device={device} N={len(y)} D={D} | blocks aligned by protein_id", flush=True)

    res = {}
    c5 = {**BASE, 'readout': args.readout, 'embed_dim': D, 'num_scores': 5}
    res['full'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                   for k, v in run_variant(c5, folds, Xf, Xr, y, il, ids, cmap, device, args.seeds).items()}
    print(f"[full        ] mean_r={res['full']['mean_r']:.4f} (sanity ~0.469)", flush=True)

    c1 = {**BASE, 'readout': args.readout, 'embed_dim': D, 'num_scores': 1}
    res['scores1_all'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v)
                          for k, v in run_variant(c1, folds, Uf, Ur, y, il, ids, cmap, device, args.seeds).items()}
    print(f"[scores1_all ] mean_r={res['scores1_all']['mean_r']:.4f} (unscaled thru model)", flush=True)

    res['emb1'] = ridge_folds(Sagg.astype(np.float64), y.astype(np.float64), folds, ids, cmap)
    print(f"[emb1        ] mean_r={res['emb1']['mean_r']:.4f} (scores-only ridge)", flush=True)

    json.dump(res, open(args.out, 'w'), indent=2, default=float)
    print(f"-> {args.out}")


if __name__ == '__main__':
    main()
