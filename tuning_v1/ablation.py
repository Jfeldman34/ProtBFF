#!/usr/bin/env python
"""Biophysical-score ablation for ProtBFF (set-to-1 convention).

The cache is (N, 5*embed_dim) with score blocks in merge_scores order
[interface, burial, lDDT, SASA, dihedral]; block k of Xf = maxpool(score_k * diff_fwd),
block k of Xr = maxpool(score_k * diff_rev).

'set-to-1' ablation of score k keeps ALL 5 blocks but replaces block k with the
*unweighted* pooled embedding (score=1 -> maxpool(diff)), isolating that score's
biophysical WEIGHTING while leaving the architecture and capacity unchanged. The
unweighted block is computed once from the merged_output (U_fwd=maxpool(diff_fwd)
=maxpool(Xr_emb); U_rev=maxpool(Xf_emb)), aligned to cache row order.
"""
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_cache, load_clusters, load_folds
from protbff_arch import run_variant, BASE
import torch

SCORE_BLOCKS = ['interface', 'burial', 'lDDT', 'SASA', 'dihedral']  # merge_scores order


def uniform_blocks(merged_dir, n_expected):
    """U_fwd, U_rev = unweighted (score=1) pooled embedding, in numeric merged order."""
    files = sorted(glob.glob(merged_dir + '/merged_*.npz'), key=lambda p: int(Path(p).stem.split('_')[1]))
    Uf, Ur = [], []
    for f in files:
        d = np.load(f, allow_pickle=True)
        if 'Xf' not in d or 'Xr' not in d:
            continue
        Uf.append(d['Xr'].max(axis=0))   # maxpool(diff_fwd)=maxpool(Xr_emb) -> goes into cache Xf
        Ur.append(d['Xf'].max(axis=0))   # maxpool(diff_rev)=maxpool(Xf_emb) -> goes into cache Xr
    Uf, Ur = np.vstack(Uf).astype(np.float32), np.vstack(Ur).astype(np.float32)
    assert Uf.shape[0] == n_expected, f"uniform blocks {Uf.shape[0]} != cache {n_expected} (order/skip mismatch)"
    return Uf, Ur


def set_to_one(Xf, Xr, Uf, Ur, k, D):
    Xf2, Xr2 = Xf.copy(), Xr.copy()
    Xf2[:, k * D:(k + 1) * D] = Uf
    Xr2[:, k * D:(k + 1) * D] = Ur
    return Xf2, Xr2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--merged_dir', required=True, help='merged_output for the encoder (to build unweighted blocks)')
    ap.add_argument('--folds_dir', required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--embed_dim', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--readout', default='dual')
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xf, Xr, y, il, ids = load_cache(args.cache)
    cmap = load_clusters(args.clusters); folds = load_folds(args.folds_dir, ids)
    D = args.embed_dim
    print(f"device={device} N={len(y)} embed_dim={D} readout={args.readout} seeds={args.seeds}", flush=True)
    print("computing unweighted (score=1) pooled blocks from merged_output ...", flush=True)
    Uf, Ur = uniform_blocks(args.merged_dir, len(y))

    variants = {'full': None}
    for name in SCORE_BLOCKS:
        variants[f'set1_{name}'] = SCORE_BLOCKS.index(name)

    res = {}
    for vname, k in variants.items():
        if k is None:
            Xf2, Xr2 = Xf, Xr
        else:
            Xf2, Xr2 = set_to_one(Xf, Xr, Uf, Ur, k, D)
        c = {**BASE, 'readout': args.readout, 'embed_dim': D, 'num_scores': 5}
        r = run_variant(c, folds, Xf2, Xr2, y, il, ids, cmap, device, seeds=args.seeds)
        res[vname] = {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in r.items()}
        print(f"[{vname:13s}] mean_r={r['mean_r']:.4f}  mean_s={r['mean_s']:.4f}  pooled_r={r['pooled_r']:.4f}", flush=True)
        json.dump(res, open(args.out, 'w'), indent=2, default=float)

    full = res['full']['mean_r']
    print("\n=== ablation: score set to 1 (Δ mean_r vs full) ===", flush=True)
    for v in variants:
        if v == 'full': continue
        print(f"  {v:13s} {res[v]['mean_r']:.4f}  (Δ {res[v]['mean_r']-full:+.4f})", flush=True)


if __name__ == '__main__':
    main()
