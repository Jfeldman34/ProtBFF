#!/usr/bin/env python
"""ilDDT auxiliary-loss ablation for ProtBFF (dual readout, honest MVA-60).

Compares the clean 'full' model (ilddt_weight=0, no auxiliary head signal) against
the paper's setting (ilddt_weight=0.2, auxiliary ilDDT regression head active). Same
architecture, folds, seeds, and clean protocol (no target standardization) for both,
so the only difference is the auxiliary loss weight. Reports the delta = effect of
adding the ilDDT auxiliary loss.
"""
import argparse, json, os, sys
import numpy as np, torch
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_cache, load_clusters, load_folds
from protbff_arch import run_variant, BASE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--folds_dir', required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--embed_dim', type=int, required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--readout', default='dual')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--ilddt_weight', type=float, default=0.2)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xf, Xr, y, il, ids = load_cache(args.cache)
    cmap = load_clusters(args.clusters); folds = load_folds(args.folds_dir, ids)
    print(f"device={device} N={len(y)} embed_dim={args.embed_dim} readout={args.readout} "
          f"seeds={args.seeds} | ilDDT: mean={il.mean():.4f} sd={il.std():.4f}", flush=True)

    variants = {'full': 0.0, 'full_ilddt': args.ilddt_weight}
    res = {}
    for vname, w in variants.items():
        c = {**BASE, 'readout': args.readout, 'embed_dim': args.embed_dim,
             'num_scores': 5, 'ilddt_weight': w}
        r = run_variant(c, folds, Xf, Xr, y, il, ids, cmap, device, seeds=args.seeds)
        res[vname] = {kk: (v.tolist() if isinstance(v, np.ndarray) else v) for kk, v in r.items()}
        print(f"[{vname:11s} ilw={w}] mean_r={r['mean_r']:.4f}  mean_s={r['mean_s']:.4f}  "
              f"pooled_r={r['pooled_r']:.4f}  pooled_s={r['pooled_s']:.4f}", flush=True)
        json.dump(res, open(args.out, 'w'), indent=2, default=float)

    d = res['full_ilddt']['mean_r'] - res['full']['mean_r']
    ds = res['full_ilddt']['mean_s'] - res['full']['mean_s']
    print(f"\n=== ilDDT auxiliary loss (ilw={args.ilddt_weight}) vs full (ilw=0) ===", flush=True)
    print(f"  mean_r  {res['full']['mean_r']:.4f} -> {res['full_ilddt']['mean_r']:.4f}  (Δ {d:+.4f})", flush=True)
    print(f"  mean_s  {res['full']['mean_s']:.4f} -> {res['full_ilddt']['mean_s']:.4f}  (Δ {ds:+.4f})", flush=True)


if __name__ == '__main__':
    main()
