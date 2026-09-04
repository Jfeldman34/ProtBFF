#!/usr/bin/env python
"""Build a DMS 'score cache' (scaled 5-block features) for the cross-attention model,
from a DMS merged dir. IDs use str(data['index']) to match load_simple_data in
dms_finetune.py (which reads the same merged dir). Output: Xf,Xr (N,5*embed_dim), y, ids.
"""
import argparse, glob, os
from pathlib import Path
import numpy as np

SCORES = ['interface', 'burial', 'lDDT', 'SASA', 'dihedral']


def maxpool(score, diff):
    return np.max(diff * score[:, None], axis=0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged_dir', required=True)
    ap.add_argument('--out', required=True)
    args = ap.parse_args()
    files = sorted(glob.glob(args.merged_dir + '/merged_*.npz'),
                   key=lambda p: int(Path(p).stem.split('_')[1]))
    Xf, Xr, y, ids = [], [], [], []
    need = {'Xf', 'Xr', 'interface', 'burial', 'lddt', 'sasa', 'dihedral', 'ddG', 'index'}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if not need.issubset(set(d.files)):
            continue
        ef, er = d['Xf'].astype(np.float64), d['Xr'].astype(np.float64)   # per-residue emb difference
        L = ef.shape[0]
        sc = {'interface': np.ravel(d['interface']).astype(float), 'burial': np.ravel(d['burial']).astype(float),
              'lDDT': 1.0 - np.ravel(d['lddt']).astype(float), 'SASA': np.ravel(d['sasa']).astype(float),
              'dihedral': np.ravel(d['dihedral']).astype(float)}
        if any(len(sc[k]) != L for k in sc):
            continue
        diff_fwd, diff_rev = er, ef
        Xf.append(np.concatenate([maxpool(sc[k], diff_fwd) for k in SCORES]))
        Xr.append(np.concatenate([maxpool(sc[k], diff_rev) for k in SCORES]))
        y.append(float(d['ddG']))
        ids.append(str(d['index']))                                      # matches load_simple_data
    Xf = np.array(Xf, np.float32); Xr = np.array(Xr, np.float32)
    y = np.array(y, np.float32); ids = np.array(ids)
    np.savez(args.out, Xf=Xf, Xr=Xr, y=y, ids=ids)
    print(f"{args.merged_dir} -> {args.out}: Xf {Xf.shape}", flush=True)


if __name__ == '__main__':
    main()
