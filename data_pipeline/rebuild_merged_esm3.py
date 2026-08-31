#!/usr/bin/env python
"""Swap ProSST embeddings for ESM3 in the merged NPZs, reusing all scores.

The per-residue biophysical scores are embedding-independent, so there is no
need to re-run calculate_all_scores.py.  This reads each merged_{i}.npz, keeps
every score array untouched, and replaces Xf/Xr with the ESM3 embedding
differences (same sign convention: Xf = wildtype - optimized).
"""
import argparse
from pathlib import Path

import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged_dir", type=Path, required=True)
    ap.add_argument("--wt_embedding_dir", type=Path, required=True)
    ap.add_argument("--opt_embedding_dir", type=Path, required=True)
    ap.add_argument("--output_dir", type=Path, required=True)
    args = ap.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.merged_dir.glob("merged_*.npz"), key=lambda p: int(p.stem.split("_")[1]))
    print(f"{len(files)} merged files", flush=True)

    written = missing = mismatched = 0
    for i, f in enumerate(files):
        d = dict(np.load(f, allow_pickle=True))
        pid = str(d["protein_id"])
        wt = args.wt_embedding_dir / f"{pid}_full_embeddings.npy"
        opt = args.opt_embedding_dir / f"{pid}_full_embeddings.npy"
        if not wt.exists() or not opt.exists():
            missing += 1
            continue
        ewt, eopt = np.load(wt), np.load(opt)
        L = len(d["burial"])
        if ewt.shape[0] != L or eopt.shape[0] != L:
            mismatched += 1
            print(f"LEN {pid}: scores {L}, wt {ewt.shape[0]}, opt {eopt.shape[0]}", flush=True)
            continue
        d["Xf"] = (ewt - eopt).astype(np.float32)
        d["Xr"] = (eopt - ewt).astype(np.float32)
        np.savez_compressed(args.output_dir / f.name, **d)
        written += 1
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)} written={written} missing={missing} mismatched={mismatched}", flush=True)
    print(f"finished: written={written} missing={missing} mismatched={mismatched}", flush=True)


if __name__ == "__main__":
    main()
