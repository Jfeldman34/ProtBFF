#!/usr/bin/env python
"""Swap ProSST embeddings for ESM2 in the merged NPZs, reusing all scores.

Same as rebuild_merged_esm3.py but for ESM2 per-residue embeddings, which are
stored as torch .pt tensors named "<protein_id>.pt" (shape (L, 1280)) under
wildtype/ and mutant/ subdirs. Scores are embedding-independent, so only Xf/Xr
change. Xf = wildtype - mutant (same sign convention as the ProSST/ESM3 pipeline).
"""
import argparse
from pathlib import Path

import numpy as np
import torch


def load_emb(p):
    t = torch.load(p, map_location="cpu")
    return t.numpy() if hasattr(t, "numpy") else np.asarray(t)


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
        wt = args.wt_embedding_dir / f"{pid}.pt"
        opt = args.opt_embedding_dir / f"{pid}.pt"
        if not wt.exists() or not opt.exists():
            missing += 1
            continue
        ewt, eopt = load_emb(wt), load_emb(opt)
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
