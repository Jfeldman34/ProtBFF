#!/usr/bin/env python
"""Per-residue ESM-C embeddings for ProtBFF (sequence-only, like ESM2).

Writes `{stem}_full_embeddings.npy` of shape (L, D) with rows in exactly the
order `embedding_pdb_full.parse_pdb_sequence` enumerates residues, so the arrays
stay aligned with the per-residue biophysical scores. Multi-chain complexes are
embedded as one concatenated sequence (ESM-C is sequence-only, no chainbreak) —
L still equals the total residue count, matching the scores.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
from esm.models.esmc import ESMC  # noqa: E402
from esm.sdk.api import ESMProtein, LogitsConfig  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_pdb_full import parse_pdb_sequence  # noqa: E402


def embed(model, sequence, device):
    protein = ESMProtein(sequence=sequence)
    tensor = model.encode(protein)
    out = model.logits(tensor, LogitsConfig(return_embeddings=True))
    emb = out.embeddings
    if emb.dim() == 3:
        emb = emb[0]
    emb = emb[1:-1]  # strip BOS / EOS
    if emb.shape[0] != len(sequence):
        raise ValueError(f"got {emb.shape[0]} rows for a {len(sequence)}-residue input")
    return emb.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Generate per-residue ESM-C embeddings.")
    ap.add_argument("--pdb-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="esmc_600m")
    ap.add_argument("--device", default=None)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--num-shards", type=int, default=1)
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    device = torch.device(args.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    args.output_dir.mkdir(parents=True, exist_ok=True)
    files = sorted(args.pdb_dir.glob("*.pdb"))[args.shard :: args.num_shards]
    if args.limit:
        files = files[: args.limit]
    print(f"device={device}  shard {args.shard}/{args.num_shards}  {len(files)} structures", flush=True)

    model = ESMC.from_pretrained(args.model, device=device)
    model.eval()

    done = skipped = failed = 0
    for i, pdb in enumerate(files):
        out = args.output_dir / f"{pdb.stem}_full_embeddings.npy"
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            seq = parse_pdb_sequence(pdb)
            with torch.no_grad():
                emb = embed(model, seq, device)
            if emb.shape[0] != len(seq) or not np.isfinite(emb).all():
                raise ValueError(f"bad emb {emb.shape} for L={len(seq)}")
            np.save(out, emb.astype(np.float32))
            done += 1
        except Exception as e:
            failed += 1
            print(f"FAIL {pdb.stem}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} done={done} skipped={skipped} failed={failed}", flush=True)
    print(f"finished: done={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
