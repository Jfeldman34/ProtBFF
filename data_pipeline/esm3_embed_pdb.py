#!/usr/bin/env python
"""Per-residue ESM3 embeddings for ProtBFF, structure-conditioned.

Drop-in replacement for Step 1 of the pipeline (tokenize_pdb_full.py +
embedding_pdb_full.py, which use ProSST-2048).  Emits
`{stem}_full_embeddings.npy` of shape (L, 1536) with rows in exactly the order
`embedding_pdb_full.parse_pdb_sequence` enumerates residues, so the arrays stay
aligned with the per-residue score vectors in calculate_all_scores.py.

Multi-chain complexes are embedded as one ESM3 input with '|' chainbreaks
between chains, so the model sees the interface.  The chainbreak rows are
dropped from the output.
"""
import argparse
import warnings
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")

from Bio.PDB import PDBParser  # noqa: E402

from esm.models.esm3 import ESM3  # noqa: E402
from esm.sdk.api import ESMProtein, LogitsConfig  # noqa: E402
from esm.utils import residue_constants as RC  # noqa: E402

import sys  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_pdb_full import STANDARD_AA, reassign_empty_chain_atoms  # noqa: E402

ATOM_ORDER = {name: i for i, name in enumerate(RC.atom_types)}
EMBED_DIM = 1536


def parse_pdb_atom37(pdb_file: Path):
    """Sequence + atom37 coords in the same residue order as parse_pdb_sequence.

    Returns (sequence, coords, break_positions) where `sequence` includes '|'
    at chain transitions, `coords` is (len(sequence), 37, 3) with NaN for
    missing atoms and for the chainbreak rows, and `break_positions` are the
    indices of the '|' rows.
    """
    structure = PDBParser(QUIET=True).get_structure("protein", pdb_file)
    structure = reassign_empty_chain_atoms(structure)

    seq, coords, breaks = [], [], []
    first_chain = True
    for model in structure:
        for chain in model:
            chain_res = []
            for residue in chain:
                if residue.id[0] != " ":
                    continue
                aa = STANDARD_AA.get(residue.get_resname())
                if not aa:
                    continue
                pos = np.full((37, 3), np.nan, dtype=np.float32)
                for atom in residue:
                    idx = ATOM_ORDER.get(atom.get_id())
                    if idx is not None:
                        pos[idx] = atom.get_coord()
                chain_res.append((aa, pos))
            if not chain_res:
                continue
            if not first_chain:
                seq.append("|")
                coords.append(np.full((37, 3), np.nan, dtype=np.float32))
                breaks.append(len(seq) - 1)
            first_chain = False
            for aa, pos in chain_res:
                seq.append(aa)
                coords.append(pos)
    return "".join(seq), np.stack(coords), breaks


@torch.no_grad()
def embed(model, sequence, coords, breaks, device):
    protein = ESMProtein(
        sequence=sequence,
        coordinates=torch.from_numpy(coords).to(device),
    )
    tensor = model.encode(protein)
    out = model.logits(tensor, LogitsConfig(return_embeddings=True))
    emb = out.embeddings
    if emb.dim() == 3:
        emb = emb[0]
    emb = emb[1:-1]  # strip BOS / EOS
    if emb.shape[0] != len(sequence):
        raise ValueError(f"got {emb.shape[0]} rows for a {len(sequence)}-token input")
    if breaks:
        keep = np.setdiff1d(np.arange(len(sequence)), np.asarray(breaks))
        emb = emb[torch.from_numpy(keep).to(emb.device)]
    return emb.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser(description="Generate per-residue ESM3 embeddings.")
    ap.add_argument("--pdb-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="esm3_sm_open_v1")
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

    model = ESM3.from_pretrained(args.model, device=device)
    model.eval()

    done = skipped = failed = 0
    for i, pdb in enumerate(files):
        out = args.output_dir / f"{pdb.stem}_full_embeddings.npy"
        if out.exists() and not args.overwrite:
            skipped += 1
            continue
        try:
            seq, coords, breaks = parse_pdb_atom37(pdb)
            emb = embed(model, seq, coords, breaks, device)
            expected = len(seq) - len(breaks)
            if emb.shape != (expected, EMBED_DIM):
                raise ValueError(f"shape {emb.shape}, expected ({expected}, {EMBED_DIM})")
            np.save(out, emb.astype(np.float32))
            done += 1
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"FAIL {pdb.name}: {type(exc).__name__}: {exc}", flush=True)
        if (i + 1) % 100 == 0:
            print(f"  {i+1}/{len(files)}  done={done} skipped={skipped} failed={failed}", flush=True)
    print(f"finished: done={done} skipped={skipped} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
