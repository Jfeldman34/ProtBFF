#!/usr/bin/env python
"""Per-residue SaProt embeddings for ProtBFF (structure-aware, Foldseek 3Di).

SaProt tokens are AA+3Di pairs. We build the SA sequence in EXACTLY the
`embedding_pdb_full.parse_pdb_sequence` residue order (chains in PDB order), so
the (L, D) output stays aligned with the per-residue biophysical scores. Foldseek
`structureto3didescriptor` gives per-chain (AA, 3Di); we concatenate in PDB chain
order and assert the AA string matches parse_pdb_sequence before embedding.
"""
import argparse, subprocess, tempfile, warnings, os
from pathlib import Path

import numpy as np
import torch

warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser  # noqa: E402
from transformers import AutoTokenizer, AutoModel  # noqa: E402

import sys  # noqa: E402
sys.path.insert(0, str(Path(__file__).resolve().parent))
from embedding_pdb_full import parse_pdb_sequence, reassign_empty_chain_atoms, STANDARD_AA  # noqa: E402

FOLDSEEK = "/n/netscratch/shakhnovich_lab/Lab/jwang/tools/foldseek/bin/foldseek"


def pdb_chain_order(pdb_file):
    s = reassign_empty_chain_atoms(PDBParser(QUIET=True).get_structure("p", pdb_file))
    order = []
    for model in s:
        for chain in model:
            if any(r.id[0] == " " and STANDARD_AA.get(r.get_resname()) for r in chain):
                order.append(chain.id)
        break
    return order


def foldseek_3di(pdb_file):
    """-> {chain_id: (aa_seq, 3di_seq)}"""
    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "out")
        subprocess.run([FOLDSEEK, "structureto3didescriptor", str(pdb_file), out,
                        "--threads", "1"], check=True, capture_output=True)
        res = {}
        for line in open(out):
            f = line.rstrip("\n").split("\t")
            if len(f) < 3:
                continue
            chain = f[0].rsplit("_", 1)[-1]
            res[chain] = (f[1], f[2])
    return res


def sa_sequence(pdb_file):
    ref_aa = parse_pdb_sequence(pdb_file)
    order = pdb_chain_order(pdb_file)
    fs = foldseek_3di(pdb_file)
    aa_parts, di_parts = [], []
    for c in order:
        if c not in fs:
            return None, ref_aa
        aa_parts.append(fs[c][0]); di_parts.append(fs[c][1])
    aa = "".join(aa_parts); di = "".join(di_parts)
    if aa != ref_aa:
        return None, ref_aa  # alignment mismatch -> skip (merge would drop anyway)
    return "".join(a + d.lower() for a, d in zip(aa, di)), ref_aa


@torch.no_grad()
def embed(model, tokenizer, sa_seq, device):
    inp = tokenizer(sa_seq, return_tensors="pt").to(device)
    out = model(**inp)
    emb = out.last_hidden_state[0, 1:-1]  # strip CLS/EOS
    return emb.float().cpu().numpy()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--pdb-dir", type=Path, required=True)
    ap.add_argument("--output-dir", type=Path, required=True)
    ap.add_argument("--model", default="westlake-repl/SaProt_650M_AF2")
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
    print(f"device={device} shard {args.shard}/{args.num_shards} {len(files)} structures", flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModel.from_pretrained(args.model).to(device).eval()

    done = skipped = failed = mism = 0
    for i, pdb in enumerate(files):
        out = args.output_dir / f"{pdb.stem}_full_embeddings.npy"
        if out.exists() and not args.overwrite:
            skipped += 1; continue
        try:
            sa, ref = sa_sequence(pdb)
            if sa is None:
                mism += 1; print(f"MISMATCH {pdb.stem}", flush=True); continue
            emb = embed(model, tok, sa, device)
            if emb.shape[0] != len(ref) or not np.isfinite(emb).all():
                raise ValueError(f"emb {emb.shape} vs L={len(ref)}")
            np.save(out, emb.astype(np.float32)); done += 1
        except Exception as e:
            failed += 1; print(f"FAIL {pdb.stem}: {e}", flush=True)
        if (i + 1) % 200 == 0:
            print(f"  {i+1}/{len(files)} done={done} skip={skipped} mism={mism} fail={failed}", flush=True)
    print(f"finished: done={done} skipped={skipped} mismatch={mism} failed={failed}", flush=True)


if __name__ == "__main__":
    main()
