#!/usr/bin/env python
"""Rebuild the DMS merged dirs with ESM-C (sequence-only) embeddings.

For each complex, per merged file, get the wildtype and mutant FULL-complex sequences,
embed both with ESM-C, and store the per-residue embedding DIFFERENCE in the merged
format (Xf = wt_emb - mt_emb, Xr = -Xf), keeping the biophysical scores / ddG / keys.

  ace2 : full seq = ACE2 chain A (from 7W9I, constant) + RBD chain E (from bloom_combined.csv
         by index; wildtype_fasta / optimized_fasta).
  7kmg,9lyp : full seq parsed from the surviving mutant / wildtype PDBs (by index).
The wildtype complex is constant per complex, so it is embedded once.
"""
import argparse, glob, os, sys, warnings
from pathlib import Path
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from embedding_pdb_full import parse_pdb_sequence
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

JF = "/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main"


def chain_seq(struct, cid):
    return "".join(seq1(r.resname, undef_code="X") for r in struct[0][cid] if r.id[0] == " ")


@torch.no_grad()
def embed(model, seq, device):
    t = model.encode(ESMProtein(sequence=seq))
    out = model.logits(t, LogitsConfig(return_embeddings=True))
    e = out.embeddings
    if e.dim() == 3:
        e = e[0]
    e = e[1:-1]                       # strip BOS/EOS
    assert e.shape[0] == len(seq), f"{e.shape[0]} vs {len(seq)}"
    return e.float().cpu().numpy()


def ace2_seqs(idx, bc, ace2_A):
    row = bc[bc["index"] == idx]
    if len(row) == 0:
        return None, None
    r = row.iloc[0]
    return ace2_A + r["wildtype_fasta"], ace2_A + r["optimized_fasta"]


def pdb_seqs(idx, opt_dir, wt_seq):
    p = f"{opt_dir}/{idx}_optimized.pdb"
    if not os.path.exists(p):
        return None, None
    return wt_seq, parse_pdb_sequence(p)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--complex", required=True, choices=["ace2", "9lyp", "7kmg"])
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMC.from_pretrained("esmc_600m", device=device).eval()
    os.makedirs(args.out, exist_ok=True)

    if args.complex == "ace2":
        merged = f"{JF}/merged_scores"
        bc = pd.read_csv(f"{JF}/bloom_combined.csv")
        st = PDBParser(QUIET=True).get_structure("x", "data/dms_esmc/7W9I.pdb")
        ace2_A = chain_seq(st, "A")
        wt_seq_const = None            # ace2 wt seq varies only in RBD -> handled per-file
    else:
        merged = f"{JF}/bloom_antibodies/{args.complex}_merged"
        opt_dir = f"{JF}/bloom_antibodies/{args.complex}_cache/optimized"
        wt_dir = f"{JF}/bloom_antibodies/{args.complex}_cache/wildtype"
        wtpdb = sorted(glob.glob(f"{wt_dir}/*.pdb"))[0]
        wt_seq_const = parse_pdb_sequence(wtpdb)

    files = sorted(glob.glob(f"{merged}/merged_*.npz"), key=lambda p: int(Path(p).stem.split("_")[1]))
    wt_cache = {}
    done = skip = 0
    for f in files:
        d = np.load(f, allow_pickle=True)
        idx = int(d["index"][0]); L = d["Xf"].shape[0]
        if args.complex == "ace2":
            wt_seq, mt_seq = ace2_seqs(idx, bc, ace2_A)
        else:
            wt_seq, mt_seq = pdb_seqs(idx, opt_dir, wt_seq_const)
        if mt_seq is None or len(mt_seq) != L or len(wt_seq) != L:
            skip += 1; continue
        if wt_seq not in wt_cache:
            wt_cache[wt_seq] = embed(model, wt_seq, device)
        wt_emb = wt_cache[wt_seq]
        mt_emb = embed(model, mt_seq, device)
        Xf = (wt_emb - mt_emb).astype(np.float32)
        out = {k: d[k] for k in d.files}
        out["Xf"] = Xf; out["Xr"] = -Xf
        np.savez(f"{args.out}/{Path(f).name}", **out)
        done += 1
        if done % 200 == 0:
            print(f"  {done} done, {skip} skipped", flush=True)
    print(f"finished {args.complex}: {done} written, {skip} skipped -> {args.out}", flush=True)


if __name__ == "__main__":
    main()
