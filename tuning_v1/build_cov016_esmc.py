#!/usr/bin/env python
"""LY-CoV016 (etesevimab, PDB 7C01) DMS build for ESM-C -> merged dir (ace2 format).
Per-residue interface/burial/SASA from the WT 7C01 complex via the pipeline's own
scores/*.py (mutation-independent structural scores); lDDT/dihedral = WT (1.0), matching the
near-WT values the real single-point DMS carries. ESM-C embeds RBD-mutant + Fab(H,L) complexes;
stores per-residue Xf=wt-mt / Xr, scores (1,L), index, ddG per mutant. WT-scored approximation
(no per-mutant FoldX structures exist for LY-CoV016) -- exploratory, to gauge the ProtBFF lift."""
import os, sys, warnings
import numpy as np, pandas as pd, torch
warnings.filterwarnings("ignore")
from Bio.PDB import PDBParser
from Bio.SeqUtils import seq1
from Bio import pairwise2
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "data_pipeline"))
from scores.interface_score import interface_scores
from scores.burial_score import burial_scores
from scores.sasa_score import sasa_scores
from esm.models.esmc import ESMC
from esm.sdk.api import ESMProtein, LogitsConfig

BASE = "/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF"
PDB = f"{BASE}/data/dms_cov016/7C01.pdb"
BLOOM = "/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/AF3Complex/df_bloom_processed.csv"
MERGED = f"{BASE}/data/dms_esmc/cov016_merged_esmc"; os.makedirs(MERGED, exist_ok=True)


def chain_residues(model, cid):
    """(aa, resnum) for standard residues in a chain, in PDB order."""
    out = []
    for r in model[cid]:
        if r.id[0] == " ":
            aa = seq1(r.resname, undef_code="X")
            if aa != "X":
                out.append((aa, r.id[1]))
    return out


def per_residue_scores(pdb, chains):
    """pipeline interface/burial/SASA at every residue of `chains` (as pseudo-mutations
    aa+chain+pos+aa; identity-preserving so geometry-only scores are unaffected)."""
    m = PDBParser(QUIET=True).get_structure("x", pdb)[0]
    muts, order = [], []
    for c in chains:
        for aa, pos in chain_residues(m, c):
            muts.append(f"{aa}{c}{pos}{aa}"); order.append((c, pos))
    inter = np.array(interface_scores(pdb, muts), float)
    bur = np.array(burial_scores(pdb, muts), float)
    sasa = np.array(sasa_scores(pdb, muts), float)
    seqs = {c: "".join(aa for aa, _ in chain_residues(m, c)) for c in chains}
    return order, inter, bur, sasa, seqs


def align_map(bloom_rbd, struct_rbd):
    """map each bloom-RBD position -> index into struct_rbd (or -1 if gap)."""
    a = pairwise2.align.globalms(bloom_rbd, struct_rbd, 2, -1, -5, -0.5, one_alignment_only=True)[0]
    mp, bi, si = [], 0, 0
    for cb, cs in zip(a.seqA, a.seqB):
        if cb != "-" and cs != "-":
            mp.append((bi, si))
        if cb != "-": bi += 1
        if cs != "-": si += 1
    m = {b: s for b, s in mp}
    return [m.get(i, -1) for i in range(len(bloom_rbd))]


@torch.no_grad()
def embed(model, seq, device):
    t = model.encode(ESMProtein(sequence=seq))
    e = model.logits(t, LogitsConfig(return_embeddings=True)).embeddings
    if e.dim() == 3:
        e = e[0]
    return e[1:-1].float().cpu().numpy()


def main():
    df = pd.read_csv(BLOOM)
    df = df[df["delta_log_kd_LY-CoV016"].notna()].reset_index(drop=True)
    rbd_muts = df["mutant_sequence"].tolist()
    y = df["delta_log_kd_LY-CoV016"].to_numpy(float)
    rbd_len = len(rbd_muts[0])
    rbd_wt = "".join(pd.Series([s[i] for s in rbd_muts]).mode()[0] for i in range(rbd_len))

    order, inter, bur, sasa, seqs = per_residue_scores(PDB, ["A", "H", "L"])
    struct_rbd = seqs["A"]; H, L = seqs["H"], seqs["L"]; ab = H + L
    na = len(struct_rbd)
    # per-residue score arrays aligned to the EMBEDDING sequence (bloom_rbd + H + L)
    r2s = align_map(rbd_wt, struct_rbd)
    def rbd_arr(a):  # a is chain-A part of the score array (first na entries)
        return np.array([a[j] if j >= 0 else 0.0 for j in r2s])
    ab_slice = slice(na, na + len(ab))
    INTER = np.concatenate([rbd_arr(inter[:na]), inter[ab_slice]])
    BUR = np.concatenate([rbd_arr(bur[:na]), bur[ab_slice]])
    SASA = np.concatenate([rbd_arr(sasa[:na]), sasa[ab_slice]])
    Lc = rbd_len + len(ab)
    LDDT = np.ones(Lc); DIH = np.ones(Lc)                 # WT: no structural deviation
    print(f"N={len(df)} rbd_len={rbd_len} struct_rbd={na} Lc={Lc} "
          f"aligned={sum(j>=0 for j in r2s)}/{rbd_len} interfaceMax={INTER.max():.3f}", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ESMC.from_pretrained("esmc_600m").to(device).eval()
    wt_emb = embed(model, rbd_wt + ab, device)
    keys = np.array([("A", i, " ") for i in range(Lc)], dtype=object)
    for i, rmut in enumerate(rbd_muts):
        me = embed(model, rmut + ab, device)
        n = min(len(wt_emb), len(me), Lc); dfd = (wt_emb[:n] - me[:n]).astype(np.float32)
        sl = slice(0, n)
        np.savez(f"{MERGED}/merged_{i}.npz", keys=keys[:n], index=np.full(n, i, np.int64),
                 interface=INTER[sl][None], burial=BUR[sl][None], lddt=LDDT[sl][None].astype(np.float32),
                 sasa=SASA[sl][None], dihedral=DIH[sl][None], Xf=dfd, Xr=(-dfd), ddG=np.float64(y[i]))
        if i % 400 == 0:
            print(f"  {i}/{len(rbd_muts)}", flush=True)
    print(f"wrote {len(rbd_muts)} merged npz -> {MERGED}", flush=True)


if __name__ == "__main__":
    main()
