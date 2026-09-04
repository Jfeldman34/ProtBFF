#!/usr/bin/env python
"""Pretrain the DMS base models on SKEMPI (for a given encoder), matching the original
Jonathan bases: a cross-attention (ProtBFF) base on the scaled cache, and a Simple
(no-scores) base on the max-pooled unscaled embedding. Saves checkpoints in the format
dms_finetune.py loads: CA -> {'model_state_dict': ...}, Simple -> raw state_dict.
Antisymmetric readout (f-r)/2, matching the original figure protocol.
"""
import argparse, os, sys, glob
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from dms_finetune import FullModelCrossAttention, DDGPredictorSimple, DdgDataset


def unscaled_from_merged(merged_dir, cache_ids):
    """U_fwd, U_rev (maxpool of unscaled embedding diff) aligned to cache_ids by protein_id."""
    blk = {}
    for f in glob.glob(merged_dir + '/merged_*.npz'):
        d = np.load(f, allow_pickle=True)
        if 'Xf' not in d or 'protein_id' not in d:
            continue
        blk[str(d['protein_id'])] = (d['Xr'].max(0).astype(np.float32), d['Xf'].max(0).astype(np.float32))
    keep = [i for i, pid in enumerate(cache_ids) if pid in blk]
    Uf = np.vstack([blk[cache_ids[i]][0] for i in keep])
    Ur = np.vstack([blk[cache_ids[i]][1] for i in keep])
    return Uf, Ur, np.array(keep)


def train(model, Xf, Xr, y, il, device, multitask, epochs=120, lr=1e-4, bs=64, patience=20):
    trn, val = train_test_split(np.arange(len(y)), test_size=0.10, random_state=42)
    dl = DataLoader(DdgDataset(Xf[trn], Xr[trn], y[trn]), batch_size=bs, shuffle=True)
    ilt = torch.tensor(il, dtype=torch.float32, device=device) if il is not None else None
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-5)
    mse = nn.MSELoss(); best, bstate, bad = -np.inf, None, 0
    Xf_t = torch.tensor(Xf, dtype=torch.float32); Xr_t = torch.tensor(Xr, dtype=torch.float32)
    for ep in range(epochs):
        model.train()
        for xf, xr, yy in dl:
            xf, xr, yy = xf.to(device), xr.to(device), yy.to(device)
            out = model(xf, xr)
            pred = out[0] if isinstance(out, tuple) else out.reshape(-1)
            loss = mse(pred, yy)          # ilddt aux omitted for the base (simple/stable)
            opt.zero_grad(); loss.backward(); opt.step()
        model.eval()
        with torch.no_grad():
            vp = []
            for i in range(0, len(val), 256):
                b = val[i:i+256]
                o = model(Xf_t[b].to(device), Xr_t[b].to(device))
                o = o[0] if isinstance(o, tuple) else o.reshape(-1)
                vp.append(o.cpu().numpy())
            vp = np.concatenate(vp); sc = pearsonr(y[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
        if sc > best: best, bad, bstate = sc, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(bstate)
    print(f"  pretrained val Pearson={best:.3f} (epoch stop)", flush=True)
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)        # SKEMPI scaled cache (encoder)
    ap.add_argument('--merged_dir', required=True)   # SKEMPI merged dir (encoder, for unscaled)
    ap.add_argument('--embed_dim', type=int, required=True)
    ap.add_argument('--out_ca', required=True)
    ap.add_argument('--out_simple', required=True)
    args = ap.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    d = np.load(args.cache, allow_pickle=True)
    Xf, Xr, y = d['Xf'].astype(np.float32), d['Xr'].astype(np.float32), d['y'].astype(np.float32)
    il = d['ilddt'].astype(np.float32) if 'ilddt' in d.files else None
    raw_ids = [str(s) for s in d['ids']]
    print(f"SKEMPI cache N={len(y)} scaled_dim={Xf.shape[1]} embed_dim={args.embed_dim}", flush=True)

    # ProtBFF cross-attention base (scaled)
    ca = FullModelCrossAttention(input_dim=5 * args.embed_dim).to(device)
    ca = train(ca, Xf, Xr, y, il, device, multitask=True)
    torch.save({'model_state_dict': ca.state_dict()}, args.out_ca)
    print(f"saved CA base -> {args.out_ca}", flush=True)

    # Simple base (unscaled)
    Uf, Ur, keep = unscaled_from_merged(args.merged_dir, raw_ids)
    ys = y[keep]
    print(f"Simple base: N={len(ys)} unscaled_dim={Uf.shape[1]}", flush=True)
    sm = DDGPredictorSimple(input_dim=args.embed_dim, hidden_dim=1024, dropout_rate=0.2, num_hidden=4).to(device)
    sm = train(sm, Uf.astype(np.float32), Ur.astype(np.float32), ys, None, device, multitask=False)
    torch.save(sm.state_dict(), args.out_simple)
    print(f"saved Simple base -> {args.out_simple}", flush=True)


if __name__ == '__main__':
    main()
