#!/usr/bin/env python
"""SARS-CoV-2 DMS few-shot curves for Figure 3B-D, with the DUAL readout.

Builds BOTH the +ProtBFF (dual) features and the bare-encoder baseline from the same
merged DMS directory (per-residue embeddings + biophysical scores + ddG), so the two
curves are on identical rows. For each training-set fraction it trains the dual ProtBFF
on the scaled 5-block features and fits a ridge[D|S] bare baseline on the unscaled pooled
embedding, and reports Spearman on the held-out test split. merge_scores convention:
block k = maxpool(score_k * diff), diff_fwd = Xr_emb (mt-wt), lDDT inverted (1-lddt).
"""
import argparse, json, os, sys, glob
from pathlib import Path
import numpy as np, torch
from scipy.stats import spearmanr, pearsonr
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_arch import FullModel, make_loader, predict, BASE

SCORES = ['interface', 'burial', 'lDDT', 'SASA', 'dihedral']


def maxpool(score, diff):
    return np.max(diff * score[:, None], axis=0)


def build(merged_dir):
    files = sorted(glob.glob(merged_dir + '/merged_*.npz'),
                   key=lambda p: int(Path(p).stem.split('_')[1]))
    Xs_f, Xs_r, Uf, Ur, ys = [], [], [], [], []
    need = {'Xf', 'Xr', 'interface', 'burial', 'lddt', 'sasa', 'dihedral', 'ddG'}
    for fp in files:
        d = np.load(fp, allow_pickle=True)
        if not need.issubset(set(d.files)):
            continue
        ef, er = d['Xf'].astype(np.float64), d['Xr'].astype(np.float64)   # (L,D) per-residue emb
        L = ef.shape[0]
        sc = {'interface': np.ravel(d['interface']).astype(float),
              'burial': np.ravel(d['burial']).astype(float),
              'lDDT': 1.0 - np.ravel(d['lddt']).astype(float),
              'SASA': np.ravel(d['sasa']).astype(float),
              'dihedral': np.ravel(d['dihedral']).astype(float)}
        if any(len(sc[k]) != L for k in sc):
            continue
        diff_fwd, diff_rev = er, ef                                       # mt-wt , wt-mt
        Xs_f.append(np.concatenate([maxpool(sc[k], diff_fwd) for k in SCORES]))
        Xs_r.append(np.concatenate([maxpool(sc[k], diff_rev) for k in SCORES]))
        Uf.append(np.max(diff_fwd, axis=0)); Ur.append(np.max(diff_rev, axis=0))
        ys.append(float(d['ddG']))
    return (np.array(Xs_f, np.float32), np.array(Xs_r, np.float32),
            np.array(Uf, np.float64), np.array(Ur, np.float64), np.array(ys, np.float64))


def train_model(Xf, Xr, y, tr, te, D, nscores, seed):
    """Train the ProtBFF architecture (dual readout) with `nscores` blocks.
    nscores=5 -> scaled ProtBFF; nscores=1 -> unscaled embedding through the same model
    (the original 'bare' baseline: biophysical scaling off)."""
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # small val split from train for early stopping
    if len(tr) >= 20:
        trn, val = train_test_split(tr, test_size=0.15, random_state=seed)
    else:
        trn, val = tr, tr
    c = {**BASE, 'readout': 'dual', 'embed_dim': D, 'num_scores': nscores, 'epochs': 60}
    y32 = y.astype(np.float32); il = np.zeros_like(y32)
    tl = make_loader(Xf, Xr, y32, il, trn, c['batch_size'], True)
    vl = make_loader(Xf, Xr, y32, il, val, c['batch_size'], False)
    el = make_loader(Xf, Xr, y32, il, te, c['batch_size'], False)
    model = FullModel(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    import torch.nn as nn
    crit = nn.MSELoss(); best, best_state, bad = -np.inf, None, 0
    for ep in range(c['epochs']):
        model.train()
        for xf, xr, yy, ii in tl:
            xf, xr, yy = xf.to(device), xr.to(device), yy.to(device)
            pd_, _ = model(xf, xr); loss = crit(pd_, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        vp = predict(model, vl, device, 0.0, 1.0); vt = y[val]
        sc = spearmanr(vt, vp)[0] if (np.std(vp) > 1e-9 and np.std(vt) > 1e-9) else -np.inf
        if not np.isfinite(sc): sc = -np.inf
        if sc > best: best, bad, best_state = sc, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 12: break
    if best_state: model.load_state_dict(best_state)
    p = predict(model, el, device, 0.0, 1.0)
    return spearmanr(y[te], p)[0]


def bare_ridge(Uf, Ur, y, tr, te, seed):
    Dp, Sp = Uf - Ur, Uf + Ur
    X = np.hstack([Dp, Sp])
    trn, val = train_test_split(tr, test_size=0.15, random_state=seed) if len(tr) >= 20 else (tr, tr)
    best_a, best_v = 1.0, -np.inf
    for a in (1, 10, 100, 1000, 10000):
        m = Ridge(alpha=a).fit(X[trn], y[trn]); vp = m.predict(X[val])
        v = spearmanr(y[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
        if v > best_v: best_v, best_a = v, a
    m = Ridge(alpha=best_a).fit(X[tr], y[tr]); p = m.predict(X[te])
    return spearmanr(y[te], p)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--merged_dir', required=True)
    ap.add_argument('--tag', required=True)          # e.g. ace2_prosst
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=3)
    ap.add_argument('--fracs', default='10,20,30,40,50,60,70,80')
    args = ap.parse_args()

    Xf, Xr, Uf, Ur, y = build(args.merged_dir)
    D = Uf.shape[1]
    print(f"[{args.tag}] N={len(y)} embed_dim={D} scaled={Xf.shape}", flush=True)
    Uf32, Ur32 = Uf.astype(np.float32), Ur.astype(np.float32)
    fracs = [int(x) for x in args.fracs.split(',')]
    res = {'tag': args.tag, 'N': int(len(y)), 'embed_dim': int(D), 'curves': {}}
    for frac in fracs:
        pb, bm, br = [], [], []
        for s in range(args.seeds):
            tr, te = train_test_split(np.arange(len(y)), test_size=1 - frac / 100.0, random_state=42 + s)
            pb.append(train_model(Xf, Xr, y, tr, te, D, 5, seed=100 + s))       # ProtBFF (scaled)
            bm.append(train_model(Uf32, Ur32, y, tr, te, D, 1, seed=100 + s))   # bare: unscaled thru model (original)
            br.append(bare_ridge(Uf, Ur, y, tr, te, seed=100 + s))             # bare: ridge[D|S] (reference)
        res['curves'][frac] = {'protbff': float(np.mean(pb)), 'protbff_sd': float(np.std(pb)),
                               'bare': float(np.mean(bm)), 'bare_sd': float(np.std(bm)),
                               'bare_ridge': float(np.mean(br)), 'bare_ridge_sd': float(np.std(br))}
        print(f"  {frac:3d}%  ProtBFF S={np.mean(pb):.3f}  bare(unscaled) S={np.mean(bm):.3f}  "
              f"bare(ridge) S={np.mean(br):.3f}", flush=True)
        json.dump(res, open(args.out, 'w'), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == '__main__':
    main()
