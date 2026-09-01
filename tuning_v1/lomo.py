#!/usr/bin/env python
"""Figure 2C recreation: leave-one-experimental-method-out, ESM-C + ProtBFF (dual).

For each experimental method with >=1% of the data, train the dual ProtBFF on all
mutations measured by OTHER methods and test on the held-out method, reporting Pearson.
As in the original panel, homology clustering is NOT applied here (the split is by
method, isolating measurement-technique effects). Method labels are joined from the raw
SKEMPI2 database by (PDB code, cleaned mutation); alignment to the cache is asserted
against ddG.
"""
import argparse, json, os, sys
import numpy as np, pandas as pd, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr
from sklearn.model_selection import train_test_split
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_cache
from protbff_arch import FullModel, make_loader, predict, BASE


def method_labels(master_csv, raw_csv):
    """-> dict {master '#Pdb' (== cache id) : experimental Method}."""
    raw = pd.read_csv(raw_csv, sep=';')
    raw['pdb'] = raw['#Pdb'].astype(str).str.split('_').str[0]
    raw['k'] = raw['pdb'] + '|' + raw['Mutation(s)_cleaned'].astype(str)
    mm = raw.groupby('k')['Method'].agg(lambda s: s.mode().iloc[0] if len(s.mode()) else s.iloc[0])
    m = pd.read_csv(master_csv)
    m['k'] = m['#Pdb_origin'].astype(str) + '|' + m['Mutation(s)_cleaned'].astype(str)
    m['Method'] = m['k'].map(mm)
    return dict(zip(m['#Pdb'].astype(str), m['Method']))


def train_eval(Xf, Xr, y, tr, te, D, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    trn, val = train_test_split(tr, test_size=0.12, random_state=seed)
    c = {**BASE, 'readout': 'dual', 'embed_dim': D, 'num_scores': 5, 'epochs': 60}
    y32 = y.astype(np.float32); il = np.zeros_like(y32)
    tl = make_loader(Xf, Xr, y32, il, trn, c['batch_size'], True)
    vl = make_loader(Xf, Xr, y32, il, val, c['batch_size'], False)
    el = make_loader(Xf, Xr, y32, il, te, c['batch_size'], False)
    model = FullModel(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    crit = nn.MSELoss(); best, bs, bad = -np.inf, None, 0
    for ep in range(c['epochs']):
        model.train()
        for xf, xr, yy, ii in tl:
            xf, xr, yy = xf.to(device), xr.to(device), yy.to(device)
            pd_, _ = model(xf, xr); loss = crit(pd_, yy)
            opt.zero_grad(); loss.backward(); opt.step()
        vp = predict(model, vl, device, 0.0, 1.0); vt = y[val]
        sc = pearsonr(vt, vp)[0] if (np.std(vp) > 1e-9 and np.std(vt) > 1e-9) else -np.inf
        if not np.isfinite(sc): sc = -np.inf
        if sc > best: best, bad, bs = sc, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= 12: break
    if bs: model.load_state_dict(bs)
    p = predict(model, el, device, 0.0, 1.0)
    return pearsonr(y[te], p)[0], spearmanr(y[te], p)[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--embed_dim', type=int, required=True)
    ap.add_argument('--master', default='data/cross_validation_folds_mva/60_percent/folds_60pct.csv')
    ap.add_argument('--raw', default='/n/netscratch/shakhnovich_lab/Lab/jwang/rde_linear_mva/data/SKEMPI_v2/skempi_v2.csv')
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=2)
    ap.add_argument('--min_frac', type=float, default=0.01)
    args = ap.parse_args()

    Xf, Xr, y, il, ids = load_cache(args.cache)
    raw_ids = [str(s) for s in np.load(args.cache, allow_pickle=True)['ids']]   # "<idx>_<PDB>" == master #Pdb
    m2m = method_labels(args.master, args.raw)
    labels = np.array([m2m.get(rid, None) for rid in raw_ids], dtype=object)
    cov = np.mean([l is not None for l in labels])
    assert cov > 0.98, f"method-label coverage only {cov:.2%} (id join failed)"
    print(f"N={len(y)} embed_dim={args.embed_dim} | method coverage {cov:.1%} (id-joined)", flush=True)

    vc = pd.Series(labels).value_counts()
    methods = [m for m in vc.index if vc[m] / len(y) >= args.min_frac]
    res = {}
    for M in methods:
        te = np.where(labels == M)[0]; tr = np.where((labels != M) & (~pd.isna(labels)))[0]
        pr, sp = [], []
        for s in range(args.seeds):
            a, b = train_eval(Xf, Xr, y, tr, te, args.embed_dim, seed=100 + s)
            pr.append(a); sp.append(b)
        res[M] = dict(pearson=float(np.mean(pr)), spearman=float(np.mean(sp)),
                      frac=float(len(te) / len(y)), n=int(len(te)))
        print(f"  {M:8s} frac={len(te)/len(y)*100:4.1f}%  Pearson={np.mean(pr):.3f}", flush=True)
        json.dump(res, open(args.out, 'w'), indent=2)
    print(f"-> {args.out}", flush=True)


if __name__ == '__main__':
    main()
