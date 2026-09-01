#!/usr/bin/env python
"""Architecture upgrade: symmetric-aware readout for ProtBFF.

Motivation (rigorously established): the paper readout (head(h_f)-head(h_r))/2 is
hard-wired ANTISYMMETRIC -> sees only D=x_f-x_r. The symmetric part S=(x_f+x_r)/2
carries MORE signal (ridge S=0.418 > D=0.355), and ridge on [D|S] hits 0.444 while
the net (D-only) gets 0.389. This module lets the SAME pooling+MLP feed a
configurable readout so the model can use both halves.

Readout modes over shared per-branch features h_f=mlp(pool(x_f)), h_r=mlp(pool(x_r)):
  antisym    : (head(h_f) - head(h_r))/2          (== paper baseline)
  dual       : head_a(h_f - h_r) + head_s(h_f + h_r)
  concat_ds  : head([h_f - h_r ; h_f + h_r])       (mirrors ridge [D|S])
  concat_pair: head([h_f ; h_r])                   (most general)

Everything else (pooling, MLP feature extractor, regularization) is UNCHANGED.
Protocol = clean condition B: no target standardization, ilddt_weight=0,
cluster-aware val, epoch selected on val Pearson. Multi-seed + paired stats.
"""
import argparse, json, time
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr, wilcoxon
from sklearn.linear_model import Ridge

import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import (load_cache, load_clusters, load_folds, cluster_aware_val,
                           make_loader, predict, CrossEmbeddingAttention)

# --------------------------------------------------------------------------- model
class PairReadout(nn.Module):
    def __init__(self, input_dim, hidden_dim, dropout, num_hidden, readout='antisym'):
        super().__init__()
        layers, in_dim = [], input_dim
        for i in range(num_hidden):
            out_dim = hidden_dim if i == 0 else max(1, hidden_dim // (2 ** i))
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.readout = readout
        self.drop = nn.Dropout(dropout * 0.8)
        if readout == 'antisym':
            self.head = nn.Linear(in_dim, 1)
        elif readout == 'dual':
            self.head_a = nn.Linear(in_dim, 1); self.head_s = nn.Linear(in_dim, 1)
        elif readout in ('concat_ds', 'concat_pair'):
            self.head = nn.Linear(2 * in_dim, 1)
        else:
            raise ValueError(readout)
    def forward(self, xf, xr):
        hf, hr = self.mlp(xf), self.mlp(xr)
        if self.readout == 'antisym':
            return (self.head(self.drop(hf)) - self.head(self.drop(hr))).squeeze(-1) / 2
        if self.readout == 'dual':
            return (self.head_a(self.drop(hf - hr)) + self.head_s(self.drop(hf + hr))).squeeze(-1)
        if self.readout == 'concat_ds':
            return self.head(self.drop(torch.cat([hf - hr, hf + hr], -1))).squeeze(-1)
        if self.readout == 'concat_pair':
            return self.head(self.drop(torch.cat([hf, hr], -1))).squeeze(-1)

class FullModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        ds = c.get('dropout_scale', 1.0)
        self.pooling = CrossEmbeddingAttention(
            reduced_dim=c['reduced_dim'], num_scores=c.get('num_scores', 5), embed_dim=c['embed_dim'],
            proj_dropout=c['proj_dropout']*ds, attn_dropout=c['attn_dropout']*ds, num_heads=c['num_heads'])
        self.ddg_predictor = PairReadout(c['reduced_dim'], c['hidden_dim'], c['mlp_dropout']*ds,
                                         c['num_hidden'], readout=c.get('readout', 'antisym'))
        self.ilddt_predictor = PairReadout(c['reduced_dim'], c['hidden_dim'], c['mlp_dropout']*ds,
                                           c['num_hidden'], readout=c.get('ilddt_readout', 'antisym'))
    def forward(self, xf, xr):
        xfp, xrp = self.pooling(xf), self.pooling(xr)
        return self.ddg_predictor(xfp, xrp), self.ilddt_predictor(xfp, xrp)

# --------------------------------------------------------------------------- train one model
def train_once(c, device, Xf, Xr, y, il, tr, val, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    if c.get('standardize', False):
        ym, ys_ = float(y[tr].mean()), float(y[tr].std()+1e-8)
        im, is_ = float(il[tr].mean()), float(il[tr].std()+1e-8)
    else:
        ym, ys_, im, is_ = 0.0, 1.0, 0.0, 1.0
    ysd = (y-ym)/ys_; ilz = (il-im)/is_
    bs = c['batch_size']
    tl = make_loader(Xf, Xr, ysd, ilz, tr, bs, True)
    vl = make_loader(Xf, Xr, ysd, ilz, val, bs, False)
    el = make_loader(Xf, Xr, ysd, ilz, te, bs, False)
    model = FullModel(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    crit = nn.SmoothL1Loss() if c.get('loss') == 'huber' else nn.MSELoss()
    w = c.get('ilddt_weight', 0.0)
    best, best_state, bad = -np.inf, None, 0
    for ep in range(1, c['epochs']+1):
        model.train()
        for xf, xr, yy, ii in tl:
            xf, xr, yy, ii = xf.to(device), xr.to(device), yy.to(device), ii.to(device)
            pd_, pi_ = model(xf, xr)
            loss = crit(pd_, yy) + (w*crit(pi_, ii) if w > 0 else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        vp = predict(model, vl, device, ym, ys_); vt = y[val]
        sc = pearsonr(vt, vp)[0] if (vt.std() > 1e-8 and vp.std() > 1e-8) else -np.inf
        if not np.isfinite(sc): sc = -np.inf
        if sc > best:
            best, bad = sc, 0
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c.get('patience', 15): break
    if best_state is not None: model.load_state_dict(best_state)
    return predict(model, el, device, ym, ys_)

# --------------------------------------------------------------------------- CV over a variant
def run_variant(c, folds, Xf, Xr, y, il, ids, cmap, device, seeds):
    per_fold_r, per_fold_s = [], []            # ensemble (avg over seeds) per fold
    per_fold_r_single = []                     # mean single-seed r per fold (stability)
    pooled_t, pooled_p = [], []
    for fi, (tr_all, te) in enumerate(folds):
        tr, val = cluster_aware_val(tr_all, ids, cmap, c.get('val_frac', 0.10), c['random_state']+fi)
        seed_preds = []
        for s in range(seeds):
            seed_preds.append(train_once(c, device, Xf, Xr, y, il, tr, val, te, seed=c['random_state']+fi+1000*s))
        seed_preds = np.array(seed_preds)                 # (seeds, n_test)
        ens = seed_preds.mean(0); tt = y[te]
        per_fold_r.append(pearsonr(tt, ens)[0]); per_fold_s.append(spearmanr(tt, ens)[0])
        per_fold_r_single.append(np.mean([pearsonr(tt, sp)[0] for sp in seed_preds]))
        pooled_t += list(tt); pooled_p += list(ens)
    return dict(fold_r=np.array(per_fold_r), fold_s=np.array(per_fold_s),
                fold_r_single=np.array(per_fold_r_single),
                mean_r=float(np.mean(per_fold_r)), sem_r=float(np.std(per_fold_r)/np.sqrt(len(folds))),
                mean_s=float(np.mean(per_fold_s)), sem_s=float(np.std(per_fold_s)/np.sqrt(len(folds))),
                pooled_r=float(pearsonr(pooled_t, pooled_p)[0]),
                pooled_s=float(spearmanr(pooled_t, pooled_p)[0]))

# --------------------------------------------------------------------------- ridge reference
def ridge_reference(folds, Xf, Xr, y, ids, cmap, alpha_grid=(100, 300, 1000, 3000, 10000)):
    """Ridge on [D|S] with alpha selected per-fold on the SAME cluster-aware val (rigorous)."""
    D = Xf - Xr; S = (Xf + Xr) / 2.0; X = np.hstack([D, S]).astype(np.float64)
    fr, fs, pt, pp = [], [], [], []
    for fi, (tr_all, te) in enumerate(folds):
        tr, val = cluster_aware_val(tr_all, ids, cmap, 0.10, 42+fi)
        best_a, best_v = None, -np.inf
        for a in alpha_grid:
            m = Ridge(alpha=a).fit(X[tr], y[tr]); vp = m.predict(X[val])
            v = pearsonr(y[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
            if v > best_v: best_v, best_a = v, a
        m = Ridge(alpha=best_a).fit(X[np.concatenate([tr, val])], y[np.concatenate([tr, val])])
        p = m.predict(X[te])
        fr.append(pearsonr(y[te], p)[0]); fs.append(spearmanr(y[te], p)[0]); pt += list(y[te]); pp += list(p)
    return dict(fold_r=np.array(fr), fold_s=np.array(fs), mean_r=float(np.mean(fr)),
                sem_r=float(np.std(fr)/np.sqrt(len(folds))), mean_s=float(np.mean(fs)),
                sem_s=float(np.std(fs)/np.sqrt(len(folds))),
                pooled_r=float(pearsonr(pt, pp)[0]), pooled_s=float(spearmanr(pt, pp)[0]))

BASE = dict(lr=1e-4, weight_decay=1e-5, epochs=50, batch_size=32, patience=15,
            random_state=42, val_frac=0.10, ilddt_weight=0.0, loss='mse', standardize=False,
            reduced_dim=512, hidden_dim=768, embed_dim=768, num_hidden=3,
            mlp_dropout=0.35, proj_dropout=0.4, attn_dropout=0.4, num_heads=8,
            dropout_scale=1.0, num_scores=5)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True); ap.add_argument('--folds_dir', required=True)
    ap.add_argument('--clusters', required=True); ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--variants', default='antisym,dual,concat_ds,concat_pair')
    ap.add_argument('--embed_dim', type=int, default=768,
                    help='per-score embedding dim (768 ProSST, 1536 ESM3)')
    ap.add_argument('--smoke', action='store_true')
    args = ap.parse_args()
    BASE['embed_dim'] = args.embed_dim
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xf, Xr, y, il, ids = load_cache(args.cache)
    cmap = load_clusters(args.clusters); folds = load_folds(args.folds_dir, ids)
    print(f"device={device} N={len(y)} folds={len(folds)} seeds={args.seeds}", flush=True)
    if args.smoke:
        c = {**BASE, 'epochs': 3, 'patience': 3, 'readout': 'concat_ds'}
        r = run_variant(c, folds[:2], Xf, Xr, y, il, ids, cmap, device, seeds=1)
        print("smoke concat_ds:", {k: r[k] for k in ('mean_r','mean_s','pooled_r','pooled_s')}); return

    variants = args.variants.split(',')
    res = {}
    print("\n=== ridge [D|S] reference (alpha val-selected per fold) ===", flush=True)
    rg = ridge_reference(folds, Xf, Xr, y, ids, cmap)
    print(f"  ridge[D|S]  mean_r={rg['mean_r']:.4f}±{rg['sem_r']:.3f}  mean_s={rg['mean_s']:.4f}  "
          f"pooled_r={rg['pooled_r']:.4f} pooled_s={rg['pooled_s']:.4f}", flush=True)
    res['ridge_DS'] = {k: (v.tolist() if isinstance(v, np.ndarray) else v) for k, v in rg.items()}
    for v in variants:
        t0 = time.time()
        c = {**BASE, 'readout': v}
        r = run_variant(c, folds, Xf, Xr, y, il, ids, cmap, device, seeds=args.seeds)
        res[v] = {k: (val.tolist() if isinstance(val, np.ndarray) else val) for k, val in r.items()}
        print(f"[{v:11s}] mean_r={r['mean_r']:.4f}±{r['sem_r']:.3f}  mean_s={r['mean_s']:.4f}±{r['sem_s']:.3f}  "
              f"pooled_r={r['pooled_r']:.4f} pooled_s={r['pooled_s']:.4f}  ({time.time()-t0:.0f}s)", flush=True)
    # paired stats vs antisym baseline
    base = np.array(res['antisym']['fold_r']); base_s = np.array(res['antisym']['fold_s'])
    print("\n=== PAIRED vs antisym baseline (10 folds, Wilcoxon) ===", flush=True)
    print(f"{'variant':12s}  dPearson(mean±sd)   p      dSpearman(mean±sd)  p", flush=True)
    for v in variants:
        if v == 'antisym': continue
        dr = np.array(res[v]['fold_r']) - base; dsp = np.array(res[v]['fold_s']) - base_s
        pr = wilcoxon(dr)[1] if np.any(dr) else 1.0
        ps = wilcoxon(dsp)[1] if np.any(dsp) else 1.0
        print(f"{v:12s}  {dr.mean():+.4f}±{dr.std():.3f}   {pr:.3f}   {dsp.mean():+.4f}±{dsp.std():.3f}   {ps:.3f}", flush=True)
    json.dump(res, open(args.out, 'w'), indent=2, default=float)
    print("\n=== SUMMARY (mean-of-folds Pearson / Spearman ; pooled) ===", flush=True)
    print(f"{'model':14s}  meanP±sem    meanS       poolP  poolS", flush=True)
    for k in ['antisym'] + [v for v in variants if v != 'antisym'] + ['ridge_DS']:
        a = res[k]
        print(f"{k:14s}  {a['mean_r']:.3f}±{a['sem_r']:.3f}  {a['mean_s']:.3f}      {a['pooled_r']:.3f}  {a['pooled_s']:.3f}", flush=True)

if __name__ == '__main__':
    main()
