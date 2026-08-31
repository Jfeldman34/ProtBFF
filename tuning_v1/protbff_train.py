#!/usr/bin/env python
"""ProtBFF training with Tier-1 protocol fixes + Tier-2 hyperparameters.

Architecture is UNCHANGED from the paper model (CrossEmbeddingAttention pooling
over 5 score blocks + antisymmetric DDGPredictor readout). Only the training
protocol and hyperparameters are configurable.

Tier-1 (protocol):
  - MVA 60% homology-aware split (train/test), cluster-aware VAL held out of train
  - epoch selected on VAL Pearson (not val loss)
  - refit on train+val at the selected epoch  (--refit)
  - seed ensemble: average predictions over K seeds  (--seeds K)
  - per-fold calibration: affine map fit on VAL, applied to test (pooled metric only)

Tier-2 (hyperparameters, no architecture change):
  - AdamW + weight_decay          - num_hidden (head depth)
  - ilddt_weight (0 disables aux) - loss: mse | huber
  - lr, batch_size, dropout scale - target standardization
"""
import argparse, json, os, time
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
from scipy.stats import pearsonr, spearmanr

# --------------------------------------------------------------------------- model
class CrossEmbeddingAttention(nn.Module):
    def __init__(self, reduced_dim=512, num_scores=5, embed_dim=768,
                 proj_dropout=0.4, attn_dropout=0.4, num_heads=8):
        super().__init__()
        self.embed_dim, self.num_scores, self.reduced_dim = embed_dim, num_scores, reduced_dim
        self.embedding_projections = nn.ModuleList([
            nn.Sequential(nn.Linear(embed_dim, reduced_dim), nn.ReLU(), nn.Dropout(proj_dropout))
            for _ in range(num_scores)])
        self.cross_attention = nn.MultiheadAttention(reduced_dim, num_heads,
                                                     dropout=attn_dropout, batch_first=True)
        self.attention_norm = nn.LayerNorm(reduced_dim)
        self.final_attention = nn.Sequential(nn.Linear(num_scores, 32), nn.ReLU(),
                                             nn.Dropout(proj_dropout), nn.Linear(32, num_scores))
    def forward(self, x):
        b = x.size(0)
        emb = x.view(b, self.num_scores, self.embed_dim)
        proj = torch.stack([p(emb[:, i, :]) for i, p in enumerate(self.embedding_projections)], dim=1)
        att, _ = self.cross_attention(proj, proj, proj)
        att = self.attention_norm(att + proj)
        att_t = att.transpose(1, 2)
        w = F.softmax(self.final_attention(att_t), dim=-1)
        return (w * att_t).sum(dim=-1)

class DDGPredictor(nn.Module):
    def __init__(self, input_dim=512, hidden_dim=768, dropout_rate=0.35, num_hidden=3):
        super().__init__()
        layers, in_dim = [], input_dim
        for i in range(num_hidden):
            out_dim = hidden_dim if i == 0 else max(1, hidden_dim // (2 ** i))
            layers += [nn.Linear(in_dim, out_dim), nn.ReLU(), nn.Dropout(dropout_rate)]
            in_dim = out_dim
        self.mlp = nn.Sequential(*layers)
        self.head = nn.Sequential(nn.Dropout(dropout_rate * 0.8), nn.Linear(in_dim, 1))
    def forward(self, xf, xr):
        return (self.head(self.mlp(xf)) - self.head(self.mlp(xr))) / 2

class FullModel(nn.Module):
    def __init__(self, c):
        super().__init__()
        ds = c.get('dropout_scale', 1.0)
        self.pooling = CrossEmbeddingAttention(
            reduced_dim=c['reduced_dim'], num_scores=c.get('num_scores', 5), embed_dim=c['embed_dim'],
            proj_dropout=c['proj_dropout']*ds, attn_dropout=c['attn_dropout']*ds, num_heads=c['num_heads'])
        mk = lambda: DDGPredictor(c['reduced_dim'], c['hidden_dim'], c['mlp_dropout']*ds, c['num_hidden'])
        self.ddg_predictor, self.ilddt_predictor = mk(), mk()
    def forward(self, xf, xr):
        xfp, xrp = self.pooling(xf), self.pooling(xr)
        return self.ddg_predictor(xfp, xrp).squeeze(-1), self.ilddt_predictor(xfp, xrp).squeeze(-1)

# --------------------------------------------------------------------------- data
def load_cache(path):
    d = np.load(path, allow_pickle=True)
    ids = np.array([str(s).split('_', 1)[1] if '_' in str(s) else str(s) for s in d['ids']])
    return (d['Xf'].astype(np.float32), d['Xr'].astype(np.float32),
            d['y'].astype(np.float32), d['ilddt'].astype(np.float32), ids)

def load_clusters(path):
    import csv
    m = {}
    with open(path) as fh:
        r = csv.reader(fh, delimiter='\t'); next(r, None)
        for row in r:
            if len(row) >= 2: m[row[0]] = row[1]
    return m

def load_folds(folds_dir, ids, n=10):
    def codes(p):
        s = set()
        for line in open(p):
            t = line.strip()
            if t and '_' in t: s.add(t.split('_', 1)[1])
        return s
    folds = []
    for k in range(1, n + 1):
        tr_c = codes(os.path.join(folds_dir, f'fold_{k}', 'train_complex_ids.txt'))
        te_c = codes(os.path.join(folds_dir, f'fold_{k}', 'test_complex_ids.txt'))
        tr = np.array([i for i, c in enumerate(ids) if c in tr_c])
        te = np.array([i for i, c in enumerate(ids) if c in te_c])
        folds.append((tr, te))
    return folds

def cluster_aware_val(train_idx, ids, cmap, val_frac, seed):
    rng = np.random.default_rng(seed)
    by = {}
    for i in train_idx:
        by.setdefault(cmap.get(ids[i], f'__solo_{ids[i]}'), []).append(i)
    cl = list(by); rng.shuffle(cl)
    target = int(round(val_frac * len(train_idx))); val = []
    for c in cl:
        if len(val) >= target: break
        if val and len(val) + len(by[c]) > 1.5 * target: continue
        val += by[c]
    if not val: val = by[min(cl, key=lambda c: len(by[c]))]
    val = np.array(sorted(val)); tr = np.array(sorted(set(map(int, train_idx)) - set(val.tolist())))
    return tr, val

# --------------------------------------------------------------------------- train
def make_loader(Xf, Xr, y, il, idx, bs, shuffle):
    ds = TensorDataset(torch.from_numpy(Xf[idx]), torch.from_numpy(Xr[idx]),
                       torch.from_numpy(y[idx]), torch.from_numpy(il[idx]))
    return DataLoader(ds, batch_size=bs, shuffle=shuffle)

def predict(model, loader, device, ymean, ystd):
    model.eval(); P = []
    with torch.no_grad():
        for xf, xr, _, _ in loader:
            p, _ = model(xf.to(device), xr.to(device))
            P.append(p.cpu().numpy())
    return np.concatenate(P) * ystd + ymean

def loss_fn(name):
    if name == 'huber': base = nn.SmoothL1Loss()
    else: base = nn.MSELoss()
    return base

def train_once(c, device, Xf, Xr, y, il, tr, val, te, seed, max_epochs=None, fixed_epochs=None):
    """Train one model. If fixed_epochs is set, train that many epochs on tr (no early stop).
    Else early-stop on val Pearson and return best epoch."""
    torch.manual_seed(seed); np.random.seed(seed)
    if c.get('standardize', True):
        ymean, ystd = float(y[tr].mean()), float(y[tr].std() + 1e-8)
        ilm, ils = float(il[tr].mean()), float(il[tr].std() + 1e-8)
    else:
        ymean, ystd, ilm, ils = 0.0, 1.0, 0.0, 1.0
    ys = (y - ymean) / ystd
    ilz = (il - ilm) / ils
    bs = c['batch_size']
    tl = make_loader(Xf, Xr, ys, ilz, tr, bs, True)
    vl = make_loader(Xf, Xr, ys, ilz, val, bs, False) if len(val) else None
    el = make_loader(Xf, Xr, ys, ilz, te, bs, False)

    model = FullModel(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    crit = loss_fn(c.get('loss', 'mse')); w = c.get('ilddt_weight', 0.2)
    epochs = fixed_epochs or c['epochs']
    best_score, best_state, best_ep, bad = -np.inf, None, epochs, 0
    patience = c.get('patience', 15)

    for ep in range(1, epochs + 1):
        model.train()
        for xf, xr, yy, ii in tl:
            xf, xr, yy, ii = xf.to(device), xr.to(device), yy.to(device), ii.to(device)
            pd_, pi_ = model(xf, xr)
            loss = crit(pd_, yy) + (w * crit(pi_, ii) if w > 0 else 0.0)
            opt.zero_grad(); loss.backward(); opt.step()
        if fixed_epochs is None and vl is not None:
            vp = predict(model, vl, device, ymean, ystd); vt = y[val]
            sc = pearsonr(vt, vp)[0] if (vt.std() > 1e-8 and vp.std() > 1e-8) else -np.inf
            if not np.isfinite(sc): sc = -np.inf
            if sc > best_score:
                best_score, best_ep, bad = sc, ep, 0
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            else:
                bad += 1
                if bad >= patience: break
    if fixed_epochs is None and best_state is not None:
        model.load_state_dict(best_state)

    val_pred = predict(model, vl, device, ymean, ystd) if vl is not None else None
    test_pred = predict(model, el, device, ymean, ystd)
    return dict(best_ep=best_ep, best_val=best_score, val_pred=val_pred, test_pred=test_pred,
                val_true=y[val] if len(val) else None, test_true=y[te])

# --------------------------------------------------------------------------- CV
def run_cv(c, folds, Xf, Xr, y, il, ids, cmap, device, fold_subset=None,
           refit=False, seeds=1, calibrate=False, verbose=True):
    idxs = fold_subset if fold_subset is not None else range(len(folds))
    rows = []; pooled_t, pooled_p, pooled_pc = [], [], []
    for fi in idxs:
        tr_all, te = folds[fi]
        tr, val = cluster_aware_val(tr_all, ids, cmap, c.get('val_frac', 0.10), c['random_state'] + fi)
        preds = []
        best_eps, val_ps, val_ts = [], None, None
        for s in range(seeds):
            r = train_once(c, device, Xf, Xr, y, il, tr, val, te, seed=c['random_state'] + fi + 1000 * s)
            if refit:
                r2 = train_once(c, device, Xf, Xr, y, il, np.concatenate([tr, val]), np.array([], int),
                                te, seed=c['random_state'] + fi + 1000 * s, fixed_epochs=r['best_ep'])
                preds.append(r2['test_pred'])
            else:
                preds.append(r['test_pred'])
            best_eps.append(r['best_ep']); val_ps, val_ts = r['val_pred'], r['val_true']
        tp = np.mean(preds, axis=0); tt = y[te]
        pr, sp = pearsonr(tt, tp)[0], spearmanr(tt, tp)[0]
        # calibration: affine fit on val, applied to test
        tp_cal = tp
        if calibrate and val_ps is not None and val_ps.std() > 1e-8:
            A = np.vstack([val_ps, np.ones_like(val_ps)]).T
            slope, intercept = np.linalg.lstsq(A, val_ts, rcond=None)[0]
            tp_cal = slope * tp + intercept
        rows.append(dict(fold=fi, n=len(te), pearson=pr, spearman=sp,
                         val_pearson=(pearsonr(val_ts, val_ps)[0] if val_ps is not None else np.nan),
                         best_ep=float(np.mean(best_eps))))
        pooled_t += list(tt); pooled_p += list(tp); pooled_pc += list(tp_cal)
        if verbose:
            print(f"  fold {fi+1:2d} n={len(te):4d}  test P={pr:.3f} S={sp:.3f}  "
                  f"val P={rows[-1]['val_pearson']:.3f}  ep~{rows[-1]['best_ep']:.0f}", flush=True)
    P = np.array([r['pearson'] for r in rows]); S = np.array([r['spearman'] for r in rows])
    V = np.array([r['val_pearson'] for r in rows])
    out = dict(rows=rows,
               mean_pearson=float(P.mean()), mean_spearman=float(S.mean()),
               mean_val_pearson=float(np.nanmean(V)),
               pooled_pearson=float(pearsonr(pooled_t, pooled_p)[0]),
               pooled_spearman=float(spearmanr(pooled_t, pooled_p)[0]),
               pooled_pearson_cal=float(pearsonr(pooled_t, pooled_pc)[0]),
               pooled_spearman_cal=float(spearmanr(pooled_t, pooled_pc)[0]))
    return out

# --------------------------------------------------------------------------- main
BASE_CFG = dict(lr=1e-4, weight_decay=1e-5, epochs=50, batch_size=32, patience=15,
                random_state=42, val_frac=0.10, ilddt_weight=0.2, loss='mse',
                reduced_dim=512, hidden_dim=768, embed_dim=768, num_hidden=3,
                mlp_dropout=0.35, proj_dropout=0.4, attn_dropout=0.4, num_heads=8,
                dropout_scale=1.0, num_scores=5, standardize=True)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--folds_dir', required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--mode', choices=['sweep', 'final', 'smoke', 'diag'], default='smoke')
    ap.add_argument('--out', required=True)
    ap.add_argument('--n_configs', type=int, default=32)
    ap.add_argument('--sweep_folds', default='0,1,4')
    ap.add_argument('--seeds', type=int, default=5)
    ap.add_argument('--config_json', default=None)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xf, Xr, y, il, ids = load_cache(args.cache)
    cmap = load_clusters(args.clusters); folds = load_folds(args.folds_dir, ids)
    print(f"device={device}  N={len(y)}  folds={len(folds)}  clusters={len(set(cmap.values()))}", flush=True)

    if args.mode == 'smoke':
        c = {**BASE_CFG, 'epochs': 4, 'patience': 4}
        r = run_cv(c, folds, Xf, Xr, y, il, ids, cmap, device, fold_subset=[0, 4], seeds=1, calibrate=True)
        print(json.dumps({k: r[k] for k in r if k != 'rows'}, indent=2)); return

    if args.mode == 'sweep':
        rng = np.random.default_rng(0)
        grid = dict(lr=[5e-5, 1e-4, 2e-4], weight_decay=[1e-4, 1e-3, 1e-2],
                    batch_size=[32, 64], num_hidden=[1, 2, 3], ilddt_weight=[0.0, 0.2],
                    loss=['mse', 'huber'], dropout_scale=[0.7, 1.0])
        sweep_folds = [int(x) for x in args.sweep_folds.split(',')]
        configs = [{**BASE_CFG}]  # index 0 = AdamW baseline (paper HPs)
        seen = set()
        while len(configs) < args.n_configs + 1:
            cfg = {**BASE_CFG, **{k: (rng.choice(v).item() if not isinstance(v[0], str)
                                      else str(rng.choice(v))) for k, v in grid.items()}}
            key = tuple(cfg[k] for k in grid)
            if key in seen: continue
            seen.add(key); configs.append(cfg)
        results = []
        for gi, cfg in enumerate(configs):
            t0 = time.time()
            r = run_cv(cfg, folds, Xf, Xr, y, il, ids, cmap, device,
                       fold_subset=sweep_folds, seeds=1, calibrate=False, verbose=False)
            ov = {k: cfg[k] for k in grid}
            results.append(dict(idx=gi, **ov, mean_val_pearson=r['mean_val_pearson'],
                                mean_test_pearson=r['mean_pearson'], mean_test_spearman=r['mean_spearman']))
            print(f"[{gi:2d}/{len(configs)-1}] valP={r['mean_val_pearson']:.3f} "
                  f"testP={r['mean_pearson']:.3f} testS={r['mean_spearman']:.3f} "
                  f"({time.time()-t0:.0f}s) {ov}", flush=True)
        results.sort(key=lambda d: -d['mean_val_pearson'])
        json.dump(results, open(args.out, 'w'), indent=2)
        print("\n=== TOP 5 by VAL Pearson ==="); 
        for d in results[:5]:
            print(f"  valP={d['mean_val_pearson']:.3f} testP={d['mean_test_pearson']:.3f} "
                  f"testS={d['mean_test_spearman']:.3f}  {{lr:{d['lr']}, wd:{d['weight_decay']}, "
                  f"bs:{d['batch_size']}, nh:{d['num_hidden']}, ilw:{d['ilddt_weight']}, "
                  f"loss:{d['loss']}, drop:{d['dropout_scale']}}}")
        return

    if args.mode == 'final':
        cfg = {**BASE_CFG, **(json.load(open(args.config_json)) if args.config_json else {})}
        print("FINAL config:", {k: cfg[k] for k in ['lr','weight_decay','batch_size','num_hidden',
              'ilddt_weight','loss','dropout_scale']}, flush=True)
        print("\n--- baseline (paper HPs, AdamW, no refit/ensemble/calibration) ---", flush=True)
        base = run_cv(BASE_CFG, folds, Xf, Xr, y, il, ids, cmap, device, seeds=1, refit=False, calibrate=False)
        print("\n--- tuned + refit + %d-seed ensemble + calibration ---" % args.seeds, flush=True)
        tuned = run_cv(cfg, folds, Xf, Xr, y, il, ids, cmap, device, seeds=args.seeds, refit=True, calibrate=True)
        summary = dict(baseline={k: base[k] for k in base if k != 'rows'},
                       tuned={k: tuned[k] for k in tuned if k != 'rows'},
                       tuned_rows=tuned['rows'])
        json.dump(summary, open(args.out, 'w'), indent=2)
        def show(tag, r):
            print(f"\n{tag}")
            print(f"  mean-of-folds : Pearson {r['mean_pearson']:.4f}   Spearman {r['mean_spearman']:.4f}")
            print(f"  pooled        : Pearson {r['pooled_pearson']:.4f}   Spearman {r['pooled_spearman']:.4f}")
            print(f"  pooled+calib  : Pearson {r['pooled_pearson_cal']:.4f}   Spearman {r['pooled_spearman_cal']:.4f}")
        show("BASELINE:", base); show("TUNED:", tuned)
        return


    if args.mode == 'diag':
        TUNED = dict(lr=2e-4, weight_decay=0.01, batch_size=64, num_hidden=2, loss='huber', dropout_scale=0.7)
        conds = [
            ('A paper+std+ilw0.2   (my baseline)', {**BASE_CFG, 'standardize': True,  'ilddt_weight': 0.2}),
            ('B paper+NOstd+ilw0   (~notebook)  ', {**BASE_CFG, 'standardize': False, 'ilddt_weight': 0.0}),
            ('C paper+NOstd+ilw0.2             ', {**BASE_CFG, 'standardize': False, 'ilddt_weight': 0.2}),
            ('D tuned+NOstd+ilw0   (reg gains) ', {**BASE_CFG, **TUNED, 'standardize': False, 'ilddt_weight': 0.0}),
            ('E tuned+NOstd+ilw0.2             ', {**BASE_CFG, **TUNED, 'standardize': False, 'ilddt_weight': 0.2}),
        ]
        allres = {}
        for name, cfg in conds:
            print(f"\n########## {name.strip()} ##########", flush=True)
            r = run_cv(cfg, folds, Xf, Xr, y, il, ids, cmap, device, seeds=1, refit=False, calibrate=True)
            allres[name] = {k: r[k] for k in r if k != 'rows'}
            print(f"  mean-of-folds  P={r['mean_pearson']:.4f}  S={r['mean_spearman']:.4f}", flush=True)
            print(f"  pooled         P={r['pooled_pearson']:.4f}  S={r['pooled_spearman']:.4f}", flush=True)
            print(f"  pooled+calib   P={r['pooled_pearson_cal']:.4f}  S={r['pooled_spearman_cal']:.4f}", flush=True)
        json.dump(allres, open(args.out, 'w'), indent=2, default=float)
        print("\n================ SUMMARY (mean-of-folds / pooled) ================", flush=True)
        print(f"{'condition':38s}  meanP  meanS  poolP  poolS  poolP_cal poolS_cal", flush=True)
        for name in allres:
            a = allres[name]
            print(f"{name:38s}  {a['mean_pearson']:.3f}  {a['mean_spearman']:.3f}  "
                  f"{a['pooled_pearson']:.3f}  {a['pooled_spearman']:.3f}  "
                  f"{a['pooled_pearson_cal']:.3f}    {a['pooled_spearman_cal']:.3f}", flush=True)
        return

if __name__ == '__main__':
    main()
