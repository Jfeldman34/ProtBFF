#!/usr/bin/env python
"""Loss-function sweep for ProtBFF (dual readout, honest MVA-60, clean protocol).

Same architecture/protocol as the headline dual model; only the training objective
changes. Motivation:
  - the eval metric is correlation, not MSE -> test a direct correlation loss and a
    pairwise ranking (Spearman-surrogate) loss;
  - the ilDDT aux head was structurally dead under the ANTISYMMETRIC readout (a
    symmetric target through an antisymmetric map); the dual readout has a symmetric
    head, so ilDDT may finally be learnable -> re-test the aux loss here;
  - robustness to SKEMPI ddG noise -> Huber / MAE.
Reports mean-of-folds and pooled Pearson & Spearman, 5-seed ensembled, vs the MSE baseline.
"""
import argparse, json, os, sys
import numpy as np, torch, torch.nn as nn
from scipy.stats import pearsonr, spearmanr
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from protbff_train import load_cache, load_clusters, load_folds
from protbff_arch import FullModel, make_loader, predict, cluster_aware_val, BASE


def pearson_loss(p, y):
    p = p - p.mean(); y = y - y.mean()
    return 1.0 - (p * y).sum() / (p.norm() * y.norm() + 1e-8)


def rank_loss(p, y, margin=0.0):
    # pairwise margin ranking over all pairs in the batch (Spearman surrogate)
    dp = p.unsqueeze(0) - p.unsqueeze(1)          # (B,B) pred diffs
    dy = y.unsqueeze(0) - y.unsqueeze(1)
    mask = (dy.abs() > 1e-6)
    s = torch.sign(dy)
    l = torch.relu(margin - s * dp)
    return (l * mask).sum() / (mask.sum() + 1e-8)


def train_once(c, mode, lam, device, Xf, Xr, y, il, tr, val, te, seed):
    torch.manual_seed(seed); np.random.seed(seed)
    ym, ys_ = 0.0, 1.0                            # clean protocol: no target standardization
    bs = c['batch_size']
    tl = make_loader(Xf, Xr, y, il, tr, bs, True)
    vl = make_loader(Xf, Xr, y, il, val, bs, False)
    el = make_loader(Xf, Xr, y, il, te, bs, False)
    model = FullModel(c).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=c['lr'], weight_decay=c['weight_decay'])
    mse = nn.MSELoss(); huber = nn.SmoothL1Loss(); mae = nn.L1Loss()
    best, best_state, bad = -np.inf, None, 0
    for ep in range(1, c['epochs'] + 1):
        model.train()
        for xf, xr, yy, ii in tl:
            xf, xr, yy, ii = xf.to(device), xr.to(device), yy.to(device), ii.to(device)
            pd_, pi_ = model(xf, xr)
            if mode == 'mse':       loss = mse(pd_, yy)
            elif mode == 'huber':   loss = huber(pd_, yy)
            elif mode == 'mae':     loss = mae(pd_, yy)
            elif mode == 'pearson': loss = mse(pd_, yy) + lam * pearson_loss(pd_, yy)
            elif mode == 'corr_only': loss = pearson_loss(pd_, yy)
            elif mode == 'rank':    loss = mse(pd_, yy) + lam * rank_loss(pd_, yy)
            elif mode == 'ilddt2':  loss = mse(pd_, yy) + 0.2 * mse(pi_, ii)
            else: raise ValueError(mode)
            opt.zero_grad(); loss.backward(); opt.step()
        vp = predict(model, vl, device, ym, ys_); vt = y[val]
        sc = pearsonr(vt, vp)[0] if (vt.std() > 1e-8 and vp.std() > 1e-8) else -np.inf
        if not np.isfinite(sc): sc = -np.inf
        if sc > best: best, bad, best_state = sc, 0, {k: v.cpu().clone() for k, v in model.state_dict().items()}
        else:
            bad += 1
            if bad >= c.get('patience', 15): break
    if best_state is not None: model.load_state_dict(best_state)
    return predict(model, el, device, ym, ys_)


def run(c, mode, lam, folds, Xf, Xr, y, il, ids, cmap, device, seeds, override=None):
    c = {**c, **(override or {})}
    fr, fs, pt, pp = [], [], [], []
    for fi, (tr_all, te) in enumerate(folds):
        tr, val = cluster_aware_val(tr_all, ids, cmap, 0.10, 42 + fi)
        seed_preds = []
        for s in range(seeds):
            seed_preds.append(train_once(c, mode, lam, device, Xf, Xr, y, il, tr, val, te,
                                         seed=c['random_state'] + fi + 1000 * s))
        p = np.mean(seed_preds, axis=0)
        fr.append(pearsonr(y[te], p)[0]); fs.append(spearmanr(y[te], p)[0])
        pt += list(y[te]); pp += list(p)
    return dict(mean_r=float(np.mean(fr)), sd_r=float(np.std(fr)),
                mean_s=float(np.mean(fs)), sd_s=float(np.std(fs)),
                pooled_r=float(pearsonr(pt, pp)[0]), pooled_s=float(spearmanr(pt, pp)[0]),
                fold_r=[float(x) for x in fr], fold_s=[float(x) for x in fs])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--cache', required=True)
    ap.add_argument('--folds_dir', required=True)
    ap.add_argument('--clusters', required=True)
    ap.add_argument('--embed_dim', type=int, default=768)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seeds', type=int, default=5)
    args = ap.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Xf, Xr, y, il, ids = load_cache(args.cache)
    cmap = load_clusters(args.clusters); folds = load_folds(args.folds_dir, ids)
    c = {**BASE, 'readout': 'dual', 'embed_dim': args.embed_dim, 'num_scores': 5}
    print(f"device={device} N={len(y)} dim={args.embed_dim} seeds={args.seeds} (dual readout)", flush=True)

    # (key, mode, lam, config-override)
    variants = [('mse', 'mse', 0.0, {}), ('huber', 'huber', 0.0, {}), ('mae', 'mae', 0.0, {}),
                ('pearson_0.5', 'pearson', 0.5, {}), ('pearson_1.0', 'pearson', 1.0, {}),
                ('corr_only', 'corr_only', 0.0, {}), ('rank_0.5', 'rank', 0.5, {}),
                ('ilddt2_antisymhead', 'ilddt2', 0.0, {}),
                ('ilddt2_dualhead', 'ilddt2', 0.0, {'ilddt_readout': 'dual'})]
    res = {}
    base_r = None
    for key, mode, lam, override in variants:
        r = run(c, mode, lam, folds, Xf, Xr, y, il, ids, cmap, device, args.seeds, override)
        res[key] = r
        if base_r is None: base_r = r['mean_r']
        print(f"[{key:11s}] meanP={r['mean_r']:.4f}±{r['sd_r']:.3f} meanS={r['mean_s']:.4f} "
              f"poolP={r['pooled_r']:.4f} poolS={r['pooled_s']:.4f}  (ΔmeanP {r['mean_r']-base_r:+.4f})", flush=True)
        json.dump(res, open(args.out, 'w'), indent=2, default=float)

    print("\n=== loss sweep vs MSE baseline (dual readout, MVA-60) ===", flush=True)
    for k, r in res.items():
        print(f"  {k:11s} meanP {r['mean_r']:.4f} (Δ {r['mean_r']-res['mse']['mean_r']:+.4f})  "
              f"meanS {r['mean_s']:.4f} (Δ {r['mean_s']-res['mse']['mean_s']:+.4f})", flush=True)


if __name__ == '__main__':
    main()
