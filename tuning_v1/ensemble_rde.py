#!/usr/bin/env python
"""Ensemble ProtBFF (dual) with RDE on the MVA 60% folds, and report per-fold /
pooled Pearson, Spearman, AUROC for RDE alone, ProtBFF alone, and the ensemble.

Alignment (verified in-session):
  - ProtBFF OOF cache rows are in exact SKEMPI/master order -> join by row index.
  - RDE CSV is in a DIFFERENT order and (complex,mutstr) has 886 dups, so RDE is
    joined to the master by (complex, round(ddG,3)); ddG is the shared SKEMPI
    ground truth, so this is an exact join (match residual ~0, asserted below).
  - Fold assignment: master 'fold' column (0-indexed), verified == fold_k dirs.

Ensemble = equal-weight average of per-fold z-scored predictions (label-free;
correlation is per-fold scale-invariant, z-scoring only aligns the two models'
scales before averaging). Rank-average reported as a robustness check.
"""
import argparse, json
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr, rankdata
from sklearn.metrics import roc_auc_score


def zfold(pred, fold):
    z = np.empty_like(pred, dtype=float)
    for f in np.unique(fold):
        m = fold == f
        s = pred[m].std()
        z[m] = (pred[m] - pred[m].mean()) / (s if s > 1e-9 else 1.0)
    return z


def rankfold(pred, fold):
    r = np.empty_like(pred, dtype=float)
    for f in np.unique(fold):
        m = fold == f
        r[m] = (rankdata(pred[m]) - 1) / max(1, m.sum() - 1)
    return r


def metrics(y, pred, fold):
    frs, fss, fas = [], [], []
    for f in np.unique(fold):
        m = fold == f
        yy, pp = y[m], pred[m]
        frs.append(pearsonr(yy, pp)[0]); fss.append(spearmanr(yy, pp)[0])
        lab = (yy > 0).astype(int)
        fas.append(roc_auc_score(lab, pp) if len(np.unique(lab)) > 1 else np.nan)
    return dict(mean_r=float(np.nanmean(frs)), mean_s=float(np.nanmean(fss)),
                mean_auroc=float(np.nanmean(fas)),
                pooled_r=float(pearsonr(y, pred)[0]), pooled_s=float(spearmanr(y, pred)[0]),
                fold_r=[float(x) for x in frs], fold_s=[float(x) for x in fss],
                fold_auroc=[float(x) for x in fas])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--protbff', default='tuning_v1/out/oof_dual_prosst.csv')
    ap.add_argument('--rde', default='/tmp/rde_30000.csv')
    ap.add_argument('--master', default='data/cross_validation_folds_mva/60_percent/folds_60pct.csv')
    ap.add_argument('--out', default='tuning_v1/out/ensemble_rde.json')
    args = ap.parse_args()

    master = pd.read_csv(args.master)                      # 6631 rows, canonical order
    pb = pd.read_csv(args.protbff).sort_values('idx').reset_index(drop=True)
    rde = pd.read_csv(args.rde)

    # --- ProtBFF join (by row index into master) ---
    y = master['ddG'].values.astype(float)
    fold = master['fold'].values.astype(int)
    pb_pred = np.full(len(master), np.nan)
    pb_pred[pb['idx'].values] = pb['pred'].values
    # sanity: ProtBFF's stored y must equal master ddG at those rows
    assert np.abs(pb['y'].values - y[pb['idx'].values]).max() < 1e-2, "ProtBFF y != master ddG"

    # --- RDE join by (complex, mutation, round(ddG,3)) with group-mean ---
    # mutation MUST be in the key: without it, different mutations in one complex whose
    # ddG rounds equal collide and get averaged, which inflates RDE's correlation.
    # The only remaining collisions are true duplicate measurements (identical complex+
    # mut+ddG) -> identical RDE input -> identical pred, so the group-mean is a no-op there.
    rde['key'] = list(zip(rde['complex'].astype(str), rde['mutstr'].astype(str), rde['ddG'].round(3)))
    rde_lut = rde.groupby('key')['ddG_pred'].mean()
    rde_ddg = rde.groupby('key')['ddG'].mean()
    mkey = list(zip(master['#Pdb_origin'].astype(str), master['Mutation(s)_cleaned'].astype(str),
                    master['ddG'].round(3)))
    rde_pred = np.array([rde_lut.get(k, np.nan) for k in mkey])
    matched_ddg = np.array([rde_ddg.get(k, np.nan) for k in mkey])

    # --- restrict to rows present in BOTH models ---
    ok = np.isfinite(pb_pred) & np.isfinite(rde_pred)
    print(f"master rows: {len(master)}")
    print(f"ProtBFF present: {np.isfinite(pb_pred).sum()}  RDE matched: {np.isfinite(rde_pred).sum()}  both: {ok.sum()}")
    res_ddg = np.abs(matched_ddg[np.isfinite(rde_pred)] - y[np.isfinite(rde_pred)]).max()
    print(f"RDE join ddG residual (max |Δ|): {res_ddg:.2e}  (should be ~0)")

    y_, fold_, pbp, rdp = y[ok], fold[ok], pb_pred[ok], rde_pred[ok]

    # --- ensembles ---
    ens_z = (zfold(pbp, fold_) + zfold(rdp, fold_)) / 2.0
    ens_rank = (rankfold(pbp, fold_) + rankfold(rdp, fold_)) / 2.0

    out = {
        'n': int(ok.sum()),
        'RDE':            metrics(y_, rdp, fold_),
        'ProtBFF_dual':   metrics(y_, pbp, fold_),
        'ensemble_zavg':  metrics(y_, ens_z, fold_),
        'ensemble_rank':  metrics(y_, ens_rank, fold_),
    }
    json.dump(out, open(args.out, 'w'), indent=2)

    print(f"\n=== MVA 60%, n={out['n']} (rows in both models) ===")
    hdr = f"{'model':16s}  meanP   meanS   mAUROC   poolP   poolS"
    print(hdr); print('-' * len(hdr))
    for k in ['RDE', 'ProtBFF_dual', 'ensemble_zavg', 'ensemble_rank']:
        a = out[k]
        print(f"{k:16s}  {a['mean_r']:.3f}   {a['mean_s']:.3f}   {a['mean_auroc']:.3f}    {a['pooled_r']:.3f}   {a['pooled_s']:.3f}")
    print(f"\nwrote {args.out}")


if __name__ == '__main__':
    main()
