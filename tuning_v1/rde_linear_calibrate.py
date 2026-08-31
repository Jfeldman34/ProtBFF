#!/usr/bin/env python
"""RDE-Linear on MVA-60: linear calibration of RDE entropy features per fold.

Loads the entropy.pkl produced by rde.linear.entropy (frozen RDE / SIM.pt over
SKEMPI structures), builds per-mutation entropy-change features (the RDE
unsupervised signal is the binding-entropy change ΔΔH = (H_b−H_ub)_mt −
(H_b−H_ub)_wt, summed over mutated + interface residues), aligns to the master
by (pdbcode, mutstr, ddG), and fits Ridge per MVA fold (alpha val-selected).
"""
import pickle, argparse, numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge
from sklearn.metrics import roc_auc_score


def agg(group):
    """aggregate a list of residue entropy dicts -> feature list."""
    if not group:
        return [0.0] * 6
    Hbwt = np.array([g.get('H_b_wt', 0.0) for g in group], float)
    Hbmt = np.array([g.get('H_b_mt', 0.0) for g in group], float)
    Huwt = np.array([g.get('H_ub_wt', g.get('H_ub', 0.0)) for g in group], float)
    Humt = np.array([g.get('H_ub_mt', g.get('H_ub', 0.0)) for g in group], float)
    ddH = (Hbmt - Humt) - (Hbwt - Huwt)     # binding-entropy change
    dHb = Hbmt - Hbwt                          # bound entropy change
    return [ddH.sum(), ddH.mean(), dHb.sum(), dHb.mean(),
            (Hbmt - Hbwt).sum(), (Humt - Huwt).sum()]


def features(e):
    return agg(e.get('mutations', [])) + agg(e.get('lignbrs', [])) + agg(e.get('receptors', []))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pkl', default='/n/netscratch/shakhnovich_lab/Lab/jwang/rde_linear_mva/entropy.pkl')
    ap.add_argument('--master', default='data/cross_validation_folds_mva/60_percent/folds_60pct.csv')
    ap.add_argument('--out', default='tuning_v1/out/rde_linear_mva.json')
    args = ap.parse_args()

    res = pickle.load(open(args.pkl, 'rb'))
    rows = []
    for e in res.values():
        rows.append(dict(pdbcode=str(e['complex']).split('_')[0], mutstr=str(e['mutstr']),
                         ddG=float(e['ddG']), feats=features(e)))
    df = pd.DataFrame(rows)
    df['key'] = list(zip(df.pdbcode, df.mutstr, df.ddG.round(3)))
    lut = {k: v for k, v in zip(df.key, df.feats)}

    m = pd.read_csv(args.master)
    mkey = list(zip(m['#Pdb_origin'].astype(str), m['Mutation(s)_cleaned'].astype(str), m.ddG.round(3)))
    X = np.array([lut.get(k, None) for k in mkey], dtype=object)
    ok = np.array([x is not None for x in X])
    print(f"master {len(m)}  RDE-Linear feats matched {ok.sum()}/{len(m)}")
    X = np.array([x for x in X[ok]], float)
    X = np.nan_to_num(X)
    y = m.ddG.values[ok].astype(float); fold = m.fold.values[ok].astype(int)

    fr, fs, fa, pt, pp = [], [], [], [], []
    for f in np.unique(fold):
        te = fold == f; tr = ~te
        # val = random 10% of train for alpha selection
        rng = np.random.default_rng(42 + f); idx = np.where(tr)[0]; rng.shuffle(idx)
        val = idx[:len(idx) // 10]; trn = idx[len(idx) // 10:]
        best_a, best_v = 1.0, -np.inf
        for a in (0.1, 1, 10, 100, 1000):
            mo = Ridge(alpha=a).fit(X[trn], y[trn]); vp = mo.predict(X[val])
            v = pearsonr(y[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
            if v > best_v: best_v, best_a = v, a
        mo = Ridge(alpha=best_a).fit(X[tr], y[tr]); p = mo.predict(X[te])
        fr.append(pearsonr(y[te], p)[0]); fs.append(spearmanr(y[te], p)[0])
        lab = (y[te] > 0).astype(int)
        fa.append(roc_auc_score(lab, p) if len(np.unique(lab)) > 1 else np.nan)
        pt += list(y[te]); pp += list(p)
    out = dict(mean_r=float(np.nanmean(fr)), mean_s=float(np.nanmean(fs)),
               mean_auroc=float(np.nanmean(fa)), pooled_r=float(pearsonr(pt, pp)[0]),
               pooled_s=float(spearmanr(pt, pp)[0]), n=int(ok.sum()))
    import json; json.dump(out, open(args.out, 'w'), indent=2)
    print(f"RDE-Linear MVA-60: mean-of-folds P={out['mean_r']:.4f} S={out['mean_s']:.4f} "
          f"AUROC={out['mean_auroc']:.4f} | pooled P={out['pooled_r']:.4f}")
    print(f"(paper leaky RDE-Linear = 0.369)  -> {args.out}")


if __name__ == '__main__':
    main()
