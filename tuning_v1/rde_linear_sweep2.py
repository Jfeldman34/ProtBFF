#!/usr/bin/env python
"""RDE-Linear swept across MVA (and CD-HIT) identity thresholds, using the SAME
frozen RDE entropy features as the canonical MVA-60 calibration (rde_linear_calibrate.py).
RDE's entropy features are split-independent, so the only thing that changes per threshold
is the fold assignment; the ridge head is refit per fold (alpha val-selected). This is the
"trained competitor we can retrain at each step" -- seconds per threshold, no GPU.

SANITY GATE: MVA-60 must reproduce the canonical 0.148 mean-of-folds Pearson.
"""
import pickle, json, os, glob
import numpy as np, pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.linear_model import Ridge

PKL = "/n/netscratch/shakhnovich_lab/Lab/jwang/rde_linear_mva/entropy.pkl"
OUT = "tuning_v1/out/rde_sweep/rde_linear_sweep2.json"


def agg(group):                                   # canonical: entropy-CHANGE features
    if not group:
        return [0.0] * 6
    Hbwt = np.array([g.get('H_b_wt', 0.0) for g in group], float)
    Hbmt = np.array([g.get('H_b_mt', 0.0) for g in group], float)
    Huwt = np.array([g.get('H_ub_wt', g.get('H_ub', 0.0)) for g in group], float)
    Humt = np.array([g.get('H_ub_mt', g.get('H_ub', 0.0)) for g in group], float)
    ddH = (Hbmt - Humt) - (Hbwt - Huwt)
    dHb = Hbmt - Hbwt
    return [ddH.sum(), ddH.mean(), dHb.sum(), dHb.mean(),
            (Hbmt - Hbwt).sum(), (Humt - Huwt).sum()]


def features(e):
    return agg(e.get('mutations', [])) + agg(e.get('lignbrs', [])) + agg(e.get('receptors', []))


def build_lut():
    res = pickle.load(open(PKL, 'rb'))
    rows = [dict(pdbcode=str(e['complex']).split('_')[0], mutstr=str(e['mutstr']),
                 ddG=float(e['ddG']), feats=features(e)) for e in res.values()]
    df = pd.DataFrame(rows)
    df['key'] = list(zip(df.pdbcode, df.mutstr, df.ddG.round(3)))
    return {k: v for k, v in zip(df.key, df.feats)}


def eval_folds(X, y, fold):
    fr, fs = [], []
    for f in np.unique(fold):
        te = fold == f; tr = ~te
        if te.sum() == 0 or tr.sum() == 0:
            continue
        rng = np.random.default_rng(42 + int(f)); idx = np.where(tr)[0]; rng.shuffle(idx)
        val = idx[:len(idx) // 10]; trn = idx[len(idx) // 10:]
        best_a, best_v = 1.0, -np.inf
        for a in (0.1, 1, 10, 100, 1000):
            mo = Ridge(alpha=a).fit(X[trn], y[trn]); vp = mo.predict(X[val])
            v = pearsonr(y[val], vp)[0] if np.std(vp) > 1e-9 else -np.inf
            if v > best_v: best_v, best_a = v, a
        mo = Ridge(alpha=best_a).fit(X[tr], y[tr]); p = mo.predict(X[te])
        fr.append(pearsonr(y[te], p)[0]); fs.append(spearmanr(y[te], p)[0])
    return dict(mean_r=float(np.mean(fr)), mean_s=float(np.mean(fs)),
                sem_r=float(np.std(fr) / len(fr) ** 0.5), n_folds=len(fr))


def align(lut, m):
    mkey = list(zip(m['#Pdb_origin'].astype(str), m['Mutation(s)_cleaned'].astype(str), m.ddG.round(3)))
    Xo = [lut.get(k, None) for k in mkey]
    ok = np.array([x is not None for x in Xo])
    X = np.nan_to_num(np.array([x for x, o in zip(Xo, ok) if o], float))
    return X, ok


def cdhit_fold_col(m, thr):
    """derive a fold column for CD-HIT from fold_k/test_complex_ids.txt (complex = #Pdb)."""
    root = f"data/cross_validation_folds_final/{thr}_percent"
    comp2fold = {}
    for d in sorted(glob.glob(f"{root}/fold_*")):
        k = int(os.path.basename(d).split('_')[1])
        tf = os.path.join(d, "test_complex_ids.txt")
        if not os.path.exists(tf):
            return None
        for line in open(tf):
            c = line.strip()
            if c:
                comp2fold[c] = k
    return m['#Pdb'].astype(str).map(comp2fold).values if '#Pdb' in m else None


def main():
    lut = build_lut()
    out = {}
    # ---- MVA sweep ----
    for t in [90, 80, 60, 50, 40, 30]:
        m = pd.read_csv(f"data/cross_validation_folds_mva/{t}_percent/folds_{t}pct.csv")
        X, ok = align(lut, m)
        y = m.ddG.values[ok].astype(float); fold = m.fold.values[ok].astype(int)
        out[f"mva_{t}"] = eval_folds(X, y, fold)
        r = out[f"mva_{t}"]
        print(f"  mva {t}%: meanP={r['mean_r']:.3f}+-{r['sem_r']:.3f} (n={r['n_folds']}, rows={ok.sum()})", flush=True)
    # ---- CD-HIT sweep ----
    for t in [100, 95, 80, 60, 40]:
        m = pd.read_csv(f"data/cross_validation_folds_mva/60_percent/folds_60pct.csv")  # master rows
        fc = cdhit_fold_col(m, t)
        if fc is None:
            print(f"  cdhit {t}%: no fold dirs, skip"); continue
        m = m.assign(fold=fc).dropna(subset=['fold'])
        X, ok = align(lut, m)
        y = m.ddG.values[ok].astype(float); fold = m.fold.values[ok].astype(int)
        out[f"cdhit_{t}"] = eval_folds(X, y, fold)
        r = out[f"cdhit_{t}"]
        print(f"  cdhit {t}%: meanP={r['mean_r']:.3f}+-{r['sem_r']:.3f} (n={r['n_folds']}, rows={ok.sum()})", flush=True)
    os.makedirs(os.path.dirname(OUT), exist_ok=True)
    json.dump(out, open(OUT, "w"), indent=2)
    g = out["mva_60"]["mean_r"]
    print(f"\nSANITY mva-60 (canonical 0.148) -> {g:.3f}  {'OK' if abs(g-0.148)<0.02 else 'MISMATCH'}")
    print("wrote", OUT)


if __name__ == "__main__":
    main()
