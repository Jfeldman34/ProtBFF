#!/usr/bin/env python
"""Diagnostic: why is the ProtBFF-vs-bare gap smaller with the new bare?
For ACE2, compare 4 configs across training fractions:
  ProtBFF scaled + dual   | ProtBFF scaled + antisym
  bare unscaled + dual    | bare unscaled + antisym  (+ ridge[D|S])
If bare-antisym << bare-dual, the original's large gap came from an artificially weak
(antisymmetric) baseline, and the new small gap reflects a fair strong baseline.
"""
import sys, os, numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import dms_fewshot as DF

MERGED = '/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/ProSST_PPI-main/merged_scores'  # ace2 ProSST


def main():
    Xf, Xr, Uf, Ur, y = DF.build(MERGED)
    D = Uf.shape[1]; Uf32, Ur32 = Uf.astype(np.float32), Ur.astype(np.float32)
    print(f"ACE2/ProSST N={len(y)} D={D}", flush=True)
    print(f"{'frac':>5s} {'PBF-dual':>9s} {'PBF-anti':>9s} {'bare-dual':>9s} {'bare-anti':>9s} {'bare-ridge':>10s}")
    for frac in (10, 40, 80):
        from sklearn.model_selection import train_test_split
        r = {k: [] for k in ('pd', 'pa', 'bd', 'ba', 'br')}
        for s in range(2):
            tr, te = train_test_split(np.arange(len(y)), test_size=1 - frac / 100, random_state=42 + s)
            r['pd'].append(DF.train_model(Xf, Xr, y, tr, te, D, 5, 100 + s, readout='dual'))
            r['pa'].append(DF.train_model(Xf, Xr, y, tr, te, D, 5, 100 + s, readout='antisym'))
            r['bd'].append(DF.train_model(Uf32, Ur32, y, tr, te, D, 1, 100 + s, readout='dual'))
            r['ba'].append(DF.train_model(Uf32, Ur32, y, tr, te, D, 1, 100 + s, readout='antisym'))
            r['br'].append(DF.bare_ridge(Uf, Ur, y, tr, te, 100 + s))
        m = {k: np.mean(v) for k, v in r.items()}
        print(f"{frac:4d}% {m['pd']:9.3f} {m['pa']:9.3f} {m['bd']:9.3f} {m['ba']:9.3f} {m['br']:10.3f}", flush=True)


if __name__ == '__main__':
    main()
