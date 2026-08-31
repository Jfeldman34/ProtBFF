import numpy as np, os, time
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr

P='/n/netscratch/shakhnovich_lab/Lab/jonathanfeldman/test_protbff/ProtBFF/model_benchmarking/score_caches/skempi_score_cache.npz'
FOLDS='/n/netscratch/shakhnovich_lab/Lab/jwang/ProtBFF/data/cross_validation_folds_mva/60_percent'
d=np.load(P, allow_pickle=True)
Xf=d['Xf'].astype(np.float32); Xr=d['Xr'].astype(np.float32)
y=d['y'].astype(np.float64); il=d['ilddt'].astype(np.float64)
ids=np.array([str(s).split('_',1)[1] if '_' in str(s) else str(s) for s in d['ids']])
D=Xf-Xr            # antisymmetric part (what the model's readout can express)
S=(Xf+Xr)/2        # symmetric part (what it cannot)
del Xf,Xr,d

def codes(p):
    s=set()
    for line in open(p):
        r=line.strip()
        if r and '_' in r: s.add(r.split('_',1)[1])
    return s

folds=[]
for k in range(1,11):
    fd=os.path.join(FOLDS,f'fold_{k}')
    tr_c=codes(os.path.join(fd,'train_complex_ids.txt')); te_c=codes(os.path.join(fd,'test_complex_ids.txt'))
    tr=np.array([i for i,c in enumerate(ids) if c in tr_c]); te=np.array([i for i,c in enumerate(ids) if c in te_c])
    folds.append((tr,te))

def run(X,t,alpha=1000.0):
    rs=[]
    for tr,te in folds:
        m=Ridge(alpha=alpha).fit(X[tr],t[tr])
        p=m.predict(X[te])
        rs.append(pearsonr(t[te],p)[0] if np.std(p)>1e-12 else 0.0)
    return np.array(rs)

t0=time.time()
for name,X in [('D = Xf - Xr  (antisymmetric)',D),('S = (Xf+Xr)/2 (symmetric)',S)]:
    for tname,t in [('ddG',y),('ilDDT',il)]:
        r=run(X,t)
        print(f'{tname:6s} from {name:30s}  mean r = {r.mean():+.3f}  (per-fold {np.round(r,2)})', flush=True)
print('elapsed %.0fs'%(time.time()-t0))
