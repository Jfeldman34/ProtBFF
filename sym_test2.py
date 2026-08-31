import numpy as np, os, time
from sklearn.linear_model import Ridge
from scipy.stats import pearsonr
exec(open('sym_test.py').read().split('t0=time.time()')[0])

def run(X,t,alpha):
    rs=[]; pooled_t=[]; pooled_p=[]
    for tr,te in folds:
        m=Ridge(alpha=alpha).fit(X[tr],t[tr]); p=m.predict(X[te])
        rs.append(pearsonr(t[te],p)[0]); pooled_t.extend(t[te]); pooled_p.extend(p)
    return np.mean(rs), pearsonr(pooled_t,pooled_p)[0]

for alpha in [100,1000,10000,100000]:
    row=[]
    for nm,X in [('D',D),('S',S)]:
        m,p=run(X,y,alpha); row.append(f'{nm}: mean {m:.3f} pooled {p:.3f}')
    print(f'alpha={alpha:<7g}  ' + '   |   '.join(row), flush=True)

C=np.hstack([D,S])
for alpha in [1000,10000]:
    m,p=run(C,y,alpha); print(f'[D|S] alpha={alpha:<7g} ddG mean {m:.3f} pooled {p:.3f}', flush=True)
