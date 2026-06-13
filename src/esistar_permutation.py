import warnings, numpy as np, pandas as pd, time
warnings.filterwarnings("ignore")
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import shap
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values; p=X.shape[1]; k=10; K=3
def xgb(): return XGBClassifier(n_estimators=80,max_depth=3,learning_rate=0.12,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
def axes(X,y,R,seed0=0):
    Phi=np.zeros((R,p))
    for r in range(R):
        skf=StratifiedKFold(n_splits=K,shuffle=True,random_state=seed0+r); oof=np.zeros((len(y),p))
        for tr,te in skf.split(X,y):
            m=xgb().fit(X[tr],y[tr]); oof[te]=np.abs(shap.TreeExplainer(m).shap_values(X[te]))
        Phi[r]=oof.mean(0)
    t=[kendalltau(Phi[a],Phi[b]).correlation for a in range(R) for b in range(a+1,R)]; Sr=np.nanmean(t)
    tops=[set(np.argsort(Phi[r])[::-1][:k]) for r in range(R)]
    ki=[(len(tops[a]&tops[b])-k*k/p)/(k-k*k/p) for a in range(R) for b in range(a+1,R)]; Ss=np.mean(ki)
    return Sr,Ss
t0=time.time(); Sr,Ss=axes(X,y,8)
B=299; nr=np.zeros(B); ns=np.zeros(B)
for b in range(B):
    a=axes(X,np.random.RandomState(2000+b).permutation(y),2,seed0=3*b); nr[b]=a[0]; ns[b]=a[1]
r0=nr.mean(); s0=ns.mean()
cr=np.clip((Sr-r0)/(1-r0),0,1); cs=np.clip((Ss-s0)/(1-s0),0,1); star=(max(cr,1e-9)*max(cs,1e-9))**0.5
ncr=np.clip((nr-r0)/(1-r0),0,1); ncs=np.clip((ns-s0)/(1-s0),0,1); nstar=np.sqrt(np.clip(ncr,1e-9,None)*np.clip(ncs,1e-9,None))
pval=(np.sum(nstar>=star)+1)/(B+1)
print(f"ESI*={star:.3f} (c_rank={cr:.3f} c_sel={cs:.3f}) | #null>=obs={int(np.sum(nstar>=star))}/{B}  p={pval:.4f}  maxnull={nstar.max():.3f}  [{time.time()-t0:.0f}s]")
