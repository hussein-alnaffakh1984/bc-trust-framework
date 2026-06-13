import warnings, numpy as np, pandas as pd, sys
warnings.filterwarnings("ignore")
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedKFold
from sklearn.impute import SimpleImputer
from xgboost import XGBClassifier
import shap
def xgb(): return XGBClassifier(n_estimators=150,max_depth=3,learning_rate=0.08,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
def axes(X,y,R,k,seed0=0):
    pp=X.shape[1]; Phi=np.zeros((R,pp))
    for r in range(R):
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed0+r); oof=np.zeros((len(y),pp))
        for tr,te in skf.split(X,y):
            m=xgb().fit(X[tr],y[tr]); oof[te]=np.abs(shap.TreeExplainer(m).shap_values(X[te]))
        Phi[r]=oof.mean(0)
    pb=Phi.mean(0); sd=Phi.std(0); cv=np.divide(sd,pb,out=np.zeros_like(sd),where=pb>0); mm=1/(1+cv)
    Smag=np.sum(pb*mm)/np.sum(pb)
    t=[kendalltau(Phi[a],Phi[b]).correlation for a in range(R) for b in range(a+1,R)]
    Srank_raw=np.nanmean(t)                      # in [-1,1], 0 under null
    tops=[set(np.argsort(Phi[r])[::-1][:k]) for r in range(R)]
    ki=[(len(tops[a]&tops[b])-k*k/pp)/(k-k*k/pp) for a in range(R) for b in range(a+1,R)]; Ssel=np.mean(ki)  # 0 under null
    return Smag,Srank_raw,Ssel
def esistar(X,y,k,R=8,B=20,seedbase=0):
    Sm,Sr,Ss=axes(X,y,R,k)
    # null means per axis
    nm=[];nr=[];ns=[]
    for b in range(B):
        a=axes(X,np.random.RandomState(seedbase+b).permutation(y),2,k,seed0=400+5*b); nm.append(a[0]);nr.append(a[1]);ns.append(a[2])
    m0=np.mean(nm); r0=np.mean(nr); s0=np.mean(ns)
    cmag=np.clip((Sm-m0)/(1-m0),0,1); crank=np.clip((Sr-r0)/(1-r0),0,1); csel=np.clip((Ss-s0)/(1-s0),0,1)
    star=(max(cmag,1e-9)*max(crank,1e-9)*max(csel,1e-9))**(1/3)
    # raw ESI for comparison: map rank to [0,1], sel clip>=0
    rawSr=(1+Sr)/2; rawSs=max(0,Ss); raw=(max(Sm,1e-9)*max(rawSr,1e-9)*max(rawSs,1e-9))**(1/3)
    return dict(Smag=Sm,Srank=rawSr,Ssel=max(0,Ss),raw=raw,m0=m0,r0=(1+r0)/2,s0=max(0,s0),
                cmag=cmag,crank=crank,csel=csel,star=star)
def load():
    d=pd.read_csv("data/wdbc.csv");d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
    W=(d.drop(columns=["diagnosis"]).values,(d["diagnosis"].values=="M").astype(int),10)
    w=pd.read_csv("data/wbc_selva.csv").drop(columns=["Id"]);w["Bare.nuclei"]=pd.to_numeric(w["Bare.nuclei"],errors="coerce")
    Wb=(SimpleImputer(strategy="median").fit_transform(w.drop(columns=["Class"]).values),w["Class"].values.astype(int),5)
    return {"WDBC":W,"WBC":Wb}
for nm,(X,y,k) in load().items():
    r=esistar(X,y,k)
    print(f"\n=== {nm} ===")
    print(f"  raw ESI={r['raw']:.3f}  (null floor: mag={r['m0']:.2f} rank={r['r0']:.2f} sel={r['s0']:.2f})")
    print(f"  chance-corrected axes: cmag={r['cmag']:.3f} crank={r['crank']:.3f} csel={r['csel']:.3f}")
    print(f"  ESI* = {r['star']:.3f}   (null floor of ESI* ~ 0 by construction)")
