import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import shap
rng=np.random.RandomState(1)
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values
def xgb(): return XGBClassifier(n_estimators=150,max_depth=3,learning_rate=0.08,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
def esi_once(X,y,R,k=10,seed0=0):
    pp=X.shape[1]; Phi=np.zeros((R,pp))
    for r in range(R):
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed0+r); oof=np.zeros((len(y),pp))
        for tr,te in skf.split(X,y):
            m=xgb().fit(X[tr],y[tr]); oof[te]=np.abs(shap.TreeExplainer(m).shap_values(X[te]))
        Phi[r]=oof.mean(0)
    pb=Phi.mean(0); sd=Phi.std(0); cv=np.divide(sd,pb,out=np.zeros_like(sd),where=pb>0); m=1/(1+cv)
    Sm=np.sum(pb*m)/np.sum(pb); t=[kendalltau(Phi[a],Phi[b]).correlation for a in range(R) for b in range(a+1,R)]; Sr=(1+np.nanmean(t))/2
    tops=[set(np.argsort(Phi[r])[::-1][:k]) for r in range(R)]; ki=[(len(tops[a]&tops[b])-k*k/pp)/(k-k*k/pp) for a in range(R) for b in range(a+1,R)]; Ss=max(0,np.mean(ki))
    return (max(Sm,1e-9)**(1/3))*(max(Sr,1e-9)**(1/3))*(max(Ss,1e-9)**(1/3))
Phi=np.load("data/Phi_wdbc.npy")  # observed already computed; recompute base quickly from it
pb=Phi.mean(0); sd=Phi.std(0); cv=np.divide(sd,pb,out=np.zeros_like(sd),where=pb>0); m=1/(1+cv)
Sm=np.sum(pb*m)/np.sum(pb); t=[kendalltau(Phi[a],Phi[b]).correlation for a in range(Phi.shape[0]) for b in range(a+1,Phi.shape[0])]; Sr=(1+np.nanmean(t))/2
k=10;pp=Phi.shape[1];tops=[set(np.argsort(Phi[r])[::-1][:k]) for r in range(Phi.shape[0])];ki=[(len(tops[a]&tops[b])-k*k/pp)/(k-k*k/pp) for a in range(Phi.shape[0]) for b in range(a+1,Phi.shape[0])];Ss=max(0,np.mean(ki))
base=(Sm**(1/3))*(Sr**(1/3))*(Ss**(1/3))
B=49; nulls=[esi_once(X,rng.permutation(y),2,seed0=300+3*b) for b in range(B)]
nulls=np.array(nulls); E0=nulls.mean(); pval=(np.sum(nulls>=base)+1)/(B+1); rel=(base-E0)/(1-E0)
print(f"base ESI={base:.3f}  B={B}  E[null]={E0:.3f}  max null={nulls.max():.3f}  ESI_rel={rel:.3f}  perm p={pval:.4f}")
