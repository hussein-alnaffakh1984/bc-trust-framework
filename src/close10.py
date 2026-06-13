import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import kendalltau
from sklearn.model_selection import StratifiedKFold
from sklearn.inspection import permutation_importance
from xgboost import XGBClassifier
import shap
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values; feat=np.array(d.drop(columns=["diagnosis"]).columns); p=X.shape[1]
def xgb(): return XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.08,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
R=6;K=5
PhiS=np.zeros((R,p)); PhiG=np.zeros((R,p)); PhiP=np.zeros((R,p))
for r in range(R):
    skf=StratifiedKFold(n_splits=K,shuffle=True,random_state=r)
    oS=np.zeros((len(y),p)); gacc=np.zeros(p); pacc=np.zeros(p); nf=0
    for tr,te in skf.split(X,y):
        m=xgb().fit(X[tr],y[tr])
        oS[te]=np.abs(shap.TreeExplainer(m).shap_values(X[te]))
        gacc+=m.feature_importances_                      # gain
        pacc+=permutation_importance(m,X[te],y[te],n_repeats=3,random_state=0,n_jobs=1).importances_mean
        nf+=1
    PhiS[r]=oS.mean(0); PhiG[r]=gacc/nf; PhiP[r]=np.clip(pacc/nf,0,None)
def esi(Phi,k=10):
    R,pp=Phi.shape; pb=Phi.mean(0); sd=Phi.std(0); cv=np.divide(sd,pb,out=np.zeros_like(sd),where=pb>0); m=1/(1+cv)
    Sm=np.sum(pb*m)/np.sum(pb) if pb.sum()>0 else 0
    t=[kendalltau(Phi[a],Phi[b]).correlation for a in range(R) for b in range(a+1,R)]; Sr=(1+np.nanmean(t))/2
    tops=[set(np.argsort(Phi[r])[::-1][:k]) for r in range(R)]; ki=[(len(tops[a]&tops[b])-k*k/pp)/(k-k*k/pp) for a in range(R) for b in range(a+1,R)]; Ss=max(0,np.mean(ki))
    return (max(Sm,1e-9)**(1/3))*(max(Sr,1e-9)**(1/3))*(max(Ss,1e-9)**(1/3)), set(np.argsort(pb)[::-1][:5])
print("ESI by attribution method (WDBC, R=6):")
res={}
for nm,Phi in [("SHAP",PhiS),("XGBoost gain",PhiG),("Permutation",PhiP)]:
    e,top5=esi(Phi); res[nm]=(e,top5); print(f"  {nm:14s} ESI={e:.3f}  top5={sorted(feat[list(top5)])[:5]}")
# cross-method top-5 agreement
ms=list(res); 
print("Top-5 overlap (Jaccard) between methods:")
for i in range(len(ms)):
    for j in range(i+1,len(ms)):
        a,b=res[ms[i]][1],res[ms[j]][1]; jac=len(a&b)/len(a|b)
        print(f"  {ms[i]} vs {ms[j]}: {jac:.2f}")
