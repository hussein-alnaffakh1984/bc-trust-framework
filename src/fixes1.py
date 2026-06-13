import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import kendalltau, wilcoxon
from sklearn.model_selection import StratifiedKFold, RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
rng=np.random.RandomState(0)
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values; p=X.shape[1]
def xgb(): return XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.08,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
def Phi_mat(X,y,R,K=5,seed0=0):
    pp=X.shape[1]; Phi=np.zeros((R,pp))
    for r in range(R):
        skf=StratifiedKFold(n_splits=K,shuffle=True,random_state=seed0+r); oof=np.zeros((len(y),pp))
        for tr,te in skf.split(X,y):
            m=xgb().fit(X[tr],y[tr]); oof[te]=np.abs(shap.TreeExplainer(m).shap_values(X[te]))
        Phi[r]=oof.mean(0)
    return Phi
def comps(Phi,k):
    R,pp=Phi.shape; pb=Phi.mean(0); sd=Phi.std(0); cv=np.divide(sd,pb,out=np.zeros_like(sd),where=pb>0); m=1/(1+cv)
    Smag=np.sum(pb*m)/np.sum(pb)
    t=[kendalltau(Phi[a],Phi[b]).correlation for a in range(R) for b in range(a+1,R)]; Srank=(1+np.nanmean(t))/2
    tops=[set(np.argsort(Phi[r])[::-1][:k]) for r in range(R)]
    ki=[(len(tops[a]&tops[b])-k*k/pp)/(k-k*k/pp) for a in range(R) for b in range(a+1,R)]; Ssel=max(0,np.mean(ki))
    return Smag,Srank,Ssel
def esi(Sm,Sr,Ss,w=(1/3,1/3,1/3)): return (max(Sm,1e-9)**w[0])*(max(Sr,1e-9)**w[1])*(max(Ss,1e-9)**w[2])
def nog(Phi,k):
    R,pp=Phi.shape; Z=np.array([[1 if j in set(np.argsort(Phi[r])[::-1][:k]) else 0 for j in range(pp)] for r in range(R)])
    pj=Z.mean(0); s2=(R/(R-1))*pj*(1-pj); kbar=Z.sum(1).mean(); return 1-(s2.mean())/((kbar/pp)*(1-kbar/pp))

Phi=Phi_mat(X,y,10); Sm,Sr,Ss=comps(Phi,10); base=esi(Sm,Sr,Ss)
np.save("data/Phi_wdbc.npy",Phi)
print(f"ESI(equal,k=10)={base:.3f} [S_mag={Sm:.3f} S_rank={Sr:.3f} S_sel={Ss:.3f}]")
g=[]
for w1 in np.arange(0.1,0.81,0.1):
    for w2 in np.arange(0.1,0.9-w1+1e-9,0.1):
        w3=1-w1-w2
        if w3>=0.099: g.append(esi(Sm,Sr,Ss,(w1,w2,w3)))
g=np.array(g); print(f"[Weights] {len(g)} vectors  ESI range [{g.min():.3f},{g.max():.3f}] spread={g.max()-g.min():.3f}")
print("[k] ", {k:round(esi(*comps(Phi,k)),3) for k in [5,8,10,12,15]})
nogv=nog(Phi,10)
be=[]; bn=[]
for b in range(300):
    idx=rng.choice(10,10,replace=True); Pb=Phi[idx]; be.append(esi(*comps(Pb,10))); bn.append(nog(Pb,10))
print(f"[ESI CI] {base:.3f} 95% CI [{np.percentile(be,2.5):.3f},{np.percentile(be,97.5):.3f}]")
print(f"[Nogueira] {nogv:.3f} 95% CI [{np.percentile(bn,2.5):.3f},{np.percentile(bn,97.5):.3f}]")
def mk():
    lg=LGBMClassifier(n_estimators=400,max_depth=3,learning_rate=0.05,random_state=0,n_jobs=1,verbose=-1)
    sv=make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0)); lr=make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000))
    return {"XGBoost":xgb(),"LightGBM":lg,"SVM":sv,"LogReg":lr,"Ensemble":VotingClassifier([("xgb",xgb()),("lgb",lg),("svm",sv),("lr",lr)],voting="soft")}
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=4,random_state=1)
sc={n:cross_validate(m,X,y,cv=cv,scoring="accuracy",n_jobs=2)["test_score"] for n,m in mk().items()}
pr=[]
for n in ["XGBoost","LightGBM","SVM","LogReg"]:
    W,pv=wilcoxon(sc["Ensemble"],sc[n]); pr.append((n,pv))
pr=sorted(pr,key=lambda z:z[1]); m=len(pr)
print("[Holm vs Ensemble]")
for i,(n,pv) in enumerate(pr):
    print(f"   vs {n:9s}: raw p={pv:.4f} Holm p={min(1,pv*(m-i)):.4f} diff={sc['Ensemble'].mean()-sc[n].mean():+.4f}")
