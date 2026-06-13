import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import kendalltau
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap

def xgb(): return XGBClassifier(n_estimators=150,max_depth=3,learning_rate=0.08,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
def ens():
    return VotingClassifier([("xgb",xgb()),
        ("lgb",LGBMClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=0,n_jobs=1,verbose=-1)),
        ("svm",make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0))),
        ("lr",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)))],voting="soft")
def ci95(v): v=np.asarray(v); return 1.96*v.std()/np.sqrt(len(v))

def perf(X,y):
    cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=4,random_state=1)
    a=cross_val_score(ens(),X,y,cv=cv,scoring="accuracy",n_jobs=2)
    au=cross_val_score(ens(),X,y,cv=cv,scoring="roc_auc",n_jobs=2)
    re=cross_val_score(ens(),X,y,cv=cv,scoring="recall",n_jobs=2)
    return (a.mean(),ci95(a)),(au.mean(),ci95(au)),(re.mean(),ci95(re))

def esi_once(X,y,R,k,seed0=0):
    p=X.shape[1]; Phi=np.zeros((R,p))
    for r in range(R):
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=seed0+r); oof=np.zeros((len(y),p))
        for tr,te in skf.split(X,y):
            m=xgb().fit(X[tr],y[tr]); oof[te]=np.abs(shap.TreeExplainer(m).shap_values(X[te]))
        Phi[r]=oof.mean(0)
    pb=Phi.mean(0); sd=Phi.std(0); cv=np.divide(sd,pb,out=np.zeros_like(sd),where=pb>0); m=1/(1+cv)
    Smag=np.sum(pb*m)/np.sum(pb)
    t=[kendalltau(Phi[a],Phi[b]).correlation for a in range(R) for b in range(a+1,R)]; Srank=(1+np.nanmean(t))/2
    tops=[set(np.argsort(Phi[r])[::-1][:k]) for r in range(R)]
    ki=[(len(tops[a]&tops[b])-k*k/p)/(k-k*k/p) for a in range(R) for b in range(a+1,R)]; Ssel=max(0,np.mean(ki))
    return (max(Smag,1e-9)**(1/3))*(max(Srank,1e-9)**(1/3))*(max(Ssel,1e-9)**(1/3))

def esi_full(X,y,k,R=8,B=6):
    obs=esi_once(X,y,R,k)
    nulls=[esi_once(X,np.random.RandomState(b).permutation(y),4,k,seed0=50+10*b) for b in range(B)]
    E0=np.mean(nulls); rel=(obs-E0)/(1-E0); p=(np.sum(np.array(nulls)>=obs)+1)/(B+1)
    return obs,rel,p

def xconformal(X,y,reps=25,alpha=0.10):
    covs=[]; defers=[]; selaccs=[]
    for s in range(reps):
        Xtr,Xt,ytr,yt=train_test_split(X,y,test_size=0.4,stratify=y,random_state=s)
        Xcal,Xte,ycal,yte=train_test_split(Xt,yt,test_size=0.5,stratify=yt,random_state=s)
        m=ens().fit(Xtr,ytr); pc=m.predict_proba(Xcal); pt=m.predict_proba(Xte); q={}
        for c in [0,1]:
            sc=1-pc[ycal==c,c]; n=len(sc); q[c]=np.quantile(sc,min(1,np.ceil((n+1)*(1-alpha))/n)) if n else 1
        sets=[{c for c in [0,1] if (1-pt[i,c])<=q[c]} for i in range(len(yte))]
        covs.append(np.mean([yte[i] in sets[i] for i in range(len(yte))]))
        sng=[len(s)==1 for s in sets]; defers.append(1-np.mean(sng))
        acc=[list(sets[i])[0]==yte[i] for i in range(len(yte)) if sng[i]]
        selaccs.append(np.mean(acc) if acc else np.nan)
    return (np.mean(covs),ci95(covs)),(np.mean(defers),ci95(defers)),(np.nanmean(selaccs),ci95(np.array(selaccs)[~np.isnan(selaccs)]))

def load():
    d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
    W=(d.drop(columns=["diagnosis"]).values,(d["diagnosis"].values=="M").astype(int),10)
    w=pd.read_csv("data/wbc_selva.csv").drop(columns=["Id"]); w["Bare.nuclei"]=pd.to_numeric(w["Bare.nuclei"],errors="coerce")
    Wb=(SimpleImputer(strategy="median").fit_transform(w.drop(columns=["Class"]).values),w["Class"].values.astype(int),5)
    c=pd.read_csv("data/coimbra.csv"); Co=(c.drop(columns=["Classification"]).values,(c["Classification"].values=="Yes").astype(int),5)
    mm=pd.read_csv("data/mammo.csv",na_values=["?"],names=["BIRADS","age","shape","margin","density","severity"]).drop(columns=["BIRADS"])
    Mm=(SimpleImputer(strategy="median").fit_transform(mm.drop(columns=["severity"]).values),mm["severity"].values.astype(int),3)
    return {"WDBC":W,"WBC":Wb,"Coimbra":Co,"Mammographic":Mm}

print(f"{'Cohort':12s} {'n':>4} | {'Acc[95%CI]':>18} {'AUC':>14} {'Recall':>14} | {'ESI':>5} {'ESI_rel':>7} {'p':>6} | {'Cov[CI]':>14} {'Defer[CI]':>14} {'SelAcc':>8}")
for nm,(X,y,k) in load().items():
    (a,ah),(au,auh),(rc,rch)=perf(X,y)
    e,erel,p=esi_full(X,y,k)
    (cov,covh),(dfr,dfrh),(sa,sah)=xconformal(X,y)
    print(f"{nm:12s} {len(y):>4} | {a:.3f}[{a-ah:.3f},{a+ah:.3f}] {au:.3f}+/-{auh:.3f} {rc:.3f}+/-{rch:.3f} | {e:.3f} {erel:.3f} {p:.3f} | {cov:.3f}+/-{covh:.3f} {dfr:.3f}+/-{dfrh:.3f} {sa:.3f}")

# proper permutation p-values (B=19 -> min p=0.05) and 4-cohort figure
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
def esi_p(X,y,k,R=8,B=19):
    obs=esi_once(X,y,R,k)
    nulls=[esi_once(X,np.random.RandomState(b).permutation(y),3,k,seed0=70+7*b) for b in range(B)]
    return obs,(np.sum(np.array(nulls)>=obs)+1)/(B+1)
print("\nProper permutation p-values (B=19):")
res={}
for nm,(X,y,k) in load().items():
    e,p=esi_p(X,y,k); res[nm]=e; print(f"  {nm:12s} ESI={e:.3f}  perm p={p:.3f}")

