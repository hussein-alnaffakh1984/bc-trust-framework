import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values; p=X.shape[1]
Phi=np.load("data/Phi_wdbc.npy"); pb=Phi.mean(0); sd=Phi.std(0); cv=np.divide(sd,pb,out=np.zeros_like(sd),where=pb>0); mstab=1/(1+cv)
def xgb(): return XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.08,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
levels=[0,0.5,1.0,1.5,2.0]

# ---- C: importance-MATCHED stable vs unstable (pair consecutive-importance features) ----
top=np.argsort(pb)[::-1][:16]
stable=[]; unstable=[]
for i in range(0,16,2):
    a,b=top[i],top[i+1]
    if mstab[a]>=mstab[b]: stable.append(a); unstable.append(b)
    else: stable.append(b); unstable.append(a)
stable=np.array(stable); unstable=np.array(unstable)
print(f"Importance-matched sets: stable sum|SHAP|={pb[stable].sum():.2f} (stab {mstab[stable].mean():.3f}) | unstable sum|SHAP|={pb[unstable].sum():.2f} (stab {mstab[unstable].mean():.3f})")
def run(cols,reps=15):
    out={L:[] for L in levels}; fstd=X[:,cols].std(0)
    for s in range(reps):
        Xtr,Xte,ytr,yte=train_test_split(X[:,cols],y,test_size=0.3,stratify=y,random_state=s); m=xgb().fit(Xtr,ytr)
        for L in levels:
            Xn=Xte+np.random.RandomState(s*9+int(L*10)).normal(0,L*fstd,Xte.shape) if L>0 else Xte
            out[L].append(roc_auc_score(yte,m.predict_proba(Xn)[:,1]))
    return {L:np.mean(v) for L,v in out.items()}
rs=run(stable); ru=run(unstable)
print("AUC under shift (no retrain):  shift |  stable | unstable")
for L in levels: print(f"   {L:>4.1f}  |  {rs[L]:.3f}  |  {ru[L]:.3f}")
print(f"AUC drop 0->2σ:  stable={rs[0]-rs[2.0]:.3f}   unstable={ru[0]-ru[2.0]:.3f}")

# ---- Actionability: does the reject option DETECT shift? (deferral/coverage vs shift) ----
def ens(): return VotingClassifier([("xgb",xgb()),("lgb",LGBMClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=0,n_jobs=1,verbose=-1)),("svm",make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0))),("lr",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)))],voting="soft")
print("\nReject option under shift (train+calibrate clean, test shifted):  shift | coverage | deferral | sel.acc")
fstd=X.std(0)
res={L:{"cov":[],"def":[],"sel":[]} for L in levels}
for s in range(15):
    Xtr,Xt,ytr,yt=train_test_split(X,y,test_size=0.45,stratify=y,random_state=s)
    Xc,Xe,yc,ye=train_test_split(Xt,yt,test_size=0.55,stratify=yt,random_state=s)
    m=ens().fit(Xtr,ytr); pc=m.predict_proba(Xc); q={}
    for cl in [0,1]:
        sc=1-pc[yc==cl,cl]; nn=len(sc); q[cl]=np.quantile(sc,min(1,np.ceil((nn+1)*0.9)/nn))
    for L in levels:
        Xen=Xe+np.random.RandomState(s*7+int(L*10)).normal(0,L*fstd,Xe.shape) if L>0 else Xe
        pe=m.predict_proba(Xen); sets=[{cl for cl in [0,1] if (1-pe[i,cl])<=q[cl]} for i in range(len(ye))]
        res[L]["cov"].append(np.mean([ye[i] in sets[i] for i in range(len(ye))]))
        sng=[len(s_)==1 for s_ in sets]; res[L]["def"].append(1-np.mean(sng))
        acc=[list(sets[i])[0]==ye[i] for i in range(len(ye)) if sng[i]]; res[L]["sel"].append(np.mean(acc) if acc else np.nan)
for L in levels:
    print(f"   {L:>4.1f}  |  {np.mean(res[L]['cov']):.3f}  |  {np.mean(res[L]['def']):.3f}  |  {np.nanmean(res[L]['sel']):.3f}")
