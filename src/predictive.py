import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import brier_score_loss
from xgboost import XGBClassifier
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values; p=X.shape[1]
Phi=np.load("data/Phi_wdbc.npy"); pb=Phi.mean(0); sd=Phi.std(0); cv=np.divide(sd,pb,out=np.zeros_like(sd),where=pb>0); mstab=1/(1+cv)
def xgb(): return XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.08,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
# importance-matched stable vs unstable (pair consecutive-importance features)
top=np.argsort(pb)[::-1][:16]; stable=[]; unstable=[]
for i in range(0,16,2):
    a,b=top[i],top[i+1]
    (stable if mstab[a]>=mstab[b] else unstable).append(a); (unstable if mstab[a]>=mstab[b] else stable).append(b)
stable=np.array(stable); unstable=np.array(unstable)
def ece(yt,pr,bins=10):
    e=0
    for k in range(bins):
        lo,hi=k/bins,(k+1)/bins; m=(pr>lo)&(pr<=hi)
        if m.sum()>0: e+=m.mean()*abs(yt[m].mean()-pr[m].mean())
    return e
cvs=StratifiedKFold(5,shuffle=True,random_state=1)
print(f"Matched sets: stable sum|SHAP|={pb[stable].sum():.2f}(stab {mstab[stable].mean():.3f}) | unstable sum|SHAP|={pb[unstable].sum():.2f}(stab {mstab[unstable].mean():.3f})")
print("\nCalibration (lower=better):  set        | Brier  | ECE")
for nm,cols in [("stable-8",stable),("unstable-8",unstable),("full-30",np.arange(p))]:
    br=[];ec=[]
    for seed in range(5):
        cvs=StratifiedKFold(5,shuffle=True,random_state=seed)
        pr=cross_val_predict(xgb(),X[:,cols],y,cv=cvs,method="predict_proba")[:,1]
        br.append(brier_score_loss(y,pr)); ec.append(ece(y,pr))
    print(f"   {nm:10s} | {np.mean(br):.4f} | {np.mean(ec):.4f}")
