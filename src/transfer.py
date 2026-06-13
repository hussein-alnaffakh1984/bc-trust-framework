import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); Xdf=d.drop(columns=["diagnosis"]); X=Xdf.values; cols=list(Xdf.columns)
def ens(): return VotingClassifier([("xgb",XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)),("lgb",LGBMClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=0,n_jobs=1,verbose=-1)),("svm",make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0))),("lr",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)))],voting="soft")
print("TRUE model transfer across covariate-defined subpopulations of WDBC (train once, test on the OTHER domain, no retraining):\n")
for sf in ["texture_mean","smoothness_mean","fractal_dimension_mean","symmetry_mean"]:
    v=Xdf[sf].values; med=np.median(v); A=v<=med; B=~A
    if y[A].sum()<5 or y[B].sum()<5 or (1-y[A]).sum()<5 or (1-y[B]).sum()<5: 
        print(f"  [{sf}] skipped (class too small in a domain)"); continue
    # transfer A->B and B->A
    mA=ens().fit(X[A],y[A]); pB=mA.predict_proba(X[B])[:,1]
    mB=ens().fit(X[B],y[B]); pA=mB.predict_proba(X[A])[:,1]
    ab_acc=accuracy_score(y[B],(pB>=.5)); ab_auc=roc_auc_score(y[B],pB)
    ba_acc=accuracy_score(y[A],(pA>=.5)); ba_auc=roc_auc_score(y[A],pA)
    # reference: re-fit within target domain (framework-level)
    refB=cross_val_score(ens(),X[B],y[B],cv=StratifiedKFold(5,shuffle=True,random_state=0),scoring="roc_auc").mean()
    print(f"  split by {sf:22s} | A:n={A.sum()}({y[A].mean():.0%}M) B:n={B.sum()}({y[B].mean():.0%}M)")
    print(f"     transfer A->B: acc={ab_acc:.3f} AUC={ab_auc:.3f} | B->A: acc={ba_acc:.3f} AUC={ba_auc:.3f} | re-fit-in-B AUC(ref)={refB:.3f}")
