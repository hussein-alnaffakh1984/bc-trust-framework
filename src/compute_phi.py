"""Generate data/Phi_wdbc.npy: repeated out-of-fold |SHAP| importance matrix (R x p).
Run once before scripts that consume it (transfer.py, close_bc.py, predictive.py)."""
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier
import shap
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values; p=X.shape[1]
R=10; Phi=np.zeros((R,p))
for r in range(R):
    skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=r); oof=np.zeros((len(y),p))
    for tr,te in skf.split(X,y):
        m=XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.08,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0).fit(X[tr],y[tr])
        oof[te]=np.abs(shap.TreeExplainer(m).shap_values(X[te]))
    Phi[r]=oof.mean(0)
np.save("data/Phi_wdbc.npy",Phi); print("saved data/Phi_wdbc.npy", Phi.shape)
