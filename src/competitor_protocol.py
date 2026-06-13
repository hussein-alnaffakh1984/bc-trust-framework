import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.feature_selection import SelectFromModel, RFE
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values
def xgb(**k): return XGBClassifier(eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0,**k)
cv=RepeatedStratifiedKFold(n_splits=5,n_repeats=4,random_state=1)
def ci(v): return 1.96*v.std()/np.sqrt(len(v))

pipes={
 "Importance-FS + XGBoost":Pipeline([("fs",SelectFromModel(xgb(n_estimators=200,max_depth=3),max_features=10,threshold=-np.inf)),("clf",xgb(n_estimators=300,max_depth=3,learning_rate=0.05))]),
 "RFE + SVM(RBF)":Pipeline([("sc",StandardScaler()),("fs",RFE(LogisticRegression(max_iter=2000),n_features_to_select=10)),("clf",SVC(C=3))]),
 "PCA + LogReg":Pipeline([("sc",StandardScaler()),("pca",PCA(n_components=10)),("clf",LogisticRegression(max_iter=2000))]),
 "Tuned single XGBoost":xgb(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9),
 "Our ensemble":VotingClassifier([("xgb",xgb(n_estimators=400,max_depth=3,learning_rate=0.05)),("lgb",LGBMClassifier(n_estimators=400,max_depth=3,learning_rate=0.05,random_state=0,n_jobs=1,verbose=-1)),("svm",make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0))),("lr",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)))],voting="soft"),
}
print("Representative literature pipelines under IDENTICAL repeated 5x4 CV (WDBC):")
for nm,p in pipes.items():
    s=cross_val_score(p,X,y,cv=cv,scoring="accuracy",n_jobs=2)
    print(f"  {nm:26s} acc = {s.mean():.4f} +/- {ci(s):.4f}")
