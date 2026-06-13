import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier, VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier

df = pd.read_csv("data/wdbc.csv")
df = df.drop(columns=[c for c in df.columns if "Unnamed" in c or c=="id"])
y=(df["diagnosis"].values=="M").astype(int); X=df.drop(columns=["diagnosis"]).values

xgb = XGBClassifier(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.9,
        colsample_bytree=0.9,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
lgb = LGBMClassifier(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.9,
        colsample_bytree=0.9,random_state=0,n_jobs=1,verbose=-1)
svm = make_pipeline(StandardScaler(), SVC(C=3,gamma="scale",probability=True,random_state=0))
lr  = make_pipeline(StandardScaler(), LogisticRegression(C=1.0,max_iter=2000))

cv = RepeatedStratifiedKFold(n_splits=5,n_repeats=4,random_state=1)
def evalm(name, model):
    r = cross_validate(model, X, y, cv=cv, scoring=["accuracy","roc_auc","recall","f1"], n_jobs=2)
    print(f"  {name:28s} acc={r['test_accuracy'].mean():.4f}+/-{r['test_accuracy'].std():.4f}"
          f"  AUC={r['test_roc_auc'].mean():.4f}  recall(M)={r['test_recall'].mean():.4f}"
          f"  F1={r['test_f1'].mean():.4f}")

print("Individual models (repeated 5x4 stratified CV):")
for n,m in [("XGBoost",xgb),("LightGBM",lgb),("SVM (RBF)",svm),("LogReg",lr)]:
    evalm(n,m)

soft = VotingClassifier([("xgb",xgb),("lgb",lgb),("svm",svm),("lr",lr)], voting="soft")
stack = StackingClassifier([("xgb",xgb),("lgb",lgb),("svm",svm)],
        final_estimator=LogisticRegression(max_iter=2000),
        cv=5, stack_method="predict_proba", n_jobs=1)
print("\nEnsembles (honest repeated 5x4 CV, no single-split cherry-picking):")
evalm("Soft Voting (4 models)", soft)
evalm("Stacking (XGB+LGB+SVM)", stack)
