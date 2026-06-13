import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
def ens(): return VotingClassifier([("xgb",XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)),("lgb",LGBMClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=0,n_jobs=1,verbose=-1)),("svm",make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0))),("lr",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)))],voting="soft")
def ci(v): v=np.asarray(v); return 1.96*v.std()/np.sqrt(len(v)) if len(v)>1 else 0
def perclass(X,y,reps=20,alpha=0.10):
    c0=[];c1=[]
    for s in range(reps):
        Xtr,Xt,ytr,yt=train_test_split(X,y,test_size=0.4,stratify=y,random_state=s)
        Xc,Xe,yc,ye=train_test_split(Xt,yt,test_size=0.5,stratify=yt,random_state=s)
        m=ens().fit(Xtr,ytr); pc=m.predict_proba(Xc); pe=m.predict_proba(Xe); q={}
        for cl in [0,1]:
            sc=1-pc[yc==cl,cl]; nn=len(sc); q[cl]=np.quantile(sc,min(1,np.ceil((nn+1)*(1-alpha))/nn))
        cov={cl:[] for cl in [0,1]}
        for i in range(len(ye)):
            ss={cl for cl in [0,1] if (1-pe[i,cl])<=q[cl]}; cov[ye[i]].append(ye[i] in ss)
        c0.append(np.mean(cov[0])); c1.append(np.mean(cov[1]))
    return (np.mean(c0),ci(c0)),(np.mean(c1),ci(c1))
def load():
    d=pd.read_csv("data/wdbc.csv");d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
    W=(d.drop(columns=["diagnosis"]).values,(d["diagnosis"].values=="M").astype(int))
    w=pd.read_csv("data/wbc_selva.csv").drop(columns=["Id"]);w["Bare.nuclei"]=pd.to_numeric(w["Bare.nuclei"],errors="coerce")
    Wb=(SimpleImputer(strategy="median").fit_transform(w.drop(columns=["Class"]).values),w["Class"].values.astype(int))
    c=pd.read_csv("data/coimbra.csv");Co=(c.drop(columns=["Classification"]).values,(c["Classification"].values=="Yes").astype(int))
    mm=pd.read_csv("data/mammo.csv",na_values=["?"],names=["BIRADS","age","shape","margin","density","severity"]).drop(columns=["BIRADS"])
    Mm=(SimpleImputer(strategy="median").fit_transform(mm.drop(columns=["severity"]).values),mm["severity"].values.astype(int))
    return {"WDBC":W,"WBC":Wb,"Coimbra":Co,"Mammographic":Mm}
print("Per-class conformal coverage (target 0.90), repeated 20 splits:")
for nm,(X,y) in load().items():
    (m0,h0),(m1,h1)=perclass(X,y)
    print(f"  {nm:12s} benign/neg cov={m0:.3f}+/-{h0:.3f}   malignant/pos cov={m1:.3f}+/-{h1:.3f}")
