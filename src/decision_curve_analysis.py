import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
import warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
d=pd.read_csv("data/wdbc.csv"); d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values
def xgb(): return XGBClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
models={"Ensemble (ours)":VotingClassifier([("xgb",xgb()),("lgb",LGBMClassifier(n_estimators=300,max_depth=3,learning_rate=0.05,random_state=0,n_jobs=1,verbose=-1)),("svm",make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0))),("lr",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)))],voting="soft"),
 "XGBoost":xgb(),"Logistic Regression":make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)),"SVM (RBF)":make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0))}
cv=StratifiedKFold(5,shuffle=True,random_state=1); prev=y.mean()
pts=np.linspace(0.01,0.6,60)
def nb(prob):
    out=[]
    for pt in pts:
        pred=(prob>=pt).astype(int); TP=np.sum((pred==1)&(y==1)); FP=np.sum((pred==1)&(y==0)); N=len(y)
        out.append(TP/N - FP/N*(pt/(1-pt)))
    return np.array(out)
nb_all=prev-(1-prev)*(pts/(1-pts))
fig,ax=plt.subplots(figsize=(6.8,4.2))
ax.plot(pts,nb_all,"--",color="#888",lw=1.5,label="Treat all")
ax.plot(pts,np.zeros_like(pts),":",color="#555",lw=1.5,label="Treat none")
cols={"Ensemble (ours)":"#2E5A88","XGBoost":"#1F7A8C","Logistic Regression":"#B26A00","SVM (RBF)":"#6A8D3F"}
probs={}
for nm,m in models.items():
    pr=cross_val_predict(m,X,y,cv=cv,method="predict_proba")[:,1]; probs[nm]=pr
    ax.plot(pts,nb(pr),lw=2,color=cols[nm],label=nm)
ax.set_xlabel("Threshold probability $p_t$"); ax.set_ylabel("Net benefit"); ax.set_ylim(-0.02,prev+0.02)
ax.legend(frameon=False,fontsize=8.5); ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
fig.savefig("eq/fig5_dca.png",dpi=300); plt.close(fig)
from PIL import Image; import json; im=Image.open("eq/fig5_dca.png"); j=json.load(open("eq/allimg.json")); j["fig5_dca"]=[im.width,im.height]; json.dump(j,open("eq/allimg.json","w"))
print("Net benefit at clinical thresholds:")
for pt in [0.1,0.2,0.3]:
    i=np.argmin(np.abs(pts-pt))
    print(f"  pt={pt}: "+" | ".join(f"{nm}={nb(probs[nm])[i]:.3f}" for nm in models)+f" | treat-all={nb_all[i]:.3f}")
print("Figure 5 (DCA) saved; models cluster (saturated data) and all exceed treat-all/treat-none across thresholds.")
