import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, warnings, numpy as np, pandas as pd
warnings.filterwarnings("ignore")
from scipy.stats import kendalltau, wilcoxon
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate, StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
import shap
plt.rcParams.update({"font.size":11,"axes.edgecolor":"#444"})
AC="#2E5A88"; TE="#1F7A8C"; GR="#5A6B7B"; OR="#B26A00"

d=pd.read_csv("data/wdbc.csv")
d=d.drop(columns=[c for c in d.columns if "Unnamed" in c or c=="id"])
y=(d["diagnosis"].values=="M").astype(int); X=d.drop(columns=["diagnosis"]).values
feat=np.array(d.drop(columns=["diagnosis"]).columns)

def xgb(): return XGBClassifier(n_estimators=200,max_depth=3,learning_rate=0.08,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)
def ens():
    return VotingClassifier([("xgb",XGBClassifier(n_estimators=400,max_depth=3,learning_rate=0.05,subsample=0.9,colsample_bytree=0.9,eval_metric="logloss",random_state=0,n_jobs=1,verbosity=0)),
        ("lgb",LGBMClassifier(n_estimators=400,max_depth=3,learning_rate=0.05,random_state=0,n_jobs=1,verbose=-1)),
        ("svm",make_pipeline(StandardScaler(),SVC(C=3,probability=True,random_state=0))),
        ("lr",make_pipeline(StandardScaler(),LogisticRegression(C=1,max_iter=2000)))],voting="soft")

# ---- Fig2: single-split variance ----
from sklearn.model_selection import train_test_split as tts
accs=[]
for s in range(300):
    Xtr,Xte,ytr,yte=tts(X,y,test_size=0.25,stratify=y,random_state=s)
    m=xgb().fit(Xtr,ytr); accs.append((m.predict(Xte)==yte).mean())
accs=np.array(accs)
fig,ax=plt.subplots(figsize=(6.6,3.6))
ax.hist(accs,bins=22,color=AC,alpha=.8,edgecolor="white")
ax.axvline(accs.mean(),color=OR,lw=2,ls="--",label=f"mean = {accs.mean():.3f}")
ax.axvline(accs.max(),color="#B0241B",lw=2,ls=":",label=f"max = {accs.max():.3f}")
ax.set_xlabel("Test accuracy of one fixed model across 300 random 75/25 splits")
ax.set_ylabel("Frequency"); ax.legend(frameon=False); ax.spines[["top","right"]].set_visible(False)
fig.tight_layout(); fig.savefig("eq/fig2_splitvar.png",dpi=300); plt.close(fig)
print(f"[Fig2] single-split acc: min={accs.min():.3f} mean={accs.mean():.3f} p95={np.percentile(accs,95):.3f} max={accs.max():.3f}")

# ---- ESI + psi for WDBC ----
R=10;K=5;p=X.shape[1];Phi=np.zeros((R,p))
for r in range(R):
    skf=StratifiedKFold(n_splits=K,shuffle=True,random_state=r);oof=np.zeros((len(y),p))
    for tr,te in skf.split(X,y):
        m=xgb().fit(X[tr],y[tr]);oof[te]=np.abs(shap.TreeExplainer(m).shap_values(X[te]))
    Phi[r]=oof.mean(0)
pb=Phi.mean(0);sd=Phi.std(0);cv=np.divide(sd,pb,out=np.zeros_like(sd),where=pb>0);mm=1/(1+cv)
k=10;tops=[set(np.argsort(Phi[r])[::-1][:k]) for r in range(R)]
fsel=np.mean([[1.0 if j in tops[r] else 0 for j in range(p)] for r in range(R)],0)
psi=fsel*mm
order=np.argsort(psi)[::-1][:12]
fig,ax=plt.subplots(figsize=(6.6,4.2))
ax.barh(range(len(order))[::-1],psi[order],color=TE,alpha=.85,edgecolor="white")
ax.set_yticks(range(len(order))[::-1]); ax.set_yticklabels([feat[j] for j in order],fontsize=9)
ax.set_xlabel("Per-feature stability  $\\psi_j$"); ax.set_xlim(0,1)
ax.spines[["top","right"]].set_visible(False); fig.tight_layout()
fig.savefig("eq/fig3_biomarkers.png",dpi=300); plt.close(fig)
print("[Fig3] top biomarkers:", ", ".join(feat[order][:6]))

# Fig4 generated separately (4-cohort, legend above) — see regen_fig4.py


# ---- professional statistics ----
cvr=RepeatedStratifiedKFold(n_splits=5,n_repeats=4,random_state=1)
re=cross_validate(ens(),X,y,cv=cvr,scoring="accuracy",n_jobs=2)["test_accuracy"]
rx=cross_validate(xgb(),X,y,cv=cvr,scoring="accuracy",n_jobs=2)["test_accuracy"]
W,pv=wilcoxon(re,rx)
ci=1.96*re.std()/np.sqrt(len(re))
print(f"[Stat] ensemble acc {re.mean():.4f} (95% CI +/-{ci:.4f}); vs XGBoost {rx.mean():.4f}; Wilcoxon p={pv:.4f}")
# ESI vs null
nulls=[]
for b in range(10):
    yp=np.random.RandomState(b).permutation(y); Ph=np.zeros((4,p))
    for r in range(4):
        skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=100+r);oof=np.zeros((len(y),p))
        for tr,te in skf.split(X,yp):
            m=xgb().fit(X[tr],yp[tr]);oof[te]=np.abs(shap.TreeExplainer(m).shap_values(X[te]))
        Ph[r]=oof.mean(0)
    pbn=Ph.mean(0);sdn=Ph.std(0);cvn=np.divide(sdn,pbn,out=np.zeros_like(sdn),where=pbn>0);mn=1/(1+cvn)
    Sm=np.sum(pbn*mn)/np.sum(pbn)
    t=[kendalltau(Ph[a],Ph[bb]).correlation for a in range(4) for bb in range(a+1,4)];Sr=(1+np.nanmean(t))/2
    tp=[set(np.argsort(Ph[r])[::-1][:10]) for r in range(4)]
    ki=[(len(tp[a]&tp[bb])-100/p)/(10-100/p) for a in range(4) for bb in range(a+1,4)];Ss=max(0,np.mean(ki))
    nulls.append((Sm**(1/3))*(Sr**(1/3))*(max(Ss,1e-9)**(1/3)))
# observed ESI
Sm=np.sum(pb*mm)/np.sum(pb)
t=[kendalltau(Phi[a],Phi[b]).correlation for a in range(R) for b in range(a+1,R)];Sr=(1+np.nanmean(t))/2
ki=[(len(tops[a]&tops[b])-100/p)/(10-100/p) for a in range(R) for b in range(a+1,R)];Ss=max(0,np.mean(ki))
ESI=(Sm**(1/3))*(Sr**(1/3))*(Ss**(1/3))
E0=np.mean(nulls);esirel=(ESI-E0)/(1-E0);pp=np.mean(np.array(nulls)>=ESI)
print(f"[Stat] ESI={ESI:.3f} (S_mag={Sm:.3f},S_rank={Sr:.3f},S_sel={Ss:.3f}); E[null]={E0:.3f}; ESI_rel={esirel:.3f}; perm p={pp:.3f}")
