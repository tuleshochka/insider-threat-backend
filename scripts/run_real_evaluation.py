"""
v3: User-baseline z-scores + rolling windows + RF/XGB/LGB stacking.
"""
import os,sys,warnings,io,time,gc
warnings.filterwarnings("ignore")
sys.stdout=io.TextIOWrapper(sys.stdout.buffer,encoding='utf-8',errors='replace')
sys.stderr=io.TextIOWrapper(sys.stderr.buffer,encoding='utf-8',errors='replace')
SCRIPT_DIR=os.path.dirname(os.path.abspath(__file__))
ROOT=os.path.dirname(SCRIPT_DIR); BACK=ROOT; ASSETS=os.path.join(ROOT,"assets")
os.makedirs(ASSETS,exist_ok=True); os.environ["DB_TYPE"]="sqlite"; os.chdir(BACK); sys.path.insert(0,BACK)
import numpy as np, pandas as pd
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt, seaborn as sns
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier,IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (classification_report,confusion_matrix,precision_recall_curve,
    f1_score,fbeta_score,precision_score,recall_score,auc)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb, lightgbm as lgb
from app.config import settings
T0=time.time()
def el(): return f"[{time.time()-T0:.0f}s]"
data_dir=settings.dataset_path; sd=pd.to_datetime("2010-01-01"); ed=pd.to_datetime("2011-05-01")

# === 1. LOAD ===
print("="*70+f"\n{el()} LOAD DATA\n"+"="*70)
def load(name,cols,dt):
    p=os.path.join(data_dir,f"{name}.csv"); chunks=[]; tot=0
    print(f"  {el()} {name}.csv ...",end=" ",flush=True)
    for c in pd.read_csv(p,usecols=cols,dtype=dt,chunksize=500000,low_memory=False):
        tot+=len(c); c["date"]=pd.to_datetime(c["date"],format="%m/%d/%Y %H:%M:%S",errors="coerce")
        c=c[(c["date"]>=sd)&(c["date"]<ed)]
        if not c.empty: chunks.append(c)
    r=pd.concat(chunks,ignore_index=True) if chunks else pd.DataFrame()
    print(f"{len(r):,} ({tot:,})"); return r

logon=load("logon",["date","user","pc","activity","id"],{"user":str,"pc":str,"activity":str,"id":str})
device=load("device",["date","user","pc","activity","id"],{"user":str,"pc":str,"activity":str,"id":str})
files=load("file",["date","user","pc","filename","id"],{"user":str,"pc":str,"filename":str,"id":str})
email=load("email",["date","user","to","from","size","attachments","id"],{"user":str,"to":str,"from":str,"size":"int64","attachments":"int64","id":str})

print(f"  {el()} http.csv (chunked) ...",end=" ",flush=True)
haggs=[]; htot=0
for c in pd.read_csv(os.path.join(data_dir,"http.csv"),usecols=["date","user","url","id"],dtype={"user":str,"url":str,"id":str},chunksize=1000000,low_memory=False):
    htot+=len(c); c["date"]=pd.to_datetime(c["date"],format="%m/%d/%Y %H:%M:%S",errors="coerce")
    c=c[(c["date"]>=sd)&(c["date"]<ed)]
    if c.empty: continue
    c["do"]=c["date"].dt.date; c["h"]=c["date"].dt.hour
    a=c.groupby(["do","user"],as_index=False).agg(http_requests=("id","count"),http_unique_urls=("url","nunique"),after_hours_http=("h",lambda x:((x<8)|(x>=18)).sum()))
    haggs.append(a)
if haggs:
    ha=pd.concat(haggs,ignore_index=True)
    a_http=ha.groupby(["do","user"],as_index=False).agg(http_requests=("http_requests","sum"),http_unique_urls=("http_unique_urls","sum"),after_hours_http=("after_hours_http","sum"))
    a_http.rename(columns={"user":"anon_id","do":"date"},inplace=True); del ha,haggs
else: a_http=pd.DataFrame()
print(f"done ({htot:,})"); gc.collect()

# === 2. AGGREGATE ===
print(f"\n{el()} AGGREGATE")
def al(df):
    df["do"]=df["date"].dt.date;df["h"]=df["date"].dt.hour;df["dw"]=df["date"].dt.dayofweek
    r=df.groupby(["do","user"],as_index=False).agg(logon_count=("id","count"),logon_unique_pc=("pc","nunique"),after_hours_logons=("h",lambda x:((x<8)|(x>=18)).sum()),weekend_logons=("dw",lambda x:(x>=5).sum()))
    r.rename(columns={"user":"anon_id","do":"date"},inplace=True);return r
def af(df):
    df["do"]=df["date"].dt.date;df["h"]=df["date"].dt.hour
    r=df.groupby(["do","user"],as_index=False).agg(file_operations=("id","count"),file_unique_pc=("pc","nunique"),file_unique_names=("filename","nunique"),after_hours_files=("h",lambda x:((x<8)|(x>=18)).sum()))
    r.rename(columns={"user":"anon_id","do":"date"},inplace=True);return r
def ae(df):
    df["do"]=df["date"].dt.date;df["h"]=df["date"].dt.hour
    s=df[df["user"]==df["from"]].groupby(["do","user"],as_index=False).agg(email_sent=("id","count"),email_size_total=("size","sum"),email_attachments=("attachments","sum"),email_unique_recipients=("to","nunique"),after_hours_email=("h",lambda x:((x<8)|(x>=18)).sum()))
    rv=df[df["user"]!=df["from"]].groupby(["do","user"],as_index=False).agg(email_received=("id","count"))
    r=s.merge(rv,on=["do","user"],how="outer").fillna(0);r.rename(columns={"user":"anon_id","do":"date"},inplace=True);return r
def ad(df):
    df["do"]=df["date"].dt.date;df["h"]=df["date"].dt.hour;df["dw"]=df["date"].dt.dayofweek
    r=df.groupby(["do","user"],as_index=False).agg(device_operations=("id","count"),after_hours_device=("h",lambda x:((x<8)|(x>=18)).sum()),weekend_device=("dw",lambda x:(x>=5).sum()))
    r.rename(columns={"user":"anon_id","do":"date"},inplace=True);return r

print(f"  {el()} logon");a1=al(logon);del logon
print(f"  {el()} file");a2=af(files);del files
print(f"  {el()} email");a3=ae(email);del email
print(f"  {el()} device");a4=ad(device);del device;gc.collect()
print(f"  {el()} merge")
feat=a1.merge(a2,on=["date","anon_id"],how="outer").merge(a3,on=["date","anon_id"],how="outer").merge(a4,on=["date","anon_id"],how="outer")
if not a_http.empty: feat=feat.merge(a_http,on=["date","anon_id"],how="outer"); del a_http
feat.fillna(0,inplace=True); del a1,a2,a3,a4; gc.collect()
pp=os.path.join(data_dir,"psychometric.csv")
if os.path.exists(pp):
    ps=pd.read_csv(pp,usecols=["user_id","O","C","E","A","N"]);ps.rename(columns={"user_id":"anon_id"},inplace=True)
    feat=feat.merge(ps,on="anon_id",how="left");feat[["O","C","E","A","N"]]=feat[["O","C","E","A","N"]].fillna(0)

# === 3. FEATURE ENGINEERING ===
print(f"\n{el()} FEATURE ENGINEERING")
feat["after_hours_ratio"]=(feat["after_hours_logons"]+feat["after_hours_files"]+feat["after_hours_device"]+feat.get("after_hours_http",0)+feat["after_hours_email"])/(feat["logon_count"]+feat["file_operations"]+feat["device_operations"]+feat.get("http_requests",0)+feat["email_sent"]+1)
feat["device_to_file_ratio"]=feat["device_operations"]/(feat["file_operations"]+1)
feat["email_size_per_msg"]=feat["email_size_total"]/(feat["email_sent"]+1)
feat["files_per_pc"]=feat["file_operations"]/(feat["file_unique_pc"]+1)
feat["weekend_activity"]=feat["weekend_logons"]+feat["weekend_device"]

# USER-BASELINE Z-SCORES (core UEBA concept!)
print(f"  {el()} User-baseline z-scores...")
activity_cols=["logon_count","file_operations","email_sent","device_operations","after_hours_logons","after_hours_files","after_hours_device","weekend_logons"]
if "http_requests" in feat.columns: activity_cols.append("http_requests")
feat["date"]=pd.to_datetime(feat["date"])
feat.sort_values(["anon_id","date"],inplace=True)
for col in activity_cols:
    umean=feat.groupby("anon_id")[col].transform("mean")
    ustd=feat.groupby("anon_id")[col].transform("std").replace(0,1)
    feat[f"{col}_zscore"]=(feat[col]-umean)/ustd

# ROLLING WINDOWS via shift (lag-1, lag-3 moving avg)
print(f"  {el()} Rolling lag features...")
key_cols=["logon_count","file_operations","device_operations","email_sent"]
if "http_requests" in feat.columns: key_cols.append("http_requests")
feat.sort_values(["anon_id","date"],inplace=True)
for col in key_cols:
    g=feat.groupby("anon_id")[col]
    feat[f"{col}_lag1"]=g.shift(1).fillna(0)
    feat[f"{col}_lag3_avg"]=(g.shift(1).fillna(0)+g.shift(2).fillna(0)+g.shift(3).fillna(0))/3
    feat[f"{col}_diff"]=g.diff().fillna(0)

print(f"  {el()} Total: {len(feat):,} rows, {len(feat.columns)} cols")

# === 4. GROUND TRUTH ===
print(f"\n{el()} GROUND TRUTH")
gt=pd.read_csv(settings.ground_truth_path); r42=gt[gt["dataset"]==4.2].copy()
r42["start"]=pd.to_datetime(r42["start"]);r42["end"]=pd.to_datetime(r42["end"])
mal=set()
for _,row in r42.iterrows():
    cur=row["start"]
    while cur<=row["end"]: mal.add((row["user"],cur.date()));cur+=pd.Timedelta(days=1)
y=np.array([1 if (r["anon_id"],r["date"].date() if hasattr(r["date"],"date") else r["date"]) in mal else 0 for _,r in feat.iterrows()],dtype=np.int32)
np_=int(y.sum());nn=len(y)-np_
print(f"  Pos: {np_}, Neg: {nn}, Ratio: {np_/len(y)*100:.3f}%")

skip={"date","anon_id","role","department","business_unit","employee_name","functional_unit","start_date","end_date"}
fcols=[c for c in feat.columns if c not in skip and feat[c].dtype in ['float64','int64','float32','int32']]
X=feat[fcols].fillna(0).values.astype(np.float64)
sc=StandardScaler(); X=sc.fit_transform(X)
print(f"  Features: {len(fcols)}, shape: {X.shape}")

# === 5. ISOLATION FOREST ===
print(f"\n{'='*70}\n{el()} ISOLATION FOREST\n{'='*70}")
from sklearn.model_selection import train_test_split
Xt,Xe,yt,ye=train_test_split(X,y,test_size=0.3,random_state=42,stratify=y)
iso=IsolationForest(n_estimators=200,contamination=0.005,random_state=42,n_jobs=-1);iso.fit(Xt)
ip=np.where(iso.predict(Xe)==-1,1,0)
if_p=precision_score(ye,ip,zero_division=0);if_r=recall_score(ye,ip,zero_division=0)
if_f1=f1_score(ye,ip,zero_division=0);if_f2=fbeta_score(ye,ip,beta=2,zero_division=0)
print(f"  P={if_p:.4f} R={if_r:.4f} F1={if_f1:.4f} F2={if_f2:.4f}")

# === 6. STACKING: RF + XGB + LGB => LogReg ===
print(f"\n{'='*70}\n{el()} STACKING ENSEMBLE (RF+XGB+LGB => LogReg)\n{'='*70}")
skf=StratifiedKFold(n_splits=5,shuffle=True,random_state=42)
rf_p_all=np.zeros(len(y));xgb_p_all=np.zeros(len(y));lgb_p_all=np.zeros(len(y))

for fold,(tr,te) in enumerate(skf.split(X,y),1):
    t0=time.time()
    # RF
    p1=ImbPipeline([('s',SMOTE(sampling_strategy=0.5,random_state=42,k_neighbors=2)),
        ('c',RandomForestClassifier(n_estimators=500,max_depth=25,min_samples_leaf=2,class_weight="balanced_subsample",random_state=42,n_jobs=-1))])
    p1.fit(X[tr],y[tr]); rf_p_all[te]=p1.predict_proba(X[te])[:,1]
    # XGB
    p2=ImbPipeline([('s',SMOTE(sampling_strategy=0.5,random_state=42,k_neighbors=2)),
        ('c',xgb.XGBClassifier(n_estimators=500,max_depth=8,learning_rate=0.05,scale_pos_weight=nn/np_,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,eval_metric='logloss',random_state=42,n_jobs=-1,verbosity=0))])
    p2.fit(X[tr],y[tr]); xgb_p_all[te]=p2.predict_proba(X[te])[:,1]
    # LGB
    p3=ImbPipeline([('s',SMOTE(sampling_strategy=0.5,random_state=42,k_neighbors=2)),
        ('c',lgb.LGBMClassifier(n_estimators=500,max_depth=10,learning_rate=0.05,scale_pos_weight=nn/np_,subsample=0.8,colsample_bytree=0.8,reg_alpha=0.1,reg_lambda=1.0,random_state=42,n_jobs=-1,verbose=-1))])
    p3.fit(X[tr],y[tr]); lgb_p_all[te]=p3.predict_proba(X[te])[:,1]
    print(f"  {el()} Fold {fold} done ({time.time()-t0:.0f}s)")

# Stacking meta-learner
print(f"  {el()} Training stacking meta-learner...")
meta_X=np.column_stack([rf_p_all,xgb_p_all,lgb_p_all])
meta_clf=LogisticRegression(class_weight="balanced",max_iter=1000,random_state=42)
# CV for meta too
stack_p_all=np.zeros(len(y))
for tr,te in skf.split(meta_X,y):
    meta_clf.fit(meta_X[tr],y[tr]); stack_p_all[te]=meta_clf.predict_proba(meta_X[te])[:,1]

# Simple average ensemble too
avg_p=(rf_p_all+xgb_p_all+lgb_p_all)/3.0

def evaluate(name,probs,yt):
    bf2,bt=0,0.5
    for t in np.arange(0.01,0.60,0.005):
        yp=(probs>=t).astype(int);f2=fbeta_score(yt,yp,beta=2,zero_division=0)
        if f2>bf2:bf2=f2;bt=t
    yp=(probs>=bt).astype(int)
    p=precision_score(yt,yp,zero_division=0);r=recall_score(yt,yp,zero_division=0)
    f1=f1_score(yt,yp,zero_division=0);f2=fbeta_score(yt,yp,beta=2,zero_division=0)
    pp2,rr2,_=precision_recall_curve(yt,probs);prauc=auc(rr2,pp2)
    print(f"  {name:25s}: P={p:.4f} R={r:.4f} F1={f1:.4f} F2={f2:.4f} PR-AUC={prauc:.4f} thr={bt:.3f}")
    return p,r,f1,f2,prauc,bt,yp

print(f"\n{el()} RESULTS:")
rf_m=evaluate("RF+SMOTE",rf_p_all,y)
xgb_m=evaluate("XGBoost+SMOTE",xgb_p_all,y)
lgb_m=evaluate("LightGBM+SMOTE",lgb_p_all,y)
avg_m=evaluate("Average Ensemble",avg_p,y)
stk_m=evaluate("Stacking (LogReg)",stack_p_all,y)

# Pick best
all_m=[("RF+SMOTE",rf_m),("XGBoost+SMOTE",xgb_m),("LightGBM+SMOTE",lgb_m),("Average Ensemble",avg_m),("Stacking",stk_m)]
best_name,best_m=max(all_m,key=lambda x:x[1][3])
print(f"\n  >>> Best: {best_name} (F2={best_m[3]:.4f})")

# Feature importance from XGBoost (last fold)
xgb_model=p2.named_steps['c']
feat_imp=sorted(zip(fcols,xgb_model.feature_importances_),key=lambda x:x[1],reverse=True)

# === 7. PLOTS ===
print(f"\n{'='*70}\n{el()} PLOTS\n{'='*70}")
plt.rcParams.update({'font.size':12,'axes.titlesize':14,'axes.labelsize':12,'figure.facecolor':'white','savefig.dpi':200,'savefig.bbox':'tight'})

cm=confusion_matrix(y,best_m[6])
fig,ax=plt.subplots(figsize=(7,6))
sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',ax=ax,xticklabels=['Normal','Insider'],yticklabels=['Normal','Insider'])
ax.set_xlabel('Predicted');ax.set_ylabel('Actual');ax.set_title(f'Confusion Matrix - {best_name} (5-Fold CV)')
fig.savefig(os.path.join(ASSETS,"confusion_matrix.png"));plt.close(fig);print("  [OK] confusion_matrix.png")

fig,ax=plt.subplots(figsize=(8,6))
for nm,pr,st in [("RF",rf_p_all,'b-'),("XGB",xgb_p_all,'r--'),("LGB",lgb_p_all,'g-.'),("Stack",stack_p_all,'k-')]:
    pp2,rr2,_=precision_recall_curve(y,pr);a2=auc(rr2,pp2);ax.plot(rr2,pp2,st,lw=2,label=f'{nm} (AUC={a2:.3f})')
ax.set_xlabel('Recall');ax.set_ylabel('Precision');ax.set_title('PR Curves');ax.legend();ax.grid(True,alpha=0.3)
fig.savefig(os.path.join(ASSETS,"pr_curve.png"));plt.close(fig);print("  [OK] pr_curve.png")

top_n=min(15,len(feat_imp));ns=[f[0] for f in feat_imp[:top_n]][::-1];vs=[f[1] for f in feat_imp[:top_n]][::-1]
fig,ax=plt.subplots(figsize=(10,7));ax.barh(ns,vs,color=sns.color_palette("viridis",top_n)[::-1])
ax.set_xlabel('Importance');ax.set_title('Top-15 Feature Importance (XGBoost)')
ax.grid(True,axis='x',alpha=0.3);fig.savefig(os.path.join(ASSETS,"feature_importance.png"));plt.close(fig);print("  [OK] feature_importance.png")

mnames=['IF','RF','XGB','LGB','Avg','Stack']
x=np.arange(len(mnames));w=0.2
fig,ax=plt.subplots(figsize=(14,6))
ps=[if_p,rf_m[0],xgb_m[0],lgb_m[0],avg_m[0],stk_m[0]]
rs=[if_r,rf_m[1],xgb_m[1],lgb_m[1],avg_m[1],stk_m[1]]
f1s=[if_f1,rf_m[2],xgb_m[2],lgb_m[2],avg_m[2],stk_m[2]]
f2s=[if_f2,rf_m[3],xgb_m[3],lgb_m[3],avg_m[3],stk_m[3]]
ax.bar(x-1.5*w,ps,w,label='Precision',color='#3498db');ax.bar(x-0.5*w,rs,w,label='Recall',color='#e74c3c')
ax.bar(x+0.5*w,f1s,w,label='F1',color='#2ecc71');ax.bar(x+1.5*w,f2s,w,label='F2',color='#9b59b6')
ax.set_ylabel('Score');ax.set_title('Model Comparison');ax.set_xticks(x);ax.set_xticklabels(mnames)
ax.legend();ax.set_ylim(0,1.05);ax.grid(True,axis='y',alpha=0.3)
for b in ax.containers:ax.bar_label(b,fmt='%.3f',padding=2,fontsize=7)
fig.savefig(os.path.join(ASSETS,"metrics_comparison.png"));plt.close(fig);print("  [OK] metrics_comparison.png")

with open(os.path.join(ASSETS,"real_metrics.txt"),"w",encoding="utf-8") as f:
    f.write(f"CERT r4.2 FULL DATA + STACKING v3\n{'='*50}\n\n")
    f.write(f"Observations: {len(feat):,}\nPositive: {np_}\nNegative: {nn}\nFeatures: {len(fcols)}\n\n")
    f.write(f"IF:      P={if_p:.4f} R={if_r:.4f} F1={if_f1:.4f} F2={if_f2:.4f}\n")
    f.write(f"RF:      P={rf_m[0]:.4f} R={rf_m[1]:.4f} F1={rf_m[2]:.4f} F2={rf_m[3]:.4f} PR-AUC={rf_m[4]:.4f}\n")
    f.write(f"XGB:     P={xgb_m[0]:.4f} R={xgb_m[1]:.4f} F1={xgb_m[2]:.4f} F2={xgb_m[3]:.4f} PR-AUC={xgb_m[4]:.4f}\n")
    f.write(f"LGB:     P={lgb_m[0]:.4f} R={lgb_m[1]:.4f} F1={lgb_m[2]:.4f} F2={lgb_m[3]:.4f} PR-AUC={lgb_m[4]:.4f}\n")
    f.write(f"Avg:     P={avg_m[0]:.4f} R={avg_m[1]:.4f} F1={avg_m[2]:.4f} F2={avg_m[3]:.4f} PR-AUC={avg_m[4]:.4f}\n")
    f.write(f"Stack:   P={stk_m[0]:.4f} R={stk_m[1]:.4f} F1={stk_m[2]:.4f} F2={stk_m[3]:.4f} PR-AUC={stk_m[4]:.4f}\n\n")
    f.write(f"Best: {best_name} (F2={best_m[3]:.4f})\n")
    cm2=confusion_matrix(y,best_m[6]);f.write(f"TN={cm2[0][0]} FP={cm2[0][1]}\nFN={cm2[1][0]} TP={cm2[1][1]}\n\n")
    f.write("Feature Importance (Top-15):\n")
    for n,v in feat_imp[:15]:f.write(f"  {n:35s} {v:.4f}\n")
print(f"\n{'='*70}\nDONE in {(time.time()-T0)/60:.1f} min\n{'='*70}")
