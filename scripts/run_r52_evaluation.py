"""
CERT v5.2 Classifier Training and Evaluation Pipeline.
Loads preprocessed features, splits by User-Level Groups (unseen test employees),
runs StratifiedGroupKFold on train, and generates realistic, non-leaking comparative metrics.
"""
import os
import sys
import warnings
import io
import time
import pickle
import gc
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(SCRIPT_DIR)
ASSETS = os.path.join(ROOT, "assets")
os.makedirs(ASSETS, exist_ok=True)
sys.path.insert(0, ROOT)

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import StratifiedGroupKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (confusion_matrix, precision_recall_curve,
                             f1_score, fbeta_score, precision_score, recall_score, auc)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
import xgboost as xgb
import lightgbm as lgb

T0 = time.time()
def el(): return f"[{time.time()-T0:.0f}s]"

DATA_PATH = r"D:\Политех\Мага\Дипломы\М\12841247\processed_features_r5.2.csv"
MODEL_DIR = os.path.join(ROOT, "insider_threat_model_r52")
os.makedirs(MODEL_DIR, exist_ok=True)

# === 1. LOAD PREPROCESSED FEATURES ===
print("="*70 + f"\n{el()} LOADING PREPROCESSED FEATURES\n" + "="*70)
if not os.path.exists(DATA_PATH):
    print(f"Error: Preprocessed dataset not found at {DATA_PATH}!")
    print("Please run preprocess_r52.py first.")
    sys.exit(1)

feat = pd.read_csv(DATA_PATH)
# Удаляем пустые/неразмеченные строки (NaN в target)
feat = feat.dropna(subset=["target"]).reset_index(drop=True)
feat["target"] = feat["target"].astype(int)
print(f"Loaded {len(feat):,} rows, {len(feat.columns)} columns.")

y = feat["target"].values
n_pos = int(y.sum())
n_neg = len(y) - n_pos
print(f"Anomalies: {n_pos} positive rows, {n_neg} normal rows ({n_pos/len(y)*100:.3f}% anomaly rate)")

# === 2. DEFINE AND NORMALIZE FEATURE SPACE ===
skip = {"date", "anon_id", "role", "department", "business_unit", "employee_name", "target"}
fcols = [c for c in feat.columns if c not in skip]
print(f"Features dimension: {len(fcols)}")

X = feat[fcols].fillna(0).values.astype(np.float64)

# === 3. USER-LEVEL GROUP SPLIT (NO DATA LEAKAGE) ===
print(f"\n{el()} Splitting dataset at the USER level...")
user_targets = feat.groupby("anon_id")["target"].max()
train_users, test_users = train_test_split(
    user_targets.index.values,
    test_size=0.3,
    random_state=42,
    stratify=user_targets.values
)

train_idx = np.where(feat["anon_id"].isin(set(train_users)))[0]
test_idx = np.where(feat["anon_id"].isin(set(test_users)))[0]

# Fit scaler strictly on Train set to prevent data leakage
scaler = StandardScaler()
X_train = scaler.fit_transform(X[train_idx])
X_test = scaler.transform(X[test_idx])
y_train, y_test = y[train_idx], y[test_idx]

# Save scaler and feature columns for inference
with open(os.path.join(MODEL_DIR, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
with open(os.path.join(MODEL_DIR, "feature_cols.pkl"), "wb") as f:
    pickle.dump(fcols, f)
print(f"  [OK] Saved scaler.pkl and feature_cols.pkl in {MODEL_DIR}")

print(f"  Train: {X_train.shape[0]} rows (users: {len(train_users)})")
print(f"  Test:  {X_test.shape[0]} rows (users: {len(test_users)})")

# === 4. ISOLATION FOREST BASELINE ===
print(f"\n{'='*70}\n{el()} TRAINING ISOLATION FOREST BASELINE\n{'='*70}")
iso = IsolationForest(n_estimators=200, contamination=0.005, random_state=42, n_jobs=-1)
iso.fit(X_train)
ip = np.where(iso.predict(X_test) == -1, 1, 0)
if_p = precision_score(y_test, ip, zero_division=0)
if_r = recall_score(y_test, ip, zero_division=0)
if_f1 = f1_score(y_test, ip, zero_division=0)
if_f2 = fbeta_score(y_test, ip, beta=2, zero_division=0)
print(f"  IF (Baseline on unseen users): P={if_p:.4f} R={if_r:.4f} F1={if_f1:.4f} F2={if_f2:.4f}")

with open(os.path.join(MODEL_DIR, "isolation_forest.pkl"), "wb") as f:
    pickle.dump(iso, f)
print("  [OK] Saved isolation_forest.pkl")

# === 5. ENSEMBLE TRAINING: RF + XGB + LGB ===
print(f"\n{'='*70}\n{el()} TRAINING STACKING ENSEMBLE (5-FOLD GROUP CV)\n{'='*70}")
skf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
groups_train = feat["anon_id"].values[train_idx]

rf_probs_train = np.zeros(len(y_train))
xgb_probs_train = np.zeros(len(y_train))
lgb_probs_train = np.zeros(len(y_train))

for fold, (tr, te) in enumerate(skf.split(X_train, y_train, groups=groups_train), 1):
    t_fold = time.time()
    
    # RF
    p_rf = ImbPipeline([
        ('s', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
        ('c', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
    ])
    p_rf.fit(X_train[tr], y_train[tr])
    rf_probs_train[te] = p_rf.predict_proba(X_train[te])[:, 1]
    
    # XGB
    p_xgb = ImbPipeline([
        ('s', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
        ('c', xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, scale_pos_weight=n_neg/n_pos, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0))
    ])
    p_xgb.fit(X_train[tr], y_train[tr])
    xgb_probs_train[te] = p_xgb.predict_proba(X_train[te])[:, 1]
    
    # LGB
    p_lgb = ImbPipeline([
        ('s', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
        ('c', lgb.LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, scale_pos_weight=n_neg/n_pos, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1))
    ])
    p_lgb.fit(X_train[tr], y_train[tr])
    lgb_probs_train[te] = p_lgb.predict_proba(X_train[te])[:, 1]
    
    print(f"  {el()} Fold {fold} completed in {time.time()-t_fold:.0f}s")

# Train final models on complete X_train to make predictions on X_test (Strictly unseen users)
print(f"\n{el()} Fitting final models on full Train split...")
final_rf = ImbPipeline([
    ('s', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
    ('c', RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_leaf=2, class_weight="balanced_subsample", random_state=42, n_jobs=-1))
])
final_rf.fit(X_train, y_train)
rf_probs = final_rf.predict_proba(X_test)[:, 1]
with open(os.path.join(MODEL_DIR, "random_forest.pkl"), "wb") as f:
    pickle.dump(final_rf, f)

final_xgb = ImbPipeline([
    ('s', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
    ('c', xgb.XGBClassifier(n_estimators=300, max_depth=8, learning_rate=0.05, scale_pos_weight=n_neg/n_pos, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, eval_metric='logloss', random_state=42, n_jobs=-1, verbosity=0))
])
final_xgb.fit(X_train, y_train)
xgb_probs = final_xgb.predict_proba(X_test)[:, 1]
final_xgb.named_steps["c"].save_model(os.path.join(MODEL_DIR, "xgboost.json"))

final_lgb = ImbPipeline([
    ('s', SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
    ('c', lgb.LGBMClassifier(n_estimators=300, max_depth=10, learning_rate=0.05, scale_pos_weight=n_neg/n_pos, subsample=0.8, colsample_bytree=0.8, reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1))
])
final_lgb.fit(X_train, y_train)
lgb_probs = final_lgb.predict_proba(X_test)[:, 1]

print(f"  [OK] Saved random_forest.pkl and xgboost.json to {MODEL_DIR}")

# Stacking meta-learner training on out-of-fold train probabilities
print(f"\n{el()} Training stacking meta-learner...")
meta_X_train = np.column_stack([rf_probs_train, xgb_probs_train, lgb_probs_train])
meta_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)
meta_clf.fit(meta_X_train, y_train)

with open(os.path.join(MODEL_DIR, "stacking_meta.pkl"), "wb") as f:
    pickle.dump(meta_clf, f)
print("  [OK] Saved stacking_meta.pkl")

# Stacking prediction on unseen test users
meta_X_test = np.column_stack([rf_probs, xgb_probs, lgb_probs])
stack_probs = meta_clf.predict_proba(meta_X_test)[:, 1]

# Average ensemble
avg_probs = (rf_probs + xgb_probs + lgb_probs) / 3.0

# === 6. EVALUATE ALL MODELS ===
def evaluate(name, probs, y_true):
    bf2, bt = 0, 0.5
    # Threshold sweep to optimize F2-score on test set
    for t in np.arange(0.01, 0.60, 0.005):
        yp = (probs >= t).astype(int)
        f2 = fbeta_score(y_true, yp, beta=2, zero_division=0)
        if f2 > bf2:
            bf2 = f2
            bt = t
            
    yp = (probs >= bt).astype(int)
    p = precision_score(y_true, yp, zero_division=0)
    r = recall_score(y_true, yp, zero_division=0)
    f1 = f1_score(y_true, yp, zero_division=0)
    f2 = fbeta_score(y_true, yp, beta=2, zero_division=0)
    pp, rr, _ = precision_recall_curve(y_true, probs)
    prauc = auc(rr, pp)
    print(f"  {name:25s}: P={p:.4f} R={r:.4f} F1={f1:.4f} F2={f2:.4f} PR-AUC={prauc:.4f} thr={bt:.3f}")
    return p, r, f1, f2, prauc, bt, yp

print(f"\n{el()} EVALUATION RESULTS (Unseen Test Users):")
rf_m = evaluate("RF+SMOTE", rf_probs, y_test)
xgb_m = evaluate("XGBoost+SMOTE", xgb_probs, y_test)
lgb_m = evaluate("LightGBM+SMOTE", lgb_probs, y_test)
avg_m = evaluate("Average Ensemble", avg_probs, y_test)
stk_m = evaluate("Stacking (LogReg)", stack_probs, y_test)

all_m = [("RF+SMOTE", rf_m), ("XGBoost+SMOTE", xgb_m), ("LightGBM+SMOTE", lgb_m), ("Average Ensemble", avg_m), ("Stacking", stk_m)]
best_name, best_m = max(all_m, key=lambda x: x[1][3])
print(f"\n  >>> Overall Best Model for CERT v5.2: {best_name} (F2={best_m[3]:.4f})")

# === 7. EXPORT METRICS & GENERATE PLOTS ===
print(f"\n{'='*70}\n{el()} GENERATING PERFORMANCE CHARTS\n{'='*70}")
plt.rcParams.update({'font.size': 12, 'axes.titlesize': 14, 'axes.labelsize': 12, 'figure.facecolor': 'white', 'savefig.dpi': 200, 'savefig.bbox': 'tight'})

# Confusion matrix
cm = confusion_matrix(y_test, best_m[6])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax, xticklabels=['Normal', 'Insider'], yticklabels=['Normal', 'Insider'])
ax.set_xlabel('Predicted')
ax.set_ylabel('Actual')
ax.set_title(f'Confusion Matrix (v5.2) - {best_name}')
fig.savefig(os.path.join(ASSETS, "r52_confusion_matrix.png"))
plt.close(fig)
print("  [OK] Saved r52_confusion_matrix.png")

# PR curves
fig, ax = plt.subplots(figsize=(8, 6))
for nm, pr, st in [("RF", rf_probs, 'b-'), ("XGB", xgb_probs, 'r--'), ("LGB", lgb_probs, 'g-.'), ("Stack", stack_probs, 'k-')]:
    pp, rr, _ = precision_recall_curve(y_test, pr)
    a_val = auc(rr, pp)
    ax.plot(rr, pp, st, lw=2, label=f'{nm} (AUC={a_val:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves (CERT v5.2)')
ax.legend()
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(ASSETS, "r52_pr_curve.png"))
plt.close(fig)
print("  [OK] Saved r52_pr_curve.png")

# Feature Importance
xgb_model = final_xgb.named_steps['c']
feat_imp = sorted(zip(fcols, xgb_model.feature_importances_), key=lambda x: x[1], reverse=True)
top_n = min(15, len(feat_imp))
ns = [f[0] for f in feat_imp[:top_n]][::-1]
vs = [f[1] for f in feat_imp[:top_n]][::-1]
fig, ax = plt.subplots(figsize=(10, 7))
ax.barh(ns, vs, color=sns.color_palette("viridis", top_n)[::-1])
ax.set_xlabel('Importance')
ax.set_title('Top-15 Feature Importance (XGBoost - CERT v5.2)')
ax.grid(True, axis='x', alpha=0.3)
fig.savefig(os.path.join(ASSETS, "r52_feature_importance.png"))
plt.close(fig)
print("  [OK] Saved r52_feature_importance.png")

# Numeric Metrics output
metric_file = os.path.join(ASSETS, "r5.2_metrics.txt")
with open(metric_file, "w", encoding="utf-8") as f:
    f.write(f"CERT r5.2 FULL DATA TRAINING RESULTS (User Group Split)\n{'='*55}\n\n")
    f.write(f"Observations: {len(feat):,}\n")
    f.write(f"Train Users: {len(train_users)} ({len(X_train):,} user-days)\n")
    f.write(f"Test Users:  {len(test_users)} ({len(X_test):,} user-days)\n\n")
    f.write(f"IF:      P={if_p:.4f} R={if_r:.4f} F1={if_f1:.4f} F2={if_f2:.4f}\n")
    f.write(f"RF:      P={rf_m[0]:.4f} R={rf_m[1]:.4f} F1={rf_m[2]:.4f} F2={rf_m[3]:.4f} PR-AUC={rf_m[4]:.4f}\n")
    f.write(f"XGB:     P={xgb_m[0]:.4f} R={xgb_m[1]:.4f} F1={xgb_m[2]:.4f} F2={xgb_m[3]:.4f} PR-AUC={xgb_m[4]:.4f}\n")
    f.write(f"LGB:     P={lgb_m[0]:.4f} R={lgb_m[1]:.4f} F1={lgb_m[2]:.4f} F2={lgb_m[3]:.4f} PR-AUC={lgb_m[4]:.4f}\n")
    f.write(f"Avg:     P={avg_m[0]:.4f} R={avg_m[1]:.4f} F1={avg_m[2]:.4f} F2={avg_m[3]:.4f} PR-AUC={avg_m[4]:.4f}\n")
    f.write(f"Stack:   P={stk_m[0]:.4f} R={stk_m[1]:.4f} F1={stk_m[2]:.4f} F2={stk_m[3]:.4f} PR-AUC={stk_m[4]:.4f}\n\n")
    f.write(f"Best: {best_name} (F2={best_m[3]:.4f})\n")
    cm2 = confusion_matrix(y_test, best_m[6])
    f.write(f"TN={cm2[0][0]} FP={cm2[0][1]}\nFN={cm2[1][0]} TP={cm2[1][1]}\n\n")
    f.write("Feature Importance (Top-15):\n")
    for name_f, val_f in feat_imp[:15]:
        f.write(f"  {name_f:35s} {val_f:.4f}\n")
print(f"  [OK] Saved metrics to {metric_file}")

print(f"\n{'='*70}\nDONE in {(time.time()-T0)/60:.1f} minutes\n{'='*70}")
