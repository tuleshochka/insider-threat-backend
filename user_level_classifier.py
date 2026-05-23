"""User-level classification: агрегируем признаки по пользователям -> классифицируем инсайдеров."""
import os; os.environ["DB_TYPE"] = "sqlite"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix, fbeta_score, make_scorer
from imblearn.over_sampling import SMOTE
import mlflow
from imblearn.pipeline import Pipeline as ImbPipeline
from app.services.preprocessor import DataPreprocessor
from app.config import settings

# ── 1. Загружаем признаки за весь год ──
print("Загрузка признаков...")
prep = DataPreprocessor()
features = prep.build_training_matrix("2010-01-01", "2010-12-31")
skip = {"date","anon_id","role","department","business_unit","employee_name"}
feat_cols = [c for c in features.columns if c not in skip]
print(f"  Всего записей пользователь-день: {len(features)}")
print(f"  Всего признаков: {len(feat_cols)}")

# ── 2. Агрегируем по пользователям ──
print("\nАгрегация признаков по пользователям...")
users_df = prep.load_users()
print(f"  Всего пользователей: {len(users_df)}")

# Для каждого пользователя считаем: mean, std, min, max, p50, p90 по каждому признаку
def agg_p90(x, axis=0): return np.percentile(x, 90, axis=axis)
agg_funcs = [np.mean, np.std, np.min, np.max, np.median, agg_p90]
agg_names = ["mean", "std", "min", "max", "p50", "p90"]

user_rows = []
for uid in users_df["anon_id"]:
    sub = features[features["anon_id"] == uid]
    if len(sub) == 0:
        user_rows.append(np.zeros(len(feat_cols) * len(agg_funcs)))
        continue
    vals = sub[feat_cols].values.astype(np.float64)
    row = np.concatenate([f(vals, axis=0) for f in agg_funcs])
    user_rows.append(row)

X = np.array(user_rows)
feature_names = [f"{c}_{a}" for a in agg_names for c in feat_cols]
print(f"  После агрегации: {X.shape[1]} признаков на пользователя")

# ── 3. Метки из ground truth ──
gt = pd.read_csv(settings.ground_truth_path)
r42 = gt[gt["dataset"] == 4.2]
insider_set = set(r42["user"])
y = np.array([1 if uid in insider_set else 0 for uid in users_df["anon_id"]])
print(f"  Инсайдеров: {y.sum()} / {len(y)} ({y.sum()/len(y)*100:.1f}%)")

# ── 4. Train/test split ──
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"\n  Train: {len(X_train)}, Test: {len(X_test)}")

# ── 5. RF без SMOTE (baseline) ──
print("\n" + "="*65)
print("RANDOM FOREST (без SMOTE)")
print("="*65)
rf = RandomForestClassifier(n_estimators=300, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred = rf.predict(X_test)
y_prob = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred, digits=4, target_names=["Benign", "Insider"]))
print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob):.4f}")

# ── 6. RF + SMOTE ──
print("\n" + "="*65)
print("RANDOM FOREST + SMOTE")
print("="*65)

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.start_run(run_name="User-Level RF+SMOTE")
mlflow.log_param("n_estimators", 300)
smote = SMOTE(random_state=42)
X_res, y_res = smote.fit_resample(X_train, y_train)
print(f"  После SMOTE: {X_res.shape[0]} samples ({int(y_res.sum())} positives)")
rf_smote = RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1)
rf_smote.fit(X_res, y_res)
y_pred_s = rf_smote.predict(X_test)
y_prob_s = rf_smote.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_s, digits=4, target_names=["Benign", "Insider"]))
print(f"  ROC-AUC: {roc_auc_score(y_test, y_prob_s):.4f}")

# ── 7. Cross-validation score ──
print("\n" + "="*65)
print("CROSS-VALIDATION (5-fold)")
print("="*65)
# Использование imblearn Pipeline предотвращает утечку данных (data leakage),
# так как SMOTE применяется только к обучающим фолдам внутри кросс-валидации.
pipeline = ImbPipeline([
    ('smote', SMOTE(random_state=42)),
    ('classifier', RandomForestClassifier(n_estimators=300, random_state=42, n_jobs=-1))
])
# Оцениваем F2-score (beta=2), который отдает больший вес полноте (Recall), 
# что критично для выявления угроз, где пропуск опаснее ложного срабатывания.
f2_scorer = make_scorer(fbeta_score, beta=2)
cv_scores = cross_val_score(pipeline, X_train, y_train, cv=5, scoring=f2_scorer)
print(f"  F2 scores: {cv_scores}")
print(f"  Mean F2: {cv_scores.mean():.4f} (+/- {cv_scores.std()*2:.4f})")

# ── 8. Лучший порог (Максимизация F2) ──
print("\n" + "="*65)
print("ПОДБОР ПОРОГА ДЛЯ F2-SCORE")
print("="*65)
best_f2 = 0
best_t = 0
for thresh in np.arange(0.1, 0.9, 0.05):
    y_t = (y_prob_s > thresh).astype(int)
    tp = int((y_t & y_test).sum())
    fp = int((y_t & (1-y_test)).sum())
    fn = int(y_test.sum()) - tp
    p = tp/max(tp+fp,1)
    r = tp/max(tp+fn,1)
    f1 = 2*p*r/max(p+r, 1e-10)
    f2 = (1 + 2**2) * p * r / max((2**2 * p) + r, 1e-10)
    
    if f2 > best_f2:
        best_f2 = f2
        best_t = thresh
        
    if np.isclose(thresh % 0.1, 0) and 0.2 < thresh < 0.8:
        print(f"  Threshold {thresh:.1f}: P={p:.4f}, R={r:.4f}, F1={f1:.4f}, F2={f2:.4f}, Flags={tp+fp}")

print(f"\n  => Оптимальный порог для F2: {best_t:.2f} (F2 = {best_f2:.4f})")

# ── 9. Важность признаков ──
print("\n" + "="*65)
print("ТОП-20 ВАЖНЫХ ПРИЗНАКОВ")
print("="*65)
imp = pd.DataFrame({"feature": feature_names, "importance": rf_smote.feature_importances_})
imp = imp.sort_values("importance", ascending=False).head(20)
print(imp.to_string(index=False))

# ── 10. Матрица ошибок ──
print("\n" + "="*65)
print("МАТРИЦА ОШИБОК (RF + SMOTE)")
print("="*65)
cm = confusion_matrix(y_test, y_pred_s)
print(f"          Predicted")
print(f"          Benign  Insider")
print(f"Actual")
print(f"  Benign   {cm[0,0]:4d}   {cm[0,1]:4d}")
print(f"  Insider  {cm[1,0]:4d}   {cm[1,1]:4d}")

mlflow.log_metrics({
    "cv_f2_mean": cv_scores.mean(),
    "best_f2_threshold": best_t,
    "best_f2_score": best_f2
})
mlflow.end_run()
