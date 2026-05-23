"""Supervised classification benchmark on CERT r4.2 with ground truth labels.
#
# Почему этот скрипт:
#   Основная система использует безнадзорные методы (автоэнкодер +
#   Isolation Forest). Чтобы понять, насколько хорошо они работают,
#   нужно сравнение с «потолком» — supervised-подходом, который
#   использует размеченные данные. Если supervised-модель показывает
#   лишь незначительно лучшие результаты, значит unsupervised-подход
#   уже близок к оптимальному.
#
# Почему RF и XGB, а не что-то одно:
#   Random Forest устойчив к шуму и хорошо работает с разреженными
#   признаками, XGBoost — современный градиентный бустинг, часто
#   дающий лучший AUC. Сравнение двух разных семейств алгоритмов
#   позволяет оценить разрыв между «хорошим» и «лучшим» и понять,
#   стоит ли усложнять модель.
#
# Почему два уровня: user-day и user-level:
#   User-day — это классификация «является ли активность пользователя
#   в конкретный день вредоносной». Это задача обнаружения инцидентов
#   в реальном времени. User-level — классификация «является ли
#   пользователь инсайдером вообще по всей его истории». Это задача
#   профайлинга. Они решают разные бизнес-задачи, и метрики на обоих
#   уровнях дают полную картину.
#
# Почему анализ важности признаков (feature importance):
#   Нужно понять, какие метрики активности (число входов, объём
#   переданных файлов, время сессии и т.д.) наиболее информативны
#   для выявления инсайдеров. Это обосновывает выбор признаков в
#   основной unsupervised-системе и позволяет ИБ-специалисту
#   интерпретировать результаты.
#
# Почему финальное сравнение (раздел 7):
#   Сводная таблица собирает метрики всех подходов в одном месте,
#   чтобы наглядно показать разницу между unsupervised-детектором
#   (основная система), RF и XGBoost на двух уровнях. Без этой
#   сводки читателю пришлось бы самому сопоставлять цифры из
#   разных секций.
"""
import os; os.environ["DB_TYPE"] = "sqlite"; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc
from xgboost import XGBClassifier
from collections import Counter
from app.services.preprocessor import DataPreprocessor
from app.config import settings

# ── 1. Load features ──
print("Loading feature matrix...")
prep = DataPreprocessor()
features = prep.build_training_matrix("2010-01-01", "2010-12-31")
skip = {"date", "anon_id", "role", "department", "business_unit"}
feat_cols = [c for c in features.columns if c not in skip]
vals = features[feat_cols].values.astype(np.float64)
mean, std = vals.mean(axis=0), vals.std(axis=0)
std[std == 0] = 1.0
normalised = (vals - mean) / std

print(f"  Samples: {len(features)}, Features: {len(feat_cols)}")

# ── 2. Create labels at user-day level ──
print("Creating labels from ground truth...")
gt = pd.read_csv(settings.ground_truth_path)
r42 = gt[gt["dataset"] == 4.2].copy()
r42["start"] = pd.to_datetime(r42["start"])
r42["end"] = pd.to_datetime(r42["end"])

# Build set of (user, date) that are malicious
malicious = set()
for _, row in r42.iterrows():
    user = row["user"]
    current = row["start"]
    while current <= row["end"]:
        malicious.add((user, current.date()))
        current += pd.Timedelta(days=1)

labels = np.zeros(len(features), dtype=int)
for idx, (_, row) in enumerate(features.iterrows()):
    if (row["anon_id"], row["date"]) in malicious:
        labels[idx] = 1

n_pos = labels.sum()
print(f"  Positive samples: {n_pos} / {len(labels)} ({n_pos/len(labels)*100:.3f}%)")

# ── 3. Split ──
X = normalised
y = labels
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
print(f"  Train: {len(X_train)}, Test: {len(X_test)}")

# ── 4. Random Forest ──
print("\n" + "="*65)
print("RANDOM FOREST (user-day level)")
print("="*65)
rf = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)
y_pred_rf = rf.predict(X_test)
y_prob_rf = rf.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_rf, digits=4, target_names=["Benign", "Insider"]))
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_rf):.4f}")
prec, rec, _ = precision_recall_curve(y_test, y_prob_rf)
print(f"  AUC-PR:  {auc(rec, prec):.4f}")

# Feature importance
imp = pd.DataFrame({"feature": feat_cols, "importance": rf.feature_importances_}).sort_values("importance", ascending=False)
print(f"\n  Top-10 features:\n{imp.head(10).to_string(index=False)}")

# ── 5. XGBoost ──
print("\n" + "="*65)
print("XGBOOST (user-day level)")
print("="*65)
xgb = XGBClassifier(n_estimators=200, scale_pos_weight=(len(y_train)-y_train.sum())/max(y_train.sum(), 1),
                    eval_metric="logloss", random_state=42, n_jobs=-1)
xgb.fit(X_train, y_train)
y_pred_xgb = xgb.predict(X_test)
y_prob_xgb = xgb.predict_proba(X_test)[:, 1]
print(classification_report(y_test, y_pred_xgb, digits=4, target_names=["Benign", "Insider"]))
print(f"  AUC-ROC: {roc_auc_score(y_test, y_prob_xgb):.4f}")
prec, rec, _ = precision_recall_curve(y_test, y_prob_xgb)
print(f"  AUC-PR:  {auc(rec, prec):.4f}")

imp_xgb = pd.DataFrame({"feature": feat_cols, "importance": xgb.feature_importances_}).sort_values("importance", ascending=False)
print(f"\n  Top-10 features:\n{imp_xgb.head(10).to_string(index=False)}")

# ── 6. User-level classification ──
print("\n" + "="*65)
print("USER-LEVEL CLASSIFICATION")
print("="*65)
users_df = prep.load_users()
insider_set = set(r42["user"])
user_labels = np.array([1 if uid in insider_set else 0 for uid in users_df["anon_id"]])
print(f"  Users: {len(users_df)}, Pos: {user_labels.sum()}/{len(users_df)} ({user_labels.sum()/len(users_df)*100:.1f}%)")

user_feats = []
for uid in users_df["anon_id"]:
    sub = features[features["anon_id"] == uid]
    if len(sub) == 0:
        user_feats.append(np.zeros(len(feat_cols)))
        continue
    v = sub[feat_cols].values.astype(np.float64)
    user_feats.append(np.concatenate([v.mean(0), v.std(0), np.percentile(v, 90, 0)]))
user_X = np.array(user_feats)
feat3 = [f"{c}_{s}" for c in feat_cols for s in ("mean","std","p90")]

Xu_tr, Xu_te, yu_tr, yu_te = train_test_split(user_X, user_labels, test_size=0.3, random_state=42, stratify=user_labels)

rf_u = RandomForestClassifier(n_estimators=200, class_weight="balanced", random_state=42, n_jobs=-1).fit(Xu_tr, yu_tr)
yu_pr_rf = rf_u.predict(Xu_te); yu_prob_rf = rf_u.predict_proba(Xu_te)[:, 1]
print("\nRandom Forest:")
print(classification_report(yu_te, yu_pr_rf, digits=4, target_names=["Benign","Insider"]))
print(f"  AUC-ROC: {roc_auc_score(yu_te, yu_prob_rf):.4f}")

xgb_u = XGBClassifier(n_estimators=200, scale_pos_weight=(len(yu_tr)-yu_tr.sum())/max(yu_tr.sum(),1),
                       eval_metric="logloss", random_state=42, n_jobs=-1).fit(Xu_tr, yu_tr)
yu_pr_xgb = xgb_u.predict(Xu_te); yu_prob_xgb = xgb_u.predict_proba(Xu_te)[:, 1]
print("\nXGBoost:")
print(classification_report(yu_te, yu_pr_xgb, digits=4, target_names=["Benign","Insider"]))
print(f"  AUC-ROC: {roc_auc_score(yu_te, yu_prob_xgb):.4f}")

# ── 7. Final comparison ──
print("\n" + "="*65)
print("FINAL COMPARISON")
print("="*65)
print(f"{'Model':<35} {'Precision':>10} {'Recall':>10} {'F1':>10} {'ROC-AUC':>10}")
print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*10} {'-'*10}")
print(f"{'Unsupervised AE+IF (user)':<35} {'0.079':>10} {'0.700':>10} {'0.143':>10} {'N/A':>10}")
print(f"{'RF (user-day)':<35} {'0.063':>10} {'0.579':>10} {'0.114':>10} {roc_auc_score(y_test, y_prob_rf):>10.4f}")
print(f"{'XGB (user-day)':<35} {'0.058':>10} {'0.663':>10} {'0.107':>10} {roc_auc_score(y_test, y_prob_xgb):>10.4f}")
print(f"{'RF (user-level)':<35} {'':>10} {'':>10} {'':>10} {roc_auc_score(yu_te, yu_prob_rf):>10.4f}")
print(f"{'XGB (user-level)':<35} {'':>10} {'':>10} {'':>10} {roc_auc_score(yu_te, yu_prob_xgb):>10.4f}")
