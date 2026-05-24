# ---
# jupyter:
#   jupytext:
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.16.0
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %% [markdown]
# # 🛡️ Insider Threat Detection — Обучение и оценка моделей
#
# **Магистерская диссертация**: «Разработка интеллектуальной системы мониторинга
# и анализа поведения пользователей для повышения уровня внутренней безопасности
# корпоративной сети»
#
# **Датасет**: [CERT Insider Threat Dataset r4.2](https://resources.sei.cmu.edu/library/asset-view.cfm?assetid=508099)
#
# ---
#
# ## Содержание
# 1. Подготовка окружения
# 2. Загрузка и предварительная обработка данных
# 3. Агрегация признаков (Feature Engineering)
# 4. Разметка ground truth
# 5. Обучение моделей
#    - 5.1 Isolation Forest (baseline)
#    - 5.2 Autoencoder (PyTorch)
#    - 5.3 Ансамбль: RF + XGBoost + LightGBM + Stacking
# 6. Оценка и визуализация результатов
# 7. SHAP-объяснения
# 8. Экспорт модели

# %% [markdown]
# ## 1. Подготовка окружения

# %%
# Для Google Colab — раскомментируйте следующую ячейку:
# !pip install torch torchvision xgboost lightgbm imbalanced-learn shap seaborn --quiet

# %%
import os
import sys
import time
import gc
import warnings
import pickle

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_curve,
    f1_score,
    fbeta_score,
    precision_score,
    recall_score,
    auc,
    roc_auc_score,
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline

import xgboost as xgb
import lightgbm as lgb

warnings.filterwarnings("ignore")
plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.facecolor": "white",
    "savefig.dpi": 200,
    "savefig.bbox": "tight",
})

print(f"PyTorch: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
T0 = time.time()

# %% [markdown]
# ## Настройка путей
#
# > **Для Google Colab**: загрузите датасет CERT r4.2 на Google Drive
# > и подключите его:
# > ```python
# > from google.colab import drive
# > drive.mount('/content/drive')
# > DATA_DIR = "/content/drive/MyDrive/cert_r4.2"
# > ```

# %%
# === ПУТЬ К ДАТАСЕТУ — ИЗМЕНИТЕ ПОД СВОЮ КОНФИГУРАЦИЮ ===
DATA_DIR = r"D:\Политех\Мага\Дипломы\М\12841247\r4.2"
GROUND_TRUTH_PATH = r"D:\Политех\Мага\Дипломы\М\12841247\answers\insiders.csv"
ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(".")), "assets")
os.makedirs(ASSETS_DIR, exist_ok=True)

# Период обучения
START_DATE = pd.to_datetime("2010-01-01")
END_DATE = pd.to_datetime("2011-05-01")

print(f"DATA_DIR: {DATA_DIR}")
print(f"Период: {START_DATE.date()} — {END_DATE.date()}")

# %% [markdown]
# ## 2. Загрузка и предварительная обработка данных
#
# Датасет CERT r4.2 содержит ~90 ГБ CSV-файлов. Загружаем чанками
# (по 500 000 строк) с фильтрацией по дате для экономии памяти.

# %%
def load_table(name: str, cols: list, dtypes: dict, start_dt, end_dt, chunksize=500_000):
    """Загружает CSV чанками с фильтрацией по дате."""
    path = os.path.join(DATA_DIR, f"{name}.csv")
    chunks = []
    total_read = 0

    print(f"  Загрузка {name}.csv ...", end=" ", flush=True)
    t = time.time()

    for chunk in pd.read_csv(path, usecols=cols, dtype=dtypes,
                              chunksize=chunksize, low_memory=False):
        total_read += len(chunk)
        chunk["date"] = pd.to_datetime(
            chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce"
        )
        chunk = chunk[(chunk["date"] >= start_dt) & (chunk["date"] < end_dt)]
        if not chunk.empty:
            chunks.append(chunk)

    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"{len(result):,} строк (из {total_read:,}), {time.time()-t:.0f}s")
    return result

# %%
print("=" * 60)
print("ЗАГРУЗКА ДАННЫХ")
print("=" * 60)

logon = load_table(
    "logon",
    ["date", "user", "pc", "activity", "id"],
    {"user": str, "pc": str, "activity": str, "id": str},
    START_DATE, END_DATE,
)
files = load_table(
    "file",
    ["date", "user", "pc", "filename", "id"],
    {"user": str, "pc": str, "filename": str, "id": str},
    START_DATE, END_DATE,
)
email = load_table(
    "email",
    ["date", "user", "to", "from", "size", "attachments", "id"],
    {"user": str, "to": str, "from": str, "size": "int64", "attachments": "int64", "id": str},
    START_DATE, END_DATE,
)
device = load_table(
    "device",
    ["date", "user", "pc", "activity", "id"],
    {"user": str, "pc": str, "activity": str, "id": str},
    START_DATE, END_DATE,
)

# HTTP загружаем с поагрегацией по чанкам (самая большая таблица)
print("  Загрузка http.csv (chunked aggregation) ...", end=" ", flush=True)
t = time.time()
http_aggs = []
http_total = 0
for chunk in pd.read_csv(
    os.path.join(DATA_DIR, "http.csv"),
    usecols=["date", "user", "url", "id"],
    dtype={"user": str, "url": str, "id": str},
    chunksize=1_000_000,
    low_memory=False,
):
    http_total += len(chunk)
    chunk["date"] = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
    chunk = chunk[(chunk["date"] >= START_DATE) & (chunk["date"] < END_DATE)]
    if chunk.empty:
        continue
    chunk["date_only"] = chunk["date"].dt.date
    chunk["hour"] = chunk["date"].dt.hour
    agg = chunk.groupby(["date_only", "user"], as_index=False).agg(
        http_requests=("id", "count"),
        http_unique_urls=("url", "nunique"),
        after_hours_http=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
    )
    http_aggs.append(agg)

if http_aggs:
    a_http = pd.concat(http_aggs, ignore_index=True)
    a_http = a_http.groupby(["date_only", "user"], as_index=False).agg(
        http_requests=("http_requests", "sum"),
        http_unique_urls=("http_unique_urls", "sum"),
        after_hours_http=("after_hours_http", "sum"),
    )
    a_http.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
    del http_aggs
else:
    a_http = pd.DataFrame()
print(f"done ({http_total:,}), {time.time()-t:.0f}s")
gc.collect()

# %% [markdown]
# ## 3. Агрегация признаков (Feature Engineering)
#
# Агрегируем сырые события в признаки на уровне **пользователь × день**.
# Это стандартный подход для UEBA-систем.

# %%
def agg_logon(df):
    """Логоны → 4 признака: количество, уникальные ПК, нерабочее время, выходные."""
    df["date_only"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour
    df["dow"] = df["date"].dt.dayofweek
    result = df.groupby(["date_only", "user"], as_index=False).agg(
        logon_count=("id", "count"),
        logon_unique_pc=("pc", "nunique"),
        after_hours_logons=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
        weekend_logons=("dow", lambda x: (x >= 5).sum()),
    )
    result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
    return result


def agg_file(df):
    """Файловые операции → 4 признака."""
    df["date_only"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour
    result = df.groupby(["date_only", "user"], as_index=False).agg(
        file_operations=("id", "count"),
        file_unique_pc=("pc", "nunique"),
        file_unique_names=("filename", "nunique"),
        after_hours_files=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
    )
    result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
    return result


def agg_email(df):
    """Email → 6 признаков (отправленные + полученные)."""
    df["date_only"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour
    sent = df[df["user"] == df["from"]].groupby(["date_only", "user"], as_index=False).agg(
        email_sent=("id", "count"),
        email_size_total=("size", "sum"),
        email_attachments=("attachments", "sum"),
        email_unique_recipients=("to", "nunique"),
        after_hours_email=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
    )
    received = df[df["user"] != df["from"]].groupby(["date_only", "user"], as_index=False).agg(
        email_received=("id", "count"),
    )
    result = sent.merge(received, on=["date_only", "user"], how="outer").fillna(0)
    result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
    return result


def agg_device(df):
    """USB/внешние устройства → 3 признака."""
    df["date_only"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour
    df["dow"] = df["date"].dt.dayofweek
    result = df.groupby(["date_only", "user"], as_index=False).agg(
        device_operations=("id", "count"),
        after_hours_device=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
        weekend_device=("dow", lambda x: (x >= 5).sum()),
    )
    result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
    return result


# %%
print("АГРЕГАЦИЯ ПРИЗНАКОВ")
print("=" * 60)

print("  logon ..."); a1 = agg_logon(logon); del logon
print("  file ...");  a2 = agg_file(files); del files
print("  email ..."); a3 = agg_email(email); del email
print("  device ...");a4 = agg_device(device); del device
gc.collect()

print("  merge ...")
feat = (
    a1.merge(a2, on=["date", "anon_id"], how="outer")
      .merge(a3, on=["date", "anon_id"], how="outer")
      .merge(a4, on=["date", "anon_id"], how="outer")
)
if not a_http.empty:
    feat = feat.merge(a_http, on=["date", "anon_id"], how="outer")
    del a_http
feat.fillna(0, inplace=True)
del a1, a2, a3, a4
gc.collect()

# Психометрические данные (Big Five)
psych_path = os.path.join(DATA_DIR, "psychometric.csv")
if os.path.exists(psych_path):
    psych = pd.read_csv(psych_path, usecols=["user_id", "O", "C", "E", "A", "N"])
    psych.rename(columns={"user_id": "anon_id"}, inplace=True)
    feat = feat.merge(psych, on="anon_id", how="left")
    feat[["O", "C", "E", "A", "N"]] = feat[["O", "C", "E", "A", "N"]].fillna(0)

print(f"  Базовая матрица: {len(feat):,} строк, {len(feat.columns)} столбцов")

# %% [markdown]
# ### Производные признаки
#
# Добавляем 5 производных признаков + user-baseline z-scores + временные лаги.

# %%
print("FEATURE ENGINEERING (производные признаки)")
print("=" * 60)

# 5 производных
feat["after_hours_ratio"] = (
    feat["after_hours_logons"] + feat["after_hours_files"] +
    feat["after_hours_device"] + feat.get("after_hours_http", 0) +
    feat["after_hours_email"]
) / (
    feat["logon_count"] + feat["file_operations"] +
    feat["device_operations"] + feat.get("http_requests", 0) +
    feat["email_sent"] + 1
)
feat["device_to_file_ratio"] = feat["device_operations"] / (feat["file_operations"] + 1)
feat["email_size_per_msg"] = feat["email_size_total"] / (feat["email_sent"] + 1)
feat["files_per_pc"] = feat["file_operations"] / (feat["file_unique_pc"] + 1)
feat["weekend_activity"] = feat["weekend_logons"] + feat["weekend_device"]

# User-baseline z-scores (ключевая идея UEBA)
print("  User-baseline z-scores ...")
activity_cols = [
    "logon_count", "file_operations", "email_sent", "device_operations",
    "after_hours_logons", "after_hours_files", "after_hours_device", "weekend_logons",
]
if "http_requests" in feat.columns:
    activity_cols.append("http_requests")

feat["date"] = pd.to_datetime(feat["date"])
feat.sort_values(["anon_id", "date"], inplace=True)

for col in activity_cols:
    u_mean = feat.groupby("anon_id")[col].transform("mean")
    u_std = feat.groupby("anon_id")[col].transform("std").replace(0, 1)
    feat[f"{col}_zscore"] = (feat[col] - u_mean) / u_std

# Rolling lag features
print("  Rolling lag features ...")
key_cols = ["logon_count", "file_operations", "device_operations", "email_sent"]
if "http_requests" in feat.columns:
    key_cols.append("http_requests")

for col in key_cols:
    g = feat.groupby("anon_id")[col]
    feat[f"{col}_lag1"] = g.shift(1).fillna(0)
    feat[f"{col}_lag3_avg"] = (g.shift(1).fillna(0) + g.shift(2).fillna(0) + g.shift(3).fillna(0)) / 3
    feat[f"{col}_diff"] = g.diff().fillna(0)

print(f"  Итого: {len(feat):,} строк, {len(feat.columns)} столбцов")

# %% [markdown]
# ## 4. Разметка Ground Truth
#
# Используем файл `insiders.csv` из CERT r4.2, содержащий информацию
# о реальных инсайдерских инцидентах.

# %%
print("GROUND TRUTH")
print("=" * 60)

gt = pd.read_csv(GROUND_TRUTH_PATH)
r42 = gt[gt["dataset"] == 4.2].copy()
r42["start"] = pd.to_datetime(r42["start"])
r42["end"] = pd.to_datetime(r42["end"])

# Создаём множество (user, date) для всех дней инсайдерской активности
malicious = set()
for _, row in r42.iterrows():
    cur = row["start"]
    while cur <= row["end"]:
        malicious.add((row["user"], cur.date()))
        cur += pd.Timedelta(days=1)

y = np.array([
    1 if (r["anon_id"], r["date"].date() if hasattr(r["date"], "date") else r["date"]) in malicious
    else 0
    for _, r in feat.iterrows()
], dtype=np.int32)

n_pos = int(y.sum())
n_neg = len(y) - n_pos
print(f"  Позитивные (инсайдеры): {n_pos}")
print(f"  Негативные (норма):     {n_neg}")
print(f"  Соотношение:            {n_pos/len(y)*100:.3f}%")

# %%
# Подготовка матрицы признаков для ML
skip_cols = {"date", "anon_id", "role", "department", "business_unit",
             "employee_name", "functional_unit", "start_date", "end_date"}
feature_cols = [c for c in feat.columns
                if c not in skip_cols and feat[c].dtype in ["float64", "int64", "float32", "int32"]]
X = feat[feature_cols].fillna(0).values.astype(np.float64)

scaler = StandardScaler()
X = scaler.fit_transform(X)

print(f"  Признаков: {len(feature_cols)}")
print(f"  Матрица: {X.shape}")

# %% [markdown]
# ## 5. Обучение моделей
#
# ### 5.1 Isolation Forest (baseline)
#
# Неконтролируемый (unsupervised) baseline — не использует метки.

# %%
print("=" * 60)
print("ISOLATION FOREST (baseline)")
print("=" * 60)

X_train_if, X_test_if, y_train_if, y_test_if = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

iso_forest = IsolationForest(
    n_estimators=200,
    contamination=0.005,  # ~0.5% аномалий
    random_state=42,
    n_jobs=-1,
)
iso_forest.fit(X_train_if)

# Предсказания: -1 = аномалия, 1 = норма
if_pred = np.where(iso_forest.predict(X_test_if) == -1, 1, 0)
if_p = precision_score(y_test_if, if_pred, zero_division=0)
if_r = recall_score(y_test_if, if_pred, zero_division=0)
if_f1 = f1_score(y_test_if, if_pred, zero_division=0)
if_f2 = fbeta_score(y_test_if, if_pred, beta=2, zero_division=0)

print(f"  Precision: {if_p:.4f}")
print(f"  Recall:    {if_r:.4f}")
print(f"  F1:        {if_f1:.4f}")
print(f"  F2:        {if_f2:.4f}")

# %% [markdown]
# ### 5.2 Autoencoder (PyTorch)
#
# Обучаем автокодировщик для unsupervised anomaly detection.
# Модель учится восстанавливать **нормальное** поведение;
# аномалии дают высокую ошибку реконструкции (MSE).

# %%
class Autoencoder(nn.Module):
    """Undercomplete autoencoder с bottleneck = input_dim // 2."""

    def __init__(self, input_dim: int, hidden_dim: int | None = None):
        super().__init__()
        hidden = hidden_dim or max(input_dim // 2, 4)
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden, input_dim),
            # ВАЖНО: НЕ используем Sigmoid, т.к. данные нормализованы Z-score
            # и содержат отрицательные значения. Sigmoid ограничит выход [0,1].
        )

    def forward(self, x):
        return self.decoder(self.encoder(x))


def train_autoencoder(model, train_loader, val_loader=None, epochs=50, lr=0.001, patience=7):
    """Обучение AE с early stopping."""
    model.to(DEVICE)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    history = {"train_loss": [], "val_loss": []}
    best_val = float("inf")
    stale = 0

    for epoch in range(1, epochs + 1):
        model.train()
        train_loss = 0.0
        for (batch,) in train_loader:
            batch = batch.to(DEVICE)
            recon = model(batch)
            loss = criterion(recon, batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item() * batch.size(0)

        train_loss /= len(train_loader.dataset)
        history["train_loss"].append(train_loss)

        if val_loader:
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for (batch,) in val_loader:
                    batch = batch.to(DEVICE)
                    recon = model(batch)
                    val_loss += criterion(recon, batch).item() * batch.size(0)
            val_loss /= len(val_loader.dataset)
            history["val_loss"].append(val_loss)

            if val_loss < best_val:
                best_val = val_loss
                stale = 0
            else:
                stale += 1
                if stale >= patience:
                    print(f"  Early stopping at epoch {epoch}")
                    break

        if epoch % 10 == 0:
            vl = f", val_loss={history['val_loss'][-1]:.6f}" if val_loader else ""
            print(f"  Epoch {epoch:3d}: train_loss={train_loss:.6f}{vl}")

    model.to("cpu")
    return history


# %%
print("=" * 60)
print("AUTOENCODER")
print("=" * 60)

data_tensor = torch.tensor(X, dtype=torch.float32)
n = len(data_tensor)
val_n = int(n * 0.1)
indices = torch.randperm(n)  # Перемешиваем!
train_data = data_tensor[indices[val_n:]]
val_data = data_tensor[indices[:val_n]]

train_loader = DataLoader(TensorDataset(train_data), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(val_data), batch_size=256)

ae_model = Autoencoder(input_dim=X.shape[1])
ae_history = train_autoencoder(ae_model, train_loader, val_loader, epochs=100, lr=0.001, patience=15)

# Визуализация learning curve
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(ae_history["train_loss"], label="Train Loss")
if ae_history["val_loss"]:
    ax.plot(ae_history["val_loss"], label="Validation Loss")
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss")
ax.set_title("Autoencoder Learning Curve")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()
fig.savefig(os.path.join(ASSETS_DIR, "ae_learning_curve.png"))
plt.show()

# %% [markdown]
# ### 5.3 Ансамбль: RF + XGBoost + LightGBM → Stacking (LogReg)
#
# Основная модель — **ансамбль с учителем** (supervised).
# Используем 5-Fold Stratified CV с SMOTE для балансировки классов.

# %%
print("=" * 60)
print("STACKING ENSEMBLE (RF + XGB + LGB → LogReg)")
print("=" * 60)

skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

rf_probs = np.zeros(len(y))
xgb_probs = np.zeros(len(y))
lgb_probs = np.zeros(len(y))

for fold, (train_idx, test_idx) in enumerate(skf.split(X, y), 1):
    t_fold = time.time()

    # Random Forest + SMOTE
    pipe_rf = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
        ("clf", RandomForestClassifier(
            n_estimators=500, max_depth=25, min_samples_leaf=2,
            class_weight="balanced_subsample", random_state=42, n_jobs=-1,
        )),
    ])
    pipe_rf.fit(X[train_idx], y[train_idx])
    rf_probs[test_idx] = pipe_rf.predict_proba(X[test_idx])[:, 1]

    # XGBoost + SMOTE
    pipe_xgb = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
        ("clf", xgb.XGBClassifier(
            n_estimators=500, max_depth=8, learning_rate=0.05,
            scale_pos_weight=n_neg / n_pos, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, eval_metric="logloss",
            random_state=42, n_jobs=-1, verbosity=0,
        )),
    ])
    pipe_xgb.fit(X[train_idx], y[train_idx])
    xgb_probs[test_idx] = pipe_xgb.predict_proba(X[test_idx])[:, 1]

    # LightGBM + SMOTE
    pipe_lgb = ImbPipeline([
        ("smote", SMOTE(sampling_strategy=0.5, random_state=42, k_neighbors=2)),
        ("clf", lgb.LGBMClassifier(
            n_estimators=500, max_depth=10, learning_rate=0.05,
            scale_pos_weight=n_neg / n_pos, subsample=0.8, colsample_bytree=0.8,
            reg_alpha=0.1, reg_lambda=1.0, random_state=42, n_jobs=-1, verbose=-1,
        )),
    ])
    pipe_lgb.fit(X[train_idx], y[train_idx])
    lgb_probs[test_idx] = pipe_lgb.predict_proba(X[test_idx])[:, 1]

    print(f"  Fold {fold}: {time.time() - t_fold:.0f}s")

# Stacking meta-learner
print("  Training stacking meta-learner ...")
meta_X = np.column_stack([rf_probs, xgb_probs, lgb_probs])
meta_clf = LogisticRegression(class_weight="balanced", max_iter=1000, random_state=42)

stack_probs = np.zeros(len(y))
for train_idx, test_idx in skf.split(meta_X, y):
    meta_clf.fit(meta_X[train_idx], y[train_idx])
    stack_probs[test_idx] = meta_clf.predict_proba(meta_X[test_idx])[:, 1]

# Simple average ensemble
avg_probs = (rf_probs + xgb_probs + lgb_probs) / 3.0


# %% [markdown]
# ### 5.4 Многоканальная нейросеть (Multi-Input Classifier + Focal Loss)
#
# Нейросетевое ядро, основанное на глубоком обучении, с раздельными ветвями признаков.
# Каждая поведенческая плоскость (интенсивность, разнообразие, временные паттерны, психометрика)
# обрабатывается своей подсетью, затем выходы объединяются через механизм внимания (Attention Fusion).
# Функция потерь Focal Loss решает проблему сильного дисбаланса классов.

# %%
print("=" * 60)
print("MULTI-INPUT CLASSIFIER (Deep Learning Core)")
print("=" * 60)

class FocalLoss(nn.Module):
    """Функция потерь Focal Loss для борьбы с сильным дисбалансом классов."""
    def __init__(self, alpha: float = 0.75, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce = F.binary_cross_entropy_with_logits(inputs, targets, reduction="none")
        pt = torch.exp(-bce)
        focal = self.alpha * (1 - pt) ** self.gamma * bce
        weights = targets * self.alpha + (1 - targets) * (1 - self.alpha)
        return (focal * weights).mean()


class Branch(nn.Module):
    """Ветвь обработки одной группы признаков."""
    def __init__(self, in_dim: int, hidden: int = 16, out_dim: int = 8):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden, out_dim),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AttentionFusion(nn.Module):
    """Внимательное слияние представлений ветвей (attention fusion)."""
    def __init__(self, branch_names: list, latent_dim: int = 8):
        super().__init__()
        self.branch_names = branch_names
        n = len(branch_names)
        self.attn_project = nn.Linear(latent_dim, 1, bias=False)
        total = n * latent_dim + latent_dim
        self.fusion = nn.Sequential(
            nn.Linear(total, 32),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(32, 1),
        )

    def forward(self, latents: list) -> tuple:
        stacked = torch.stack(latents, dim=1)
        scores = self.attn_project(stacked).squeeze(-1)
        attn_weights = F.softmax(scores, dim=1)
        weighted = (attn_weights.unsqueeze(-1) * stacked).sum(dim=1)
        concat = stacked.view(stacked.size(0), -1)
        fused = torch.cat([weighted, concat], dim=1)
        return self.fusion(fused), attn_weights


class MultiInputClassifier(nn.Module):
    """Полная модель: несколько ветвей + внимание + классификатор."""
    def __init__(self, branches: dict, latent_dim: int = 8):
        super().__init__()
        self.branch_names = list(branches.keys())
        self.branch_modules = nn.ModuleDict({
            name: Branch(in_dim, hidden=max(in_dim * 2, 16), out_dim=latent_dim)
            for name, in_dim in branches.items()
        })
        self.fusion = AttentionFusion(self.branch_names, latent_dim=latent_dim)

    def forward(self, inputs: dict) -> torch.Tensor:
        latents = [self.branch_modules[name](inputs[name]) for name in self.branch_names]
        logits, attn_weights = self.fusion(latents)
        return logits

    def predict_proba(self, inputs: dict) -> torch.Tensor:
        return torch.sigmoid(self.forward(inputs))

    def get_attention(self, inputs: dict) -> tuple:
        latents = [self.branch_modules[name](inputs[name]) for name in self.branch_names]
        logits, attn_weights = self.fusion(latents)
        return logits, attn_weights

# Импортируем functional для FocalLoss
import torch.nn.functional as F

BRANCHES = {
    "intensity": ["logon_count", "file_operations", "email_sent", "email_received",
                  "device_operations", "http_requests", "email_attachments", "email_size_total"],
    "diversity": ["logon_unique_pc", "file_unique_pc", "file_unique_names",
                  "email_unique_recipients", "http_unique_urls"],
    "temporal":  ["after_hours_logons", "after_hours_files", "after_hours_email",
                  "after_hours_device", "after_hours_http", "weekend_logons", "weekend_device"],
    "psychometric": ["O", "C", "E", "A", "N"],
}

# Гарантируем, что все необходимые колонки есть в датафрейме
available_cols = set(feat.columns)
for name, cols in BRANCHES.items():
    for c in cols:
        if c not in available_cols:
            feat[c] = 0.0

# Разделение на train/val/test
train_idx, test_idx = train_test_split(np.arange(len(y)), test_size=0.3, random_state=42, stratify=y)
tr_idx, val_idx = train_test_split(train_idx, test_size=0.2, random_state=42, stratify=y[train_idx])

train_data = {}
val_data = {}
test_data = {}
for name, cols in BRANCHES.items():
    v = feat[cols].values.astype(np.float64)
    m, s = v[tr_idx].mean(axis=0), v[tr_idx].std(axis=0)
    s[s == 0] = 1.0
    v_norm = (v - m) / s
    train_data[name] = v_norm[tr_idx]
    val_data[name] = v_norm[val_idx]
    test_data[name] = v_norm[test_idx]

# Применяем SMOTE на обучающих данных
X_flat = np.column_stack([train_data[n] for n in BRANCHES])
smote = SMOTE(random_state=42, k_neighbors=2, sampling_strategy=0.2)
X_res, y_res = smote.fit_resample(X_flat, y[tr_idx])

# Разрезаем ресемплированные данные обратно по ветвям
branch_dims = [len(BRANCHES[n]) for n in BRANCHES]
offsets = np.cumsum([0] + branch_dims)
train_data_res = {}
for i, name in enumerate(BRANCHES):
    train_data_res[name] = X_res[:, offsets[i]:offsets[i+1]]

# Инициализируем и обучаем модель
mi_model = MultiInputClassifier({n: len(BRANCHES[n]) for n in BRANCHES}, latent_dim=8)
optimizer_mi = torch.optim.Adam(mi_model.parameters(), lr=0.001)
criterion_mi = FocalLoss(alpha=0.75, gamma=2.0)
batch_size_mi = 256
epochs_mi = 100
patience_mi = 15
best_val_loss = float("inf")
stale_mi = 0
best_weights = None

print("  Обучение Multi-Input классификатора с Focal Loss ...")
for epoch in range(1, epochs_mi + 1):
    mi_model.train()
    perm = np.random.permutation(len(y_res))
    total_loss = 0.0
    for i in range(0, len(perm), batch_size_mi):
        idx = perm[i:i+batch_size_mi]
        inp = {n: torch.tensor(train_data_res[n][idx], dtype=torch.float32) for n in BRANCHES}
        target = torch.tensor(y_res[idx], dtype=torch.float32).unsqueeze(1)
        pred = mi_model(inp)
        loss = criterion_mi(pred, target)
        
        optimizer_mi.zero_grad()
        loss.backward()
        optimizer_mi.step()
        
        total_loss += loss.item() * len(idx)
        
    # Валидация на реальных данных
    mi_model.eval()
    with torch.no_grad():
        inp_val = {n: torch.tensor(val_data[n], dtype=torch.float32) for n in BRANCHES}
        target_val = torch.tensor(y[val_idx], dtype=torch.float32).unsqueeze(1)
        val_loss = criterion_mi(mi_model(inp_val), target_val).item()
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_weights = {k: v.clone() for k, v in mi_model.state_dict().items()}
            stale_mi = 0
        else:
            stale_mi += 1
            
    if epoch % 5 == 0 or stale_mi == 0:
        # AUC на тесте
        with torch.no_grad():
            inp_test = {n: torch.tensor(test_data[n], dtype=torch.float32) for n in BRANCHES}
            vp = torch.sigmoid(mi_model(inp_test)).squeeze().numpy()
            vauc = roc_auc_score(y[test_idx], vp)
        print(f"    Эпоха {epoch:2d}: loss={total_loss/len(perm):.4f}, val_loss={val_loss:.4f}, test-AUC={vauc:.4f} {'(New Best)' if stale_mi==0 else ''}")
        
    if stale_mi >= patience_mi:
        print(f"    Ранняя остановка на эпохе {epoch}")
        break

# Восстанавливаем лучшие веса
if best_weights is not None:
    mi_model.load_state_dict(best_weights)

# Вычисляем вероятности для всего датасета (используя обучающие статистики)
all_data = {}
for name, cols in BRANCHES.items():
    v = feat[cols].values.astype(np.float64)
    m, s = v[tr_idx].mean(axis=0), v[tr_idx].std(axis=0)
    s[s == 0] = 1.0
    v_norm = (v - m) / s
    all_data[name] = torch.tensor(v_norm, dtype=torch.float32)

mi_model.eval()
with torch.no_grad():
    mi_probs = torch.sigmoid(mi_model(all_data)).squeeze().numpy()


# %% [markdown]
# ### 5.5 Двухуровневый гибридный пайплайн (AE + IF -> Multi-Input Classifier)
#
# Первым уровнем (ансамбль Autoencoder + Isolation Forest) отфильтровываются наиболее
# вероятные нормальные образцы, пропуская только top-20% подозрительных.
# Второй уровень (Multi-Input Classifier) верифицирует отобранные события.

# %%
print("=" * 60)
print("TWO-LEVEL HYBRID PIPELINE (AE + IF -> Multi-Input)")
print("=" * 60)

# 1. Считаем скоры Isolation Forest для всех точек
if_scores = 1 - (iso_forest.decision_function(X) - iso_forest.decision_function(X).min()) / (
    iso_forest.decision_function(X).max() - iso_forest.decision_function(X).min() + 1e-10
)

# 2. Считаем скоры Autoencoder (MSE) для всех точек
ae_model.eval()
with torch.no_grad():
    recon_all = ae_model(torch.tensor(X, dtype=torch.float32))
    ae_scores = torch.mean((torch.tensor(X, dtype=torch.float32) - recon_all)**2, dim=1).numpy()

ae_norm = (ae_scores - ae_scores.min()) / (ae_scores.max() - ae_scores.min() + 1e-10)

# 3. Объединяем скоры через среднее геометрическое
ensemble_scores = np.sqrt(ae_norm * if_scores)

# 4. Пропускаем только top-20%
top_k_all = int(len(ensemble_scores) * 0.20)
keep_idx_all = np.argsort(-ensemble_scores)[:top_k_all]
l1_pass_all = np.zeros(len(X), dtype=np.int32)
l1_pass_all[keep_idx_all] = 1

# 5. Итоговый скор двух уровней: если точка не прошла первый уровень, то скор = 0,
# иначе — берем вероятность от Multi-Input модели
two_level_probs = np.where(l1_pass_all == 1, mi_probs, 0.0)

print(f"  Первый уровень отсеял {len(X) - top_k_all:,} нормальных точек.")
print(f"  Оставлено {top_k_all:,} точек для верификации нейросетью.")


# %% [markdown]
# ## 6. Оценка и визуализация результатов

# %%
def evaluate_model(name, probs, y_true):
    """Оценка модели с подбором оптимального порога по F2."""
    best_f2, best_threshold = 0, 0.5
    for t in np.arange(0.01, 0.60, 0.005):
        yp = (probs >= t).astype(int)
        f2 = fbeta_score(y_true, yp, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_threshold = t

    y_pred = (probs >= best_threshold).astype(int)
    p = precision_score(y_true, y_pred, zero_division=0)
    r = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    f2 = fbeta_score(y_true, y_pred, beta=2, zero_division=0)
    pr_p, pr_r, _ = precision_recall_curve(y_true, probs)
    pr_auc = auc(pr_r, pr_p)

    print(f"  {name:25s}: P={p:.4f}  R={r:.4f}  F1={f1:.4f}  F2={f2:.4f}  "
          f"PR-AUC={pr_auc:.4f}  thr={best_threshold:.3f}")

    return p, r, f1, f2, pr_auc, best_threshold, y_pred


print("=" * 60)
print("РЕЗУЛЬТАТЫ")
print("=" * 60)

if_probs = 1 - (iso_forest.decision_function(X) - iso_forest.decision_function(X).min()) / (
    iso_forest.decision_function(X).max() - iso_forest.decision_function(X).min() + 1e-10
)
if_metrics  = evaluate_model("Isolation Forest", if_probs, y)
rf_metrics  = evaluate_model("RF+SMOTE", rf_probs, y)
xgb_metrics = evaluate_model("XGBoost+SMOTE", xgb_probs, y)
lgb_metrics = evaluate_model("LightGBM+SMOTE", lgb_probs, y)
avg_metrics = evaluate_model("Average Ensemble", avg_probs, y)
stk_metrics = evaluate_model("Stacking (LogReg)", stack_probs, y)
ae_metrics  = evaluate_model("Autoencoder", ae_norm, y)
mi_metrics  = evaluate_model("Multi-Input Network", mi_probs, y)
tl_metrics  = evaluate_model("Two-Level Pipeline", two_level_probs, y)

# Определение лучшей модели
all_models = [
    ("Isolation Forest", if_metrics),
    ("Autoencoder", ae_metrics),
    ("RF+SMOTE", rf_metrics),
    ("XGBoost+SMOTE", xgb_metrics),
    ("LightGBM+SMOTE", lgb_metrics),
    ("Average Ensemble", avg_metrics),
    ("Stacking", stk_metrics),
    ("Multi-Input Network", mi_metrics),
    ("Two-Level Pipeline", tl_metrics),
]
best_name, best_metrics = max(all_models, key=lambda x: x[1][3])
print(f"\n  >>> Лучшая модель: {best_name} (F2 = {best_metrics[3]:.4f})")

# %% [markdown]
# ### Confusion Matrix

# %%
cm = confusion_matrix(y, best_metrics[6])
fig, ax = plt.subplots(figsize=(7, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
            xticklabels=["Normal", "Insider"],
            yticklabels=["Normal", "Insider"])
ax.set_xlabel("Predicted")
ax.set_ylabel("Actual")
ax.set_title(f"Confusion Matrix — {best_name} (5-Fold CV)")
fig.savefig(os.path.join(ASSETS_DIR, "confusion_matrix.png"))
plt.show()

print(f"  TN={cm[0][0]:,}  FP={cm[0][1]:,}")
print(f"  FN={cm[1][0]:,}  TP={cm[1][1]:,}")

# %% [markdown]
# ### PR-кривые

# %%
fig, ax = plt.subplots(figsize=(10, 7))
for name, probs, style in [
    ("RF", rf_probs, "b-"),
    ("XGBoost", xgb_probs, "r--"),
    ("LightGBM", lgb_probs, "g-."),
    ("Stacking", stack_probs, "k-"),
    ("Autoencoder", ae_norm, "y:"),
    ("Multi-Input", mi_probs, "m-"),
    ("Two-Level", two_level_probs, "c-."),
]:
    pr_p, pr_r, _ = precision_recall_curve(y, probs)
    pr_auc_val = auc(pr_r, pr_p)
    ax.plot(pr_r, pr_p, style, lw=2, label=f"{name} (AUC={pr_auc_val:.3f})")

ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curves")
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
fig.savefig(os.path.join(ASSETS_DIR, "pr_curve.png"))
plt.show()

# %% [markdown]
# ### Сравнение метрик всех моделей

# %%
model_names = ["IF", "AE", "RF", "XGB", "LGB", "Avg", "Stack", "Multi-Input", "Two-Level"]
x_pos = np.arange(len(model_names))
width = 0.15

ps = [if_metrics[0], ae_metrics[0], rf_metrics[0], xgb_metrics[0], lgb_metrics[0], avg_metrics[0], stk_metrics[0], mi_metrics[0], tl_metrics[0]]
rs = [if_metrics[1], ae_metrics[1], rf_metrics[1], xgb_metrics[1], lgb_metrics[1], avg_metrics[1], stk_metrics[1], mi_metrics[1], tl_metrics[1]]
f1s = [if_metrics[2], ae_metrics[2], rf_metrics[2], xgb_metrics[2], lgb_metrics[2], avg_metrics[2], stk_metrics[2], mi_metrics[2], tl_metrics[2]]
f2s = [if_metrics[3], ae_metrics[3], rf_metrics[3], xgb_metrics[3], lgb_metrics[3], avg_metrics[3], stk_metrics[3], mi_metrics[3], tl_metrics[3]]

fig, ax = plt.subplots(figsize=(16, 6))
ax.bar(x_pos - 1.5 * width, ps, width, label="Precision", color="#3498db")
ax.bar(x_pos - 0.5 * width, rs, width, label="Recall", color="#e74c3c")
ax.bar(x_pos + 0.5 * width, f1s, width, label="F1", color="#2ecc71")
ax.bar(x_pos + 1.5 * width, f2s, width, label="F2", color="#9b59b6")

ax.set_ylabel("Score")
ax.set_title("Сравнение моделей: Precision, Recall, F1, F2")
ax.set_xticks(x_pos)
ax.set_xticklabels(model_names)
ax.legend()
ax.set_ylim(0, 1.05)
ax.grid(True, axis="y", alpha=0.3)
for container in ax.containers:
    ax.bar_label(container, fmt="%.3f", padding=2, fontsize=7)

fig.savefig(os.path.join(ASSETS_DIR, "metrics_comparison.png"))
plt.show()


# %% [markdown]
# ### Оценка на уровне пользователей (User-Level Evaluation)
#
# В реальных условиях ИБ-аналитикам SOC важнее классифицировать не отдельные дни,
# а выявлять конкретных сотрудников-нарушителей по их истории (User-Level).
# Мы агрегируем посуточные предсказания каждого алгоритма по каждому пользователю
# (выбирая максимальный риск-балл за весь период наблюдения) и сравниваем с
# реальным списком инсайдеров.

# %%
print("=" * 60)
print("ОЦЕНКА НА УРОВНЕ ПОЛЬЗОВАТЕЛЕЙ (USER-LEVEL)")
print("=" * 60)

# Загружаем список всех пользователей
users_csv_path = os.path.join(DATA_DIR, "users.csv")
if os.path.exists(users_csv_path):
    users_df = pd.read_csv(users_csv_path)
    # Предполагаем, что колонка user_id содержит anon_id
    users_df.rename(columns={"user_id": "anon_id"}, inplace=True)
else:
    users_df = pd.DataFrame({"anon_id": feat["anon_id"].unique()})

insider_set = set(r42["user"])
y_true_user = np.array([1 if uid in insider_set else 0 for uid in users_df["anon_id"]])

def evaluate_user_level(name, probs_day, feat_df, y_true_u, users_list):
    # Картируем вероятности дней на пользователей
    user_probs = {}
    for i, (_, row) in enumerate(feat_df.iterrows()):
        uid = row["anon_id"]
        user_probs[uid] = max(user_probs.get(uid, 0.0), probs_day[i])
        
    y_prob_u = np.array([user_probs.get(uid, 0.0) for uid in users_list["anon_id"]])
    
    # Подбираем оптимальный порог по F2 на уровне пользователей
    best_f2, best_thr = 0.0, 0.5
    for t in np.arange(0.01, 0.95, 0.01):
        yp = (y_prob_u >= t).astype(int)
        f2 = fbeta_score(y_true_u, yp, beta=2, zero_division=0)
        if f2 > best_f2:
            best_f2 = f2
            best_thr = t
            
    y_pred_u = (y_prob_u >= best_thr).astype(int)
    p = precision_score(y_true_u, y_pred_u, zero_division=0)
    r = recall_score(y_true_u, y_pred_u, zero_division=0)
    f1 = f1_score(y_true_u, y_pred_u, zero_division=0)
    f2 = fbeta_score(y_true_u, y_pred_u, beta=2, zero_division=0)
    auc_u = roc_auc_score(y_true_u, y_prob_u) if len(np.unique(y_true_u)) > 1 else 0.0
    
    print(f"  {name:25s}: P={p:.4f}  R={r:.4f}  F1={f1:.4f}  F2={f2:.4f}  ROC-AUC={auc_u:.4f}  thr={best_thr:.2f}")
    return p, r, f1, f2, auc_u

print("Метрики моделей на уровне пользователей:")
u_if_metrics   = evaluate_user_level("Isolation Forest", if_probs, feat, y_true_user, users_df)
u_ae_metrics   = evaluate_user_level("Autoencoder", ae_norm, feat, y_true_user, users_df)
u_xgb_metrics  = evaluate_user_level("XGBoost+SMOTE", xgb_probs, feat, y_true_user, users_df)
u_stk_metrics  = evaluate_user_level("Stacking", stack_probs, feat, y_true_user, users_df)
u_mi_metrics   = evaluate_user_level("Multi-Input Network", mi_probs, feat, y_true_user, users_df)
u_tl_metrics   = evaluate_user_level("Two-Level Pipeline", two_level_probs, feat, y_true_user, users_df)


# %% [markdown]
# ### Feature Importance (XGBoost)

# %%
xgb_model = pipe_xgb.named_steps["clf"]
feat_imp = sorted(
    zip(feature_cols, xgb_model.feature_importances_),
    key=lambda x: x[1], reverse=True,
)

top_n = min(15, len(feat_imp))
names = [f[0] for f in feat_imp[:top_n]][::-1]
values = [f[1] for f in feat_imp[:top_n]][::-1]

fig, ax = plt.subplots(figsize=(10, 7))
colors = sns.color_palette("viridis", top_n)[::-1]
ax.barh(names, values, color=colors)
ax.set_xlabel("Importance")
ax.set_title("Top-15 Feature Importance (XGBoost)")
ax.grid(True, axis="x", alpha=0.3)
fig.savefig(os.path.join(ASSETS_DIR, "feature_importance.png"))
plt.show()

# %% [markdown]
# ## 7. SHAP-объяснения
#
# Используем TreeExplainer для XGBoost (быстрый и точный)
# для интерпретации предсказаний модели.

# %%
try:
    import shap

    print("Вычисление SHAP-значений ...")
    explainer = shap.TreeExplainer(xgb_model)

    # Берём случайную выборку для визуализации
    sample_idx = np.random.choice(len(X), size=min(500, len(X)), replace=False)
    X_sample = X[sample_idx]

    shap_values = explainer.shap_values(X_sample)

    fig, ax = plt.subplots(figsize=(12, 8))
    shap.summary_plot(
        shap_values, X_sample,
        feature_names=feature_cols,
        show=False, max_display=15,
    )
    plt.title("SHAP Summary Plot (XGBoost)")
    plt.tight_layout()
    plt.savefig(os.path.join(ASSETS_DIR, "shap_summary.png"))
    plt.show()

except ImportError:
    print("⚠️ shap не установлен. Пропускаем SHAP-визуализацию.")
    print("  Для установки: pip install shap")

# %% [markdown]
# ### 7.2 Веса внимания Multi-Branch нейросети (Explainable AI)
#
# Механизм Attention Fusion позволяет модели динамически взвешивать важность
# каждой поведенческой ветви. Анализ средних весов внимания показывает вклад
# каждой плоскости в итоговые решения о наличии угроз.

# %%
mi_model.eval()
with torch.no_grad():
    _, attn_all = mi_model.get_attention(all_data)
    attn_weights_mean = attn_all.mean(dim=0).numpy()

branch_names_plot = list(BRANCHES.keys())

fig, ax = plt.subplots(figsize=(8, 4))
colors_attn = sns.color_palette("coolwarm", len(branch_names_plot))
bars = ax.barh(branch_names_plot, attn_weights_mean, color=colors_attn)
ax.set_xlabel("Средний вес внимания (Attention Weight)")
ax.set_title("Важность поведенческих плоскостей (Attention Fusion)")
ax.set_xlim(0, 1.0)
ax.grid(True, axis="x", alpha=0.3)

for bar in bars:
    width = bar.get_width()
    ax.text(width + 0.02, bar.get_y() + bar.get_height()/2, f"{width*100:.1f}%",
            va='center', ha='left', fontsize=10, weight='bold')

plt.tight_layout()
plt.savefig(os.path.join(ASSETS_DIR, "attention_weights.png"))
plt.show()

# %% [markdown]
# ## 8. Экспорт модели
#
# Сохраняем обученные модели и препроцессор для использования
# в production (FastAPI backend).

# %%
print("=" * 60)
print("ЭКСПОРТ МОДЕЛИ")
print("=" * 60)

models_dir = os.path.join(ASSETS_DIR, "models")
os.makedirs(models_dir, exist_ok=True)

# Autoencoder
torch.save(ae_model.state_dict(), os.path.join(models_dir, "autoencoder.pt"))
print("  [OK] autoencoder.pt")

# Isolation Forest
with open(os.path.join(models_dir, "isolation_forest.pkl"), "wb") as f:
    pickle.dump(iso_forest, f)
print("  [OK] isolation_forest.pkl")

# XGBoost (лучшая single model)
pipe_xgb.named_steps["clf"].save_model(os.path.join(models_dir, "xgboost.json"))
print("  [OK] xgboost.json")

# Multi-Input Classifier
torch.save(mi_model.state_dict(), os.path.join(models_dir, "multi_input.pt"))
print("  [OK] multi_input.pt")

# StandardScaler
with open(os.path.join(models_dir, "scaler.pkl"), "wb") as f:
    pickle.dump(scaler, f)
print("  [OK] scaler.pkl")

# Feature columns list
with open(os.path.join(models_dir, "feature_cols.pkl"), "wb") as f:
    pickle.dump(feature_cols, f)
print("  [OK] feature_cols.pkl")

# Метрики
with open(os.path.join(ASSETS_DIR, "training_metrics.txt"), "w", encoding="utf-8") as f:
    f.write(f"CERT r4.2 — Training Results\n{'=' * 50}\n\n")
    f.write(f"Observations: {len(feat):,}\n")
    f.write(f"Positive: {n_pos}\n")
    f.write(f"Negative: {n_neg}\n")
    f.write(f"Features: {len(feature_cols)}\n\n")
    f.write(f"USER-DAY LEVEL METRICS:\n")
    f.write(f"IF:     P={if_metrics[0]:.4f} R={if_metrics[1]:.4f} F1={if_metrics[2]:.4f} F2={if_metrics[3]:.4f}\n")
    f.write(f"AE:     P={ae_metrics[0]:.4f} R={ae_metrics[1]:.4f} F1={ae_metrics[2]:.4f} F2={ae_metrics[3]:.4f}\n")
    f.write(f"RF:     P={rf_metrics[0]:.4f} R={rf_metrics[1]:.4f} F1={rf_metrics[2]:.4f} F2={rf_metrics[3]:.4f} PR-AUC={rf_metrics[4]:.4f}\n")
    f.write(f"XGB:    P={xgb_metrics[0]:.4f} R={xgb_metrics[1]:.4f} F1={xgb_metrics[2]:.4f} F2={xgb_metrics[3]:.4f} PR-AUC={xgb_metrics[4]:.4f}\n")
    f.write(f"LGB:    P={lgb_metrics[0]:.4f} R={lgb_metrics[1]:.4f} F1={lgb_metrics[2]:.4f} F2={lgb_metrics[3]:.4f} PR-AUC={lgb_metrics[4]:.4f}\n")
    f.write(f"Avg:    P={avg_metrics[0]:.4f} R={avg_metrics[1]:.4f} F1={avg_metrics[2]:.4f} F2={avg_metrics[3]:.4f} PR-AUC={avg_metrics[4]:.4f}\n")
    f.write(f"Stack:  P={stk_metrics[0]:.4f} R={stk_metrics[1]:.4f} F1={stk_metrics[2]:.4f} F2={stk_metrics[3]:.4f} PR-AUC={stk_metrics[4]:.4f}\n")
    f.write(f"Multi:  P={mi_metrics[0]:.4f} R={mi_metrics[1]:.4f} F1={mi_metrics[2]:.4f} F2={mi_metrics[3]:.4f} PR-AUC={mi_metrics[4]:.4f}\n")
    f.write(f"TwoLvl: P={tl_metrics[0]:.4f} R={tl_metrics[1]:.4f} F1={tl_metrics[2]:.4f} F2={tl_metrics[3]:.4f} PR-AUC={tl_metrics[4]:.4f}\n\n")
    f.write(f"USER LEVEL METRICS:\n")
    f.write(f"IF:     P={u_if_metrics[0]:.4f} R={u_if_metrics[1]:.4f} F1={u_if_metrics[2]:.4f} F2={u_if_metrics[3]:.4f} ROC-AUC={u_if_metrics[4]:.4f}\n")
    f.write(f"AE:     P={u_ae_metrics[0]:.4f} R={u_ae_metrics[1]:.4f} F1={u_ae_metrics[2]:.4f} F2={u_ae_metrics[3]:.4f} ROC-AUC={u_ae_metrics[4]:.4f}\n")
    f.write(f"XGB:    P={u_xgb_metrics[0]:.4f} R={u_xgb_metrics[1]:.4f} F1={u_xgb_metrics[2]:.4f} F2={u_xgb_metrics[3]:.4f} ROC-AUC={u_xgb_metrics[4]:.4f}\n")
    f.write(f"Stack:  P={u_stk_metrics[0]:.4f} R={u_stk_metrics[1]:.4f} F1={u_stk_metrics[2]:.4f} F2={u_stk_metrics[3]:.4f} ROC-AUC={u_stk_metrics[4]:.4f}\n")
    f.write(f"Multi:  P={u_mi_metrics[0]:.4f} R={u_mi_metrics[1]:.4f} F1={u_mi_metrics[2]:.4f} F2={u_mi_metrics[3]:.4f} ROC-AUC={u_mi_metrics[4]:.4f}\n")
    f.write(f"TwoLvl: P={u_tl_metrics[0]:.4f} R={u_tl_metrics[1]:.4f} F1={u_tl_metrics[2]:.4f} F2={u_tl_metrics[3]:.4f} ROC-AUC={u_tl_metrics[4]:.4f}\n\n")
    f.write(f"Best Model (User-Day F2): {best_name} (F2={best_metrics[3]:.4f})\n")
print("  [OK] training_metrics.txt")

total_time = (time.time() - T0) / 60
print(f"\n{'=' * 60}")
print(f"ГОТОВО за {total_time:.1f} мин")
print(f"{'=' * 60}")
