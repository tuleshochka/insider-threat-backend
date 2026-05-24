import os
import sys
import time
import gc
import pandas as pd
import numpy as np

# === CONFIGURATION ===
DATA_DIR = r"D:\Политех\Мага\Дипломы\М\12841247\r4.2"
GROUND_TRUTH_PATH = r"D:\Политех\Мага\Дипломы\М\12841247\answers\insiders.csv"
OUTPUT_PATH = r"D:\Политех\Мага\Дипломы\М\12841247\processed_features_r4.2.csv"

START_DATE = pd.to_datetime("2010-01-01")
END_DATE = pd.to_datetime("2011-05-01")

print(f"Starting local preprocessing for CERT r4.2")
print(f"Data source: {DATA_DIR}")
print(f"Ground truth path: {GROUND_TRUTH_PATH}")
print(f"Output destination: {OUTPUT_PATH}")
print(f"Date range: {START_DATE.date()} to {END_DATE.date()}")
T_START = time.time()

# === HELPER FOR CHUNKED LOADING ===
def load_table(name: str, cols: list, dtypes: dict, start_dt, end_dt, chunksize=500_000):
    path = os.path.join(DATA_DIR, f"{name}.csv")
    if not os.path.exists(path):
        print(f"Error: {path} not found!")
        sys.exit(1)
        
    chunks = []
    total_read = 0
    t = time.time()
    print(f"  Reading {name}.csv...", end=" ", flush=True)

    for chunk in pd.read_csv(path, usecols=cols, dtype=dtypes, chunksize=chunksize, low_memory=False):
        total_read += len(chunk)
        chunk["date"] = pd.to_datetime(chunk["date"], format="%m/%d/%Y %H:%M:%S", errors="coerce")
        chunk = chunk[(chunk["date"] >= start_dt) & (chunk["date"] < end_dt)]
        if not chunk.empty:
            chunks.append(chunk)

    result = pd.concat(chunks, ignore_index=True) if chunks else pd.DataFrame()
    print(f"{len(result):,} rows loaded (out of {total_read:,}) in {time.time()-t:.1f}s")
    return result

# === AGGREGATION FUNCTIONS ===
def agg_logon(df):
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
    """Email → 6 признаков (все события в email.csv — это отправка с ПК пользователя)."""
    df["date_only"] = df["date"].dt.date
    df["hour"] = df["date"].dt.hour
    
    result = df.groupby(["date_only", "user"], as_index=False).agg(
        email_sent=("id", "count"),
        email_size_total=("size", "sum"),
        email_attachments=("attachments", "sum"),
        email_unique_recipients=("to", "nunique"),
        after_hours_email=("hour", lambda x: ((x < 8) | (x >= 18)).sum()),
    )
    result["email_received"] = 0.0  # Заглушка для обратной совместимости
    result.rename(columns={"user": "anon_id", "date_only": "date"}, inplace=True)
    return result

def agg_device(df):
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

# === LOADING & PROCESSING ===
print("\n--- STEP 1: LOADING RAW DATA ---")
logon = load_table("logon", ["date", "user", "pc", "activity", "id"], {"user": str, "pc": str, "activity": str, "id": str}, START_DATE, END_DATE)
files = load_table("file", ["date", "user", "pc", "filename", "id"], {"user": str, "pc": str, "filename": str, "id": str}, START_DATE, END_DATE)
email = load_table("email", ["date", "user", "to", "from", "size", "attachments", "id"], {"user": str, "to": str, "from": str, "size": "int64", "attachments": "int64", "id": str}, START_DATE, END_DATE)
device = load_table("device", ["date", "user", "pc", "activity", "id"], {"user": str, "pc": str, "activity": str, "id": str}, START_DATE, END_DATE)

# HTTP table (large, processed via chunked aggregation)
http_path = os.path.join(DATA_DIR, "http.csv")
print("  Processing http.csv in chunks...", end=" ", flush=True)
t_http = time.time()
http_aggs = []
http_total = 0
for chunk in pd.read_csv(http_path, usecols=["date", "user", "url", "id"], dtype={"user": str, "url": str, "id": str}, chunksize=1_000_000, low_memory=False):
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
print(f"done ({http_total:,} rows processed) in {time.time()-t_http:.1f}s")
gc.collect()

# === STEP 2: AGGREGATE TO USER-DAY MATRICES ===
print("\n--- STEP 2: AGGREGATING FEATURES ---")
print("  Aggregating logon..."); a1 = agg_logon(logon); del logon
print("  Aggregating file...");  a2 = agg_file(files); del files
print("  Aggregating email..."); a3 = agg_email(email); del email
print("  Aggregating device...");a4 = agg_device(device); del device
gc.collect()

print("  Merging aggregated features...")
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

# Load and merge psychometrics (Big Five traits)
psych_path = os.path.join(DATA_DIR, "psychometric.csv")
if os.path.exists(psych_path):
    print("  Merging psychometric traits...")
    psych = pd.read_csv(psych_path, usecols=["user_id", "O", "C", "E", "A", "N"])
    psych.rename(columns={"user_id": "anon_id"}, inplace=True)
    feat = feat.merge(psych, on="anon_id", how="left")
    feat[["O", "C", "E", "A", "N"]] = feat[["O", "C", "E", "A", "N"]].fillna(0)
    del psych

print(f"  Base matrix shape: {feat.shape}")

# === STEP 3: FEATURE ENGINEERING (RATIOS & DERIVED FEATURES) ===
print("\n--- STEP 3: CALCULATING DERIVED RATIOS ---")
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

# === STEP 4: HISTORICAL USER-RELATIVE Z-SCORES ===
print("\n--- STEP 4: CALCULATING USER-RELATIVE Z-SCORES ---")
activity_cols = [
    "logon_count", "file_operations", "email_sent", "device_operations",
    "after_hours_logons", "after_hours_files", "after_hours_device", "weekend_logons",
]
if "http_requests" in feat.columns:
    activity_cols.append("http_requests")

# Ensure proper sorting by user and chronological date
feat["date"] = pd.to_datetime(feat["date"])
feat.sort_values(["anon_id", "date"], inplace=True)

# Vectorized grouping for user-level mean and standard deviations
for col in activity_cols:
    u_mean = feat.groupby("anon_id")[col].transform("mean")
    u_std = feat.groupby("anon_id")[col].transform("std").replace(0, 1)
    feat[f"{col}_zscore"] = (feat[col] - u_mean) / u_std

# === STEP 5: ROLLING LAG FEATURES ===
print("\n--- STEP 5: CALCULATING ROLLING LAG FEATURES ---")
key_cols = ["logon_count", "file_operations", "device_operations", "email_sent"]
if "http_requests" in feat.columns:
    key_cols.append("http_requests")

for col in key_cols:
    g = feat.groupby("anon_id")[col]
    feat[f"{col}_lag1"] = g.shift(1).fillna(0)
    feat[f"{col}_lag3_avg"] = (g.shift(1).fillna(0) + g.shift(2).fillna(0) + g.shift(3).fillna(0)) / 3
    feat[f"{col}_diff"] = g.diff().fillna(0)

# === STEP 6: LABELING GROUND TRUTH ===
print("\n--- STEP 6: LABELING GROUND TRUTH ---")
if not os.path.exists(GROUND_TRUTH_PATH):
    print(f"Error: insiders.csv ground truth not found at {GROUND_TRUTH_PATH}!")
    sys.exit(1)

gt = pd.read_csv(GROUND_TRUTH_PATH)
r42 = gt[gt["dataset"] == 4.2].copy()
r42["start"] = pd.to_datetime(r42["start"])
r42["end"] = pd.to_datetime(r42["end"])

# Build set of malicious (user, date) tuples
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

feat["target"] = y
n_pos = int(y.sum())
n_neg = len(y) - n_pos
print(f"  Labeled anomalies: {n_pos} positive rows, {n_neg} normal rows ({n_pos/len(y)*100:.3f}% anomaly rate)")

# === STEP 7: SAVE PREPROCESSED FEATURES ===
print("\n--- STEP 7: EXPORTING COMPRESSED DATASET ---")
feat.to_csv(OUTPUT_PATH, index=False)
print(f"Success! Final preprocessed dataset saved to: {OUTPUT_PATH}")
print(f"File size: {os.path.getsize(OUTPUT_PATH) / 1024 / 1024:.2f} MB")
print(f"Total time elapsed: {(time.time() - T_START)/60:.2f} minutes")
