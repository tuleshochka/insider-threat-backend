"""
Локальный скрипт подготовки NLP-датасета из сырых логов CERT r5.2.
С микро-признаками времени (hour, is_working_hours, time_since_last_event)

Запуск:
    python prepare_nlp_dataset.py
"""

import csv
import os
import json
import numpy as np
from collections import defaultdict
from datetime import datetime

# ============================================================
# НАСТРОЙКИ
# ============================================================
DATASET_DIR = r"D:\Политех\Мага\Дипломы\М\12841247\r5.2"
ANSWERS_DIR = r"D:\Политех\Мага\Дипломы\М\12841247\answers"
OUTPUT_FILE = r"D:\Политех\Мага\Дипломы\М\insider_threat_recongition\insider-threat-backend\notebooks\cert_nlp_sequences.npz"

MAX_SEQ_LEN = 200  # Увеличено до 200, чтобы не терять важные события

print("=" * 60)
print("ЭТАП 1: Чтение ответов (answers)")
print("=" * 60)

malicious_event_ids = set()

answer_scenarios = [d for d in os.listdir(ANSWERS_DIR) if d.startswith("r5.2")]
for scenario in sorted(answer_scenarios):
    scenario_path = os.path.join(ANSWERS_DIR, scenario)
    for fname in os.listdir(scenario_path):
        if not fname.endswith(".csv"): continue
        fpath = os.path.join(scenario_path, fname)
        with open(fpath, "r", encoding="utf-8", errors="replace") as f:
            for row in csv.reader(f):
                if len(row) < 4: continue
                event_id, date_str, user = row[1].strip(), row[2].strip(), row[3].strip()
                malicious_event_ids.add(event_id)

print(f"  Найдено вредоносных event-IDs: {len(malicious_event_ids)}")

TOKEN_MAP = {
    "<PAD>": 0,
    "<UNK>": 1,

    "LOGON": 2,
    "LOGOFF": 3,
    "LOGON_AFTER_HOURS": 4,
    "LOGON_WEEKEND": 5,

    "USB_CONNECT": 6,
    "USB_DISCONNECT": 7,
    "USB_LARGE_TRANSFER": 8,

    "FILE_OPEN": 9,
    "FILE_WRITE": 10,
    "FILE_COPY": 11,
    "FILE_DELETE": 12,

    "FILE_OPEN_USB": 13,
    "FILE_WRITE_USB": 14,
    "FILE_COPY_USB": 15,
    "FILE_DELETE_USB": 16,

    "FILE_WRITE_USB_ARCHIVE": 17,
    "FILE_WRITE_USB_EXE": 18,
    "FILE_COPY_USB_ARCHIVE": 19,
    "FILE_COPY_USB_EXE": 20,
    "FILE_BULK_COPY": 21,

    "EMAIL_SEND_INT": 22,
    "EMAIL_SEND_EXT": 23,
    "EMAIL_SEND_EXT_LARGE": 24,
    "EMAIL_SEND_EXT_BULK": 25,

    "HTTP_BROWSE": 26,
    "HTTP_UPLOAD": 27,
    "HTTP_UPLOAD_LARGE": 28,
    "HTTP_DOWNLOAD": 29
}

VOCAB_SIZE = len(TOKEN_MAP)

print("\n" + "=" * 60)
print("ЭТАП 2: Чтение и токенизация сырых логов (С СОРТИРОВКОЙ И ИЗВЛЕЧЕНИЕМ ВРЕМЕНИ)")
print("=" * 60)

user_day_events = defaultdict(list)

def parse_date(date_str):
    try:
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y %H:%M:%S")
        return dt, dt.strftime("%Y-%m-%d"), dt.timestamp(), dt.hour
    except ValueError:
        return None, None, None, None

def process_logon(row):
    if len(row) < 5: return None, None, None, None, None, None
    dt, day, ts, hour = parse_date(row[1])
    if dt is None: return None, None, None, None, None, None
    activity = row[4].strip().lower()
    is_logon = (activity == "logon")
    token = "LOGON" if is_logon else "LOGOFF"
    
    is_weekend = (dt.weekday() >= 5)
    is_after_hours = (hour < 6 or hour > 20)
    
    if is_logon:
        if is_weekend:
            token = "LOGON_WEEKEND"
        elif is_after_hours:
            token = "LOGON_AFTER_HOURS"
    return row[0], row[2], day, ts, hour, token

def process_device(row):
    if len(row) < 6: return None, None, None, None, None, None
    dt, day, ts, hour = parse_date(row[1])
    token = "USB_CONNECT" if row[5].strip().lower() == "connect" else "USB_DISCONNECT"
    return row[0], row[2], day, ts, hour, token

def process_file(row):
    if len(row) < 8: return None, None, None, None, None, None
    dt, day, ts, hour = parse_date(row[1])
    if dt is None: return None, None, None, None, None, None
    activity, is_usb = row[5].strip().lower(), (row[6].strip().lower() == "true" or row[7].strip().lower() == "true")
    filename = row[4].strip().lower()
    
    is_archive = any(filename.endswith(ext) for ext in [".zip", ".rar", ".7z", ".tar", ".gz"])
    is_exe = any(filename.endswith(ext) for ext in [".exe", ".bat", ".sh", ".dll", ".msi"])
    
    if "open" in activity: token = "FILE_OPEN_USB" if is_usb else "FILE_OPEN"
    elif "write" in activity:
        if is_usb:
            token = "FILE_WRITE_USB_EXE" if is_exe else ("FILE_WRITE_USB_ARCHIVE" if is_archive else "FILE_WRITE_USB")
        else:
            token = "FILE_WRITE"
    elif "copy" in activity:
        if is_usb:
            token = "FILE_COPY_USB_EXE" if is_exe else ("FILE_COPY_USB_ARCHIVE" if is_archive else "FILE_COPY_USB")
        else:
            token = "FILE_COPY"
    elif "delete" in activity: token = "FILE_DELETE_USB" if is_usb else "FILE_DELETE"
    else: token = "FILE_OPEN_USB" if is_usb else "FILE_OPEN"
    return row[0], row[2], day, ts, hour, token

INTERNAL_DOMAINS = set(os.environ.get("INTERNAL_DOMAINS", "@dtaa.com,@dtaa.org").split(","))

def process_email(row):
    if len(row) < 5: return None, None, None, None, None, None
    dt, day, ts, hour = parse_date(row[1])
    if dt is None: return None, None, None, None, None, None
    recipients = (row[4] if len(row)>4 else "").split(";")
    has_external = False
    ext_count = 0
    for r in recipients:
        r_clean = r.strip().lower()
        if r_clean and not any(r_clean.endswith(d) for d in INTERNAL_DOMAINS):
            has_external = True
            ext_count += 1
            
    token = "EMAIL_SEND_INT"
    if has_external:
        try:
            size = int(row[8]) if len(row) > 8 else 0
        except Exception:
            size = 0
        if size > 10_000_000:
            token = "EMAIL_SEND_EXT_LARGE"
        elif ext_count > 5:
            token = "EMAIL_SEND_EXT_BULK"
        else:
            token = "EMAIL_SEND" if has_external else "EMAIL_SEND_INT"
            # Wait, token is "EMAIL_SEND_EXT"
            token = "EMAIL_SEND_EXT"
    return row[0], row[2], day, ts, hour, token

def process_http(row):
    if len(row) < 5: return None, None, None, None, None, None
    dt, day, ts, hour = parse_date(row[1])
    url = row[4].strip().lower() if len(row) > 4 else ""
    if any(kw in url for kw in ["upload", "dropbox", "drive.google", "wetransfer", "github", "mega.nz", "sendspace"]):
        token = "HTTP_UPLOAD"
    elif any(kw in url for kw in ["download", ".exe", ".zip", ".rar", ".msi", ".tar", ".gz"]):
        token = "HTTP_DOWNLOAD"
    else:
        token = "HTTP_BROWSE"
    return row[0], row[2], day, ts, hour, token

log_files = [
    ("logon.csv", process_logon), ("device.csv", process_device), 
    ("file.csv", process_file), ("email.csv", process_email), ("http.csv", process_http)
]

total_events, total_malicious = 0, 0

for filename, processor in log_files:
    filepath = os.path.join(DATASET_DIR, filename)
    if not os.path.exists(filepath): continue
    print(f"\n  Обработка {filename} ({(os.path.getsize(filepath)/1024**3):.2f} ГБ)...")
    
    with open(filepath, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.reader(f)
        next(reader, None)
        for i, row in enumerate(reader):
            event_id, user, day, ts, hour, token = processor(row)
            if user is None or day is None or ts is None or hour is None: continue
            
            is_mal = (event_id in malicious_event_ids)
            user_day_events[(user, day)].append((ts, hour, TOKEN_MAP.get(token, 1), is_mal))
            
            total_events += 1
            if is_mal: total_malicious += 1
            if total_events % 5_000_000 == 0: print(f"    ... обработано {total_events:,} строк")

print(f"\n  ВСЕГО событий: {total_events:,} | Вредоносных: {total_malicious:,}")

print("\n" + "=" * 60)
print("ЭТАП 3: Формирование последовательностей и User Behavioral Baselines (v9)")
print("=" * 60)

# 1. Сначала сортируем все события внутри дней
for (user, day), events in sorted(user_day_events.items(), key=lambda x: (x[0][0], x[0][1])):
    events.sort(key=lambda x: x[0])

# 2. Вычисляем сырые суточные метрики для каждого дня пользователя (log1p от счетчиков)
# Метрики: [общий счет событий, число USB-событий, число внешних email-событий, число HTTP-загрузок]
raw_features = {}
for (user, day), events in user_day_events.items():
    event_count = len(events)
    # USB токены: USB_CONNECT(4), USB_DISCONNECT(5), FILE_OPEN_USB(10), FILE_WRITE_USB(11), FILE_COPY_USB(12), FILE_DELETE_USB(13)
    usb_count = sum(1 for e in events if e[2] in [4, 5, 10, 11, 12, 13])
    # EMAIL_SEND_EXT(15)
    email_count = sum(1 for e in events if e[2] == 15)
    # HTTP_UPLOAD(17)
    http_count = sum(1 for e in events if e[2] == 17)
    
    raw_features[(user, day)] = np.array([
        np.log1p(event_count),
        np.log1p(usb_count),
        np.log1p(email_count),
        np.log1p(http_count)
    ], dtype=np.float32)


# Behavioral drift features
behavioral_drift = {}
ROLLING_WINDOW_DAYS = 14

user_days_map = defaultdict(list)

for (user, day) in raw_features.keys():
    user_days_map[user].append(day)

for user in user_days_map:
    user_days_map[user] = sorted(user_days_map[user])

for user, days in user_days_map.items():
    for idx, day in enumerate(days):
        current = raw_features[(user, day)]
        prev_days = days[max(0, idx - ROLLING_WINDOW_DAYS):idx]

        if len(prev_days) == 0:
            drift = np.zeros_like(current)
        else:
            baseline = np.mean([
                raw_features[(user, d)]
                for d in prev_days
            ], axis=0)

            drift = current - baseline

        behavioral_drift[(user, day)] = np.clip(
            drift,
            -5.0,
            5.0
        ).astype(np.float32)

# 3. Делаем разбиение на пользователей для вычисления профилей (Train-only)
unique_users = sorted(list(set(u for u, d in user_day_events.keys())))
np.random.seed(42)
shuffled_users = unique_users.copy()
np.random.shuffle(shuffled_users)
n_users = len(shuffled_users)
train_split = int(0.7 * n_users)
val_split = int(0.85 * n_users)
train_users_list = shuffled_users[:train_split]
val_users_list = shuffled_users[train_split:val_split]
test_users_list = shuffled_users[val_split:]

train_users = set(train_users_list)

print(f"  Выделено пользователей для профилирования (Train): {len(train_users)}")

# Собираем сырые признаки для всех Train дней по пользователям
train_user_features = defaultdict(list)
for (user, day), feats in raw_features.items():
    if user in train_users:
        train_user_features[user].append(feats)

# Считаем среднее и std для каждого Train пользователя
user_stats = {}
for user, feats_list in train_user_features.items():
    feats_arr = np.array(feats_list, dtype=np.float32)
    user_stats[user] = {
        'mean': feats_arr.mean(axis=0),
        'std': feats_arr.std(axis=0)
    }

# Вычисляем глобальные средние по Train выборке с учетом веса ( pooled mean и std )
weighted_means = []
weighted_vars = []
total_days = 0

for user, feats_list in train_user_features.items():
    n = len(feats_list)
    feats_arr = np.array(feats_list, dtype=np.float32)
    user_mean = feats_arr.mean(axis=0)
    user_var = feats_arr.var(axis=0)
    weighted_means.append(n * user_mean)
    weighted_vars.append(n * user_var)
    total_days += n

global_mean = np.sum(weighted_means, axis=0) / total_days if total_days > 0 else np.zeros(4, dtype=np.float32)
global_var = np.sum(weighted_vars, axis=0) / total_days if total_days > 0 else np.ones(4, dtype=np.float32)
global_std = np.sqrt(global_var) + 1e-8

print(f"  Глобальный профиль - Mean: {global_mean}, Std: {global_std}")

# 4. Формируем последовательности с эвристической обрезкой (Suspicious Trimming) и Z-score отклонениями
sequences, hours_seq, working_hours_seq, tsle_seq = [], [], [], []
labels, users_list, dates_list, dev_z_list = [], [], [], []

CONTEXT_RATIO = 0.1

def compute_trim_score(token_id, hour):
    score = 0
    if token_id in [4, 5, 10, 11, 12, 13, 15, 17, 21, 22, 23, 24, 25, 26]:
        score += 3
    if hour < 8 or hour > 18:
        score += 1
    return score

for (user, day), events in user_day_events.items():
    # Извлекаем baseline параметры
    if user in train_users and user in user_stats:
        mean = user_stats[user]['mean']
        std = user_stats[user]['std']
    else:
        mean = global_mean
        std = global_std
    
    # Считаем Z-score отклонение с ограничением std снизу и клиппированием
    daily_raw = raw_features[(user, day)]
    safe_std = np.maximum(std, global_std * 0.25)
    dev_z = (daily_raw - mean) / safe_std
    dev_z = np.clip(dev_z, -10.0, 10.0)
    
    seq, h_seq, wh_seq, tsle_s = [], [], [], []
    prev_ts = None
    for ts, hour, token_id, is_mal in events:
        seq.append(token_id)
        h_seq.append(hour)
        wh_seq.append(1 if 8 <= hour <= 18 else 0)
        tsle_s.append(0.0 if prev_ts is None else (ts - prev_ts))
        prev_ts = ts
        
    has_malicious = any(e[3] for e in events)
    
    # Обрезаем события без утечки разметки (Suspicious Trimming)
    if len(seq) > MAX_SEQ_LEN:
        scores = [compute_trim_score(e[2], e[1]) for e in events]
        
        if max(scores) > 0:
            # Находим индекс максимального скора (suspicious anchor)
            anchor = int(np.argmax(scores))
            context_len = int(MAX_SEQ_LEN * CONTEXT_RATIO)
            start = anchor - context_len
            start = max(0, min(len(events) - MAX_SEQ_LEN, start))
        else:
            # Если подозрительных событий нет, берем с конца
            start = len(events) - MAX_SEQ_LEN
            
        seq = seq[start : start + MAX_SEQ_LEN]
        h_seq = h_seq[start : start + MAX_SEQ_LEN]
        wh_seq = wh_seq[start : start + MAX_SEQ_LEN]
        tsle_s = tsle_s[start : start + MAX_SEQ_LEN]
        tsle_s[0] = 0.0  # Сбрасываем интервал первого события обрезанного дня
    else:
        pad_len = MAX_SEQ_LEN - len(seq)
        seq = seq + [TOKEN_MAP["<PAD>"]] * pad_len
        h_seq = h_seq + [0] * pad_len
        wh_seq = wh_seq + [0] * pad_len
        tsle_s = tsle_s + [0.0] * pad_len
        
    sequences.append(seq)
    hours_seq.append(h_seq)
    working_hours_seq.append(wh_seq)
    tsle_seq.append(tsle_s)
    
    labels.append(1 if has_malicious else 0)
    users_list.append(user)
    dates_list.append(day)
    combined_features = np.concatenate([
        dev_z,
        behavioral_drift[(user, day)]
    ]).astype(np.float32)

    dev_z_list.append(combined_features)

X = np.array(sequences, dtype=np.int32)
X_hours = np.array(hours_seq, dtype=np.int32)
X_working_hours = np.array(working_hours_seq, dtype=np.int32)
X_tsle = np.array(tsle_seq, dtype=np.float32)
X_dev = np.array(dev_z_list, dtype=np.float32)
y = np.array(labels, dtype=np.int32)

print(f"  Размер X: {X.shape} | Размер X_dev: {X_dev.shape} | Размер y: {y.shape}")

metadata = {
    "vocab_size": VOCAB_SIZE, "max_seq_len": MAX_SEQ_LEN, "token_map": TOKEN_MAP,
    "num_sequences": len(sequences), "num_positive": int(y.sum()),
    "global_mean": global_mean.tolist(), "global_std": global_std.tolist()
}

np.savez_compressed(
    OUTPUT_FILE, 
    X=X, 
    X_hours=X_hours,
    X_working_hours=X_working_hours,
    X_tsle=X_tsle,
    X_dev=X_dev,
    y=y, 
    users=np.array(users_list), 
    dates=np.array(dates_list),
    users_dev=np.array(users_list),
    dates_dev=np.array(dates_list),
    train_users=np.array(train_users_list),
    val_users=np.array(val_users_list),
    test_users=np.array(test_users_list),
    metadata=json.dumps(metadata)
)

split_info = {
    'train_users': train_users_list,
    'val_users': val_users_list,
    'test_users': test_users_list,
    'seed': 42
}

split_json_path = OUTPUT_FILE.replace('.npz', '_split.json')
with open(split_json_path, 'w', encoding='utf-8') as f:
    json.dump(split_info, f, ensure_ascii=False, indent=2)

print(f"  Файл разбиения сохранен: {split_json_path}")

print(f"\n  Файл сохранен: {OUTPUT_FILE} ({(os.path.getsize(OUTPUT_FILE)/1024**2):.1f} МБ)")
print("ГОТОВО! Загрузите в Colab.")
