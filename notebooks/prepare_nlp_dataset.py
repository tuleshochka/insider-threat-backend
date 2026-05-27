"""
Локальный скрипт подготовки NLP-датасета из сырых логов CERT r5.2.
ИСПРАВЛЕННАЯ ВЕРСИЯ (Сортировка по времени)

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
malicious_user_dates = set()

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
                try:
                    day = datetime.strptime(date_str, "%m/%d/%Y %H:%M:%S").strftime("%Y-%m-%d")
                    malicious_user_dates.add((user, day))
                except ValueError: pass

print(f"  Найдено вредоносных event-IDs: {len(malicious_event_ids)}")

TOKEN_MAP = {
    "<PAD>": 0, "<UNK>": 1, "LOGON": 2, "LOGOFF": 3, "USB_CONNECT": 4, 
    "USB_DISCONNECT": 5, "FILE_OPEN": 6, "FILE_WRITE": 7, "FILE_COPY": 8, 
    "FILE_DELETE": 9, "FILE_OPEN_USB": 10, "FILE_WRITE_USB": 11, 
    "FILE_COPY_USB": 12, "FILE_DELETE_USB": 13, "EMAIL_SEND_INT": 14, 
    "EMAIL_SEND_EXT": 15, "HTTP_BROWSE": 16, "HTTP_UPLOAD": 17, "HTTP_DOWNLOAD": 18
}
VOCAB_SIZE = len(TOKEN_MAP)

print("\n" + "=" * 60)
print("ЭТАП 2: Чтение и токенизация сырых логов (С СОРТИРОВКОЙ)")
print("=" * 60)

# Теперь храним список кортежей: user_day_events[(user, day)] = [(timestamp, token_id, is_malicious), ...]
user_day_events = defaultdict(list)

def parse_date(date_str):
    try:
        dt = datetime.strptime(date_str.strip(), "%m/%d/%Y %H:%M:%S")
        return dt, dt.strftime("%Y-%m-%d"), dt.timestamp()
    except ValueError:
        return None, None, None

def process_logon(row):
    if len(row) < 5: return None, None, None, None, None
    dt, day, ts = parse_date(row[1])
    token = "LOGON" if row[4].strip().lower() == "logon" else "LOGOFF"
    return row[0], row[2], day, ts, token

def process_device(row):
    if len(row) < 6: return None, None, None, None, None
    dt, day, ts = parse_date(row[1])
    token = "USB_CONNECT" if row[5].strip().lower() == "connect" else "USB_DISCONNECT"
    return row[0], row[2], day, ts, token

def process_file(row):
    if len(row) < 8: return None, None, None, None, None
    dt, day, ts = parse_date(row[1])
    activity, is_usb = row[5].strip().lower(), (row[6].strip().lower() == "true" or row[7].strip().lower() == "true")
    if "open" in activity: token = "FILE_OPEN_USB" if is_usb else "FILE_OPEN"
    elif "write" in activity: token = "FILE_WRITE_USB" if is_usb else "FILE_WRITE"
    elif "copy" in activity: token = "FILE_COPY_USB" if is_usb else "FILE_COPY"
    elif "delete" in activity: token = "FILE_DELETE_USB" if is_usb else "FILE_DELETE"
    else: token = "FILE_OPEN_USB" if is_usb else "FILE_OPEN"
    return row[0], row[2], day, ts, token

def process_email(row):
    if len(row) < 5: return None, None, None, None, None
    dt, day, ts = parse_date(row[1])
    has_external = any(r.strip() and "@dtaa.com" not in r for r in (row[4] if len(row)>4 else "").split(";"))
    return row[0], row[2], day, ts, "EMAIL_SEND_EXT" if has_external else "EMAIL_SEND_INT"

def process_http(row):
    if len(row) < 5: return None, None, None, None, None
    dt, day, ts = parse_date(row[1])
    return row[0], row[2], day, ts, "HTTP_BROWSE"

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
            event_id, user, day, ts, token = processor(row)
            if user is None: continue
            
            is_mal = (event_id in malicious_event_ids)
            user_day_events[(user, day)].append((ts, TOKEN_MAP.get(token, 1), is_mal))
            
            total_events += 1
            if is_mal: total_malicious += 1
            if total_events % 5_000_000 == 0: print(f"    ... обработано {total_events:,} строк")

print(f"\n  ВСЕГО событий: {total_events:,} | Вредоносных: {total_malicious:,}")

print("\n" + "=" * 60)
print("ЭТАП 3: Сортировка по времени и формирование последовательностей")
print("=" * 60)

sequences, labels, users_list, dates_list = [], [], [], []

for (user, day), events in user_day_events.items():
    # 1. КРИТИЧЕСКИ ВАЖНО: Сортируем события внутри дня по timestamp
    events.sort(key=lambda x: x[0])
    
    # Извлекаем токены и проверяем, есть ли вредоносные в этом окне
    seq = [e[1] for e in events]
    has_malicious = any(e[2] for e in events)
    
    # 2. Обрезаем. Если событий больше MAX_SEQ_LEN, берем ПОСЛЕДНИЕ события, 
    # так как аномалии чаще в конце сессии, либо можно брать первые. Берем последние.
    if len(seq) > MAX_SEQ_LEN:
        seq = seq[-MAX_SEQ_LEN:]
    else:
        seq = seq + [TOKEN_MAP["<PAD>"]] * (MAX_SEQ_LEN - len(seq))
        
    sequences.append(seq)
    labels.append(1 if has_malicious else 0)
    users_list.append(user)
    dates_list.append(day)

X, y = np.array(sequences, dtype=np.int32), np.array(labels, dtype=np.int32)

print(f"  Размер X: {X.shape} | Размер y: {y.shape}")
print(f"  Атак: {y.sum()} | Нормы: {(y==0).sum()}")

metadata = {
    "vocab_size": VOCAB_SIZE, "max_seq_len": MAX_SEQ_LEN, "token_map": TOKEN_MAP,
    "num_sequences": len(sequences), "num_positive": int(y.sum())
}

np.savez_compressed(OUTPUT_FILE, X=X, y=y, users=np.array(users_list), dates=np.array(dates_list), metadata=json.dumps(metadata))
print(f"\n  Файл сохранен: {OUTPUT_FILE} ({(os.path.getsize(OUTPUT_FILE)/1024**2):.1f} МБ)")
print("ГОТОВО! Загрузите в Colab.")
