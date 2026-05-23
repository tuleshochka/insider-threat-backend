"""Evaluate anomaly detection against CERT r4.2 ground truth.
#
# Почему этот скрипт существует:
#   Основной пайплайн обучения (Trainer) — безнадзорный, он не использует
#   разметку. Этот скрипт — отдельный оценочный модуль, который берёт
#   результаты работы детектора (таблица Anomaly) и сравнивает их с
#   эталонным списком инсайдеров из CERT r4.2 (ground truth). Так мы
#   отделяем логику обучения от логики валидации.
#
# Почему метрики на уровне пользователей (user-level), а не по дням:
#   Внутренняя безопасность интересуется не «в какой день пользователь
#   был подозрителен», а «является ли пользователь злоумышленником в
#   принципе». User-level метрики (Precision, Recall, F1 на множестве
#   пользователей) соответствуют реальному решению — кого блокировать,
#   за кем усилить наблюдение. Per-day метрики были бы чувствительны
#   к шуму и не отражали бы практическую ценность системы.
#
# Почему сравнение с ground truth:
#   Без эталонной разметки любая оценка качества детекции аномалий
#   сводится к субъективным суждениям. CERT r4.2 предоставляет
#   проверенный список инсайдеров и временные окна их активности.
#   Сравнение с этим эталоном даёт объективную, воспроизводимую метрику.
#
# Почему перебор порогов (threshold sweep):
#   Безнадзорные детекторы выдают непрерывный score аномальности.
#   Выбор порога отсечки — гиперпараметр, который кардинально меняет
#   баланс Precision/Recall. Перебор порогов (90, 95, 97, 99 процентили)
#   показывает, как меняется качество в зависимости от жёсткости
#   детекции, и помогает выбрать порог под конкретные требования ИБ.
"""
import os
os.environ["DB_TYPE"] = "sqlite"
os.chdir(os.path.dirname(os.path.abspath(__file__)))

from datetime import datetime
from collections import Counter
import pandas as pd
from app.models.db import SessionLocal
from app.models.orm import Anomaly, User
from app.config import settings

# ── 1. Ground truth ──
gt = pd.read_csv(settings.ground_truth_path)
r42 = gt[gt["dataset"] == 4.2].copy()
r42["start"] = pd.to_datetime(r42["start"])
r42["end"] = pd.to_datetime(r42["end"])
insider_info = {row["user"]: (row["start"], row["end"]) for _, row in r42.iterrows()}
insider_ids = set(insider_info.keys())
print(f"Ground truth: {len(insider_ids)} insider users")

# ── 2. Load DB data ──
db = SessionLocal()
users = {u.id: u.anon_id for u in db.query(User).all()}
anomalies = db.query(Anomaly).all()
db.close()
print(f"DB: {len(users)} users, {len(anomalies)} anomaly records")

# ── 3. Per-user anomaly aggregation ──
flagged_users = set()  # users with at least one anomaly
user_anomaly_dates: dict[str, set] = {}
for a in anomalies:
    anon = users.get(a.user_id)
    if not anon:
        continue
    flagged_users.add(anon)
    if anon not in user_anomaly_dates:
        user_anomaly_dates[anon] = set()
    user_anomaly_dates[anon].add(a.detected_at.date())

# ── 4. Per-user metrics ──
tp_users = flagged_users & insider_ids
fp_users = flagged_users - insider_ids
fn_users = insider_ids - flagged_users
tn_users = set(users.values()) - flagged_users - insider_ids

prec = len(tp_users) / len(flagged_users) if flagged_users else 0
rec = len(tp_users) / len(insider_ids) if insider_ids else 0
f1 = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0
acc = (len(tp_users) + len(tn_users)) / len(users) if users else 0

print(f"\n{'='*60}")
print(f"ПОЛЬЗОВАТЕЛЬСКИЙ УРОВЕНЬ (user-level)")
print(f"{'='*60}")
print(f"  True Positives:  {len(tp_users):3d}  ({len(tp_users)/len(insider_ids)*100:.1f}% инсайдеров)")
print(f"  False Positives: {len(fp_users):3d}  ({len(fp_users)/len(flagged_users)*100:.1f}% флагов)")
print(f"  False Negatives: {len(fn_users):3d}  ({len(fn_users)/len(insider_ids)*100:.1f}% пропущено)")
print(f"  True Negatives:  {len(tn_users):3d}")
print(f"  Precision: {prec:.4f} ({prec*100:.1f}%)")
print(f"  Recall:    {rec:.4f} ({rec*100:.1f}%)")
print(f"  F1-Score:  {f1:.4f} ({f1*100:.1f}%)")
print(f"  Accuracy:  {acc:.4f} ({acc*100:.1f}%)")

# ── 5. Per-user-date metrics (matching ground truth windows) ──
print(f"\n{'='*60}")
print(f"ДЕТЕКЦИЯ ВО ВРЕМЕННЫХ ОКНАХ ИНСАЙДЕРОВ")
print(f"{'='*60}")
detected_in_window = 0
for anon in insider_ids:
    if anon not in user_anomaly_dates:
        print(f"  {anon}: не обнаружен")
        continue
    s, e = insider_info[anon]
    window_dates = set()
    d = s.date()
    while d <= e.date():
        window_dates.add(d)
        d += pd.Timedelta(days=1)
    hits = user_anomaly_dates[anon] & window_dates
    if hits:
        detected_in_window += 1
        print(f"  {anon}: обнаружен ({len(hits)}/{len(window_dates)} дней в окне)")
    else:
        print(f"  {anon}: аномалии есть, но вне временного окна")

print(f"  Инсайдеров с детекцией во временном окне: {detected_in_window}/{len(insider_ids)} ({detected_in_window/len(insider_ids)*100:.1f}%)")

# ── 6. Top anomalies ──
print(f"\n{'='*60}")
print(f"ТОП-20 ПОЛЬЗОВАТЕЛЕЙ ПО ЧИСЛУ АНОМАЛИЙ")
print(f"{'='*60}")
cnt = Counter()
for a in anomalies:
    anon = users.get(a.user_id, f"id:{a.user_id}")
    cnt[anon] += 1
for anon, c in cnt.most_common(20):
    label = "ЗЛОУМЫШЛЕННИК" if anon in insider_ids else "обычный"
    print(f"  {anon:>10}: {c:3d} аномалий ({label})")

# ── 7. Threshold sweep ──
print(f"\n{'='*60}")
print(f"АНАЛИЗ ПОРОГОВ (ANOMALY SCORE PERCENTILE)")
print(f"{'='*60}")
scores = sorted([a.score for a in anomalies], reverse=True)
for pct in [90, 95, 97, 99]:
    threshold = sorted(scores)[int(len(scores) * (100 - pct) / 100)] if scores else 0
    flagged_above = {users.get(a.user_id) for a in anomalies if a.score >= threshold}
    flagged_above.discard(None)
    tp_above = flagged_above & insider_ids
    p_above = len(tp_above) / len(flagged_above) if flagged_above else 0
    r_above = len(tp_above) / len(insider_ids) if insider_ids else 0
    print(f"  Порог {pct}%: threshold={threshold:.2f}, флагов={len(flagged_above)}, "
          f"TP={len(tp_above)}, Precision={p_above:.3f}, Recall={r_above:.3f}")
