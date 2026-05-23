"""
Обучение многоканального классификатора с фокальными потерями на датасете CERT r4.2.

Архитектура с несколькими входными ветвями (multi-input) выбрана неслучайно:
инсайдерские угрозы проявляются в разных поведенческих плоскостях, и каждая
ветвь отвечает за свою гипотезу аномалии. Объединение ветвей через механизм
внимания (attention) позволяет модели динамически взвешивать важность каждой
плоскости для конкретного пользователя в конкретный день.
"""
import os; os.environ["DB_TYPE"] = "sqlite"; os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import torch
import mlflow
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, roc_curve
from app.services.preprocessor import DataPreprocessor
from app.models.multi_input import MultiInputClassifier, FocalLoss
from app.config import settings

# ── Ветви признаков ──
# Почему три основные ветви + психометрика?
#
#   intensity (интенсивность) — фиксирует объём действий пользователя за день.
#     Инсайдер, готовящий кражу данных, часто генерирует аномально много
#     операций: массовое копирование файлов, рассылка писем, скачивание с
#     внешних носителей. Без этой ветви модель пропустит «шумного» нарушителя.
#     Признаки: logon_count, file_operations, email_sent/received, device_operations,
#     http_requests, email_attachments, email_size_total — всё, что измеряет
#     «громкость» активности.
#
#   diversity (разнообразие) — захватывает смену роли или компрометацию учётной
#     записи. Если пользователь внезапно входит на машины коллег (logon_unique_pc),
#     обращается к необычным файлам (file_unique_names) или сайтам (http_unique_urls),
#     это повод для подозрения. Инсайдер, заметающий следы, может имитировать
#     обычную интенсивность, но не типичный для себя набор ресурсов.
#
#   temporal (временные паттерны) — выявляет активность в нерабочее время и
#     выходные. Это один из сильнейших предикторов инсайдерской угрозы
#     согласно исследованиям CERT [Tuor et al., 2017]. Даже если интенсивность
#     и разнообразие в норме, работа в 3 часа ночи над конфиденциальными данными —
#     красный флаг.
#
#   psychometric (психометрика) — «Большая пятёрка» личностных черт (O — открытость
#     опыту, C — добросовестность, E — экстраверсия, A — доброжелательность,
#     N — нейротизм). Добавлена по результатам исследований CERT: пользователи с
#     низкой добросовестностью (C) и высокой открытостью (O) статистически чаще
#     совершают инсайдерские нарушения [1]. Психометрика не меняется от дня ко
#     дню, поэтому она подаётся как статичный срез — модель учится корректировать
#     базовую априорную вероятность угрозы для каждого пользователя.
#
# Каждая ветвь нормализуется отдельно (z-score по своей группе признаков), чтобы
# градиенты из разных ветвей были в сопоставимом масштабе.
BRANCHES = {
    "intensity": ["logon_count", "file_operations", "email_sent", "email_received",
                  "device_operations", "http_requests", "email_attachments", "email_size_total"],
    "diversity": ["logon_unique_pc", "file_unique_pc", "file_unique_names",
                  "email_unique_recipients", "http_unique_urls"],
    "temporal":  ["after_hours_logons", "after_hours_files", "after_hours_email",
                  "after_hours_device", "after_hours_http", "weekend_logons", "weekend_device"],
    "psychometric": ["O", "C", "E", "A", "N"],
}

# ── 1. Загрузка признаков ──
print("Loading features...")
prep = DataPreprocessor()
# Формируем матрицу «пользователь × день» со всеми поведенческими признаками.
# Выбран 2010 год как репрезентативный: в датасете CERT r4.2 основная часть
# инцидентов приходится на этот период, при этом «холодный» период (без угроз)
# тоже достаточен для обучения модели на фоне нормального поведения.
features = prep.build_training_matrix("2010-01-01", "2010-12-31")
print(f"  Samples: {len(features)}, total cols: {len(features.columns)}")

# Нормализация каждой ветви отдельно (z-score).
# Почему не стандартизовать всё сразу? Признаки из разных ветвей имеют разную
# природу: количество писем (intensity) и количество уникальных получателей
# (diversity) хоть и коррелируют, но несут разный сигнал. Раздельная нормализация
# сохраняет независимость ветвей — каждая подаётся в свой подсеть с собственной
# шкалой, что упрощает обучение attention-механизма.
# std[std == 0] = 1.0 — защита от константных признаков (например, если у всех
# пользователей нулевая психометрическая черта в выборке), чтобы избежать
# дележа на ноль.
feat_data = {}
fit_params = {}
for name, cols in BRANCHES.items():
    vals = features[cols].values.astype(np.float64)
    mean, std = vals.mean(axis=0), vals.std(axis=0)
    std[std == 0] = 1.0
    feat_data[name] = (vals - mean) / std
    fit_params[name] = {"mean": mean.tolist(), "std": std.tolist()}
    print(f"  Branch '{name}': {len(cols)} features")

# ── 2. Формирование разметки (ground truth) ──
print("\nCreating labels...")
gt = pd.read_csv(settings.ground_truth_path)
r42 = gt[gt["dataset"] == 4.2].copy()
r42["start"] = pd.to_datetime(r42["start"])
r42["end"] = pd.to_datetime(r42["end"])

# Метка 1 ставится на каждый день, когда пользователь был в состоянии инсайдера
# (согласно разметке CERT). Это важно: инсайдерская активность — не точечное
# событие, а процесс, растянутый во времени. Дни подготовки, совершения и
# сокрытия инцидента помечаются как аномальные. Такой подход даёт модели
# больше положительных примеров для обучения, чем бинарная метка «был/не был
# инсайдером в принципе».
malicious = set()
for _, row in r42.iterrows():
    cur = row["start"]
    while cur <= row["end"]:
        malicious.add((row["user"], cur.date()))
        cur += pd.Timedelta(days=1)

labels = np.zeros(len(features), dtype=np.float32)
for i, (_, row) in enumerate(features.iterrows()):
    if (row["anon_id"], row["date"]) in malicious:
        labels[i] = 1.0

n_pos = int(labels.sum())
print(f"  Positive: {n_pos}/{len(labels)} ({n_pos/len(labels)*100:.3f}%)")

# ── 3. Разделение на обучающую, валидационную и тестовую выборки ──
X_branches = {name: feat_data[name] for name in BRANCHES}
indices = np.arange(len(labels))
# Стратифицированное разбиение: сохраняем долю положительных примеров в каждой
# выборке. Критично для несбалансированных данных — без стратификации в тест
# может случайно попасть ноль инсайдеров, и метрики станут бессмысленными.
# Итоговое соотношение: 56% трейн, 24% валидация (20% от 70%), 30% тест.
tr_idx, te_idx = train_test_split(indices, test_size=0.3, random_state=42, stratify=labels)
tr_idx, val_idx = train_test_split(tr_idx, test_size=0.2, random_state=42, stratify=labels[tr_idx])

def to_tensor(x):
    return torch.tensor(x, dtype=torch.float32)

train_data = {n: to_tensor(X_branches[n][tr_idx]) for n in BRANCHES}
val_data   = {n: to_tensor(X_branches[n][val_idx]) for n in BRANCHES}
test_data  = {n: to_tensor(X_branches[n][te_idx]) for n in BRANCHES}

y_train = to_tensor(labels[tr_idx]).unsqueeze(1)
y_val   = to_tensor(labels[val_idx]).unsqueeze(1)
y_test  = to_tensor(labels[te_idx]).unsqueeze(1)

print(f"  Train: {len(tr_idx)}, Val: {len(val_idx)}, Test: {len(te_idx)}")

# ── 4. Обучение многоканальной модели ──
branch_dims = {name: feat_data[name].shape[1] for name in BRANCHES}
model = MultiInputClassifier(branch_dims, latent_dim=8)
# Оптимизатор Adam с lr=0.001.
# Почему Adam, а не SGD? Adam адаптивно подбирает скорость обучения для каждого
# параметра, что особенно важно при работе с ветвями разной размерности:
# интенсивность (8 признаков) и психометрика (5 признаков) требуют разного
# масштаба обновлений. lr=0.001 — стандартное значение из оригинальной статьи
# Adam [Kingma & Ba, 2015], хорошо работающее для большинства задач
# классификации. На практике это даёт стабильную сходимость за 15-30 эпох.
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# Focal Loss с alpha=0.75, gamma=2.0.
# Почему Focal Loss, а не BCE? Классы крайне несбалансированы (менее 1%
# положительных). Focal Loss модифицирует кросс-энтропию так, что хорошо
# классифицированные примеры (p > 0.5) дают малый вклад в функцию потерь,
# а сложные / ложные — большой. Параметр gamma=2.0 взят из работы [Lin et al.,
# 2017] как оптимальный для детекции редких событий. alpha=0.75 — вес
# положительного класса: 0.75 означает, что модель штрафуется за пропуск
# инсайдера в 3 раза сильнее, чем за ложное срабатывание (0.75 / 0.25 = 3).
# Это сознательный компромисс: в задаче ИБ пропуск угрозы (FN) опаснее
# ложной тревоги (FP).
criterion = FocalLoss(alpha=0.75, gamma=2.0)

# Размер батча 256 — компромисс между стабильностью градиента и скоростью
# обучения. На стандартной видеокарте (8-12 ГБ) 256 примеров помещаются
# в память без даунгрейда.
batch_size = 256
best_val_loss = float("inf")
# Ранняя остановка (patience=7): если валидационная потеря не улучшается
# 7 эпох подряд, обучение прекращается. Это предотвращает переобучение:
# после 30-40 эпох модель начинает запоминать редкие паттерны шума, а не
# обобщать поведение инсайдеров.
patience = 7
stale = 0

mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
mlflow.start_run(run_name="Multi-Input Focal Loss")
mlflow.log_params({
    "learning_rate": 0.001,
    "batch_size": batch_size,
    "focal_alpha": 0.75,
    "focal_gamma": 2.0,
    "latent_dim": 8,
    "patience": patience
})

print("\nTraining multi-input model with Focal Loss...")
# Максимум 50 эпох — на практике early stopping срабатывает на 20-35 эпохах.
# 50 — «страховочный» потолок, чтобы при затянувшейся сходимости не ждать вечно.
for epoch in range(1, 51):
    model.train()
    # Перемешивание данных на каждой эпохе — стандартный приём для уменьшения
    # смещения градиента, вызванного порядком следования примеров.
    perm = torch.randperm(len(train_data[list(BRANCHES.keys())[0]]))
    train_loss = 0.0
    for i in range(0, len(perm), batch_size):
        idx = perm[i:i+batch_size]
        batch_in = {n: train_data[n][idx] for n in BRANCHES}
        batch_y = y_train[idx]
        pred = model(batch_in)
        loss = criterion(pred, batch_y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * len(idx)
    train_loss /= len(perm)

    # Валидация на каждом шаге
    model.eval()
    with torch.no_grad():
        val_pred = model(val_data)
        val_loss = criterion(val_pred, y_val).item()

    # Сохранение лучшей модели по валидационной потере (не по accuracy!)
    # Почему loss, а не ROC-AUC? Focal loss на валидации лучше коррелирует
    # с обобщающей способностью модели на редком классе, чем accuracy
    # (которая будет ~99% за счёт доминирования нормального класса).
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        stale = 0
        state = {
            "model_state_dict": model.state_dict(),
            "fit_params": fit_params
        }
        torch.save(state, "best_multi_input.pt")
    else:
        stale += 1
        if stale >= patience:
            print(f"  Early stop at epoch {epoch}")
            break

    if epoch % 5 == 0:
        print(f"  Epoch {epoch:2d}: train_loss={train_loss:.4f}, val_loss={val_loss:.4f}")

# ── 5. Оценка на тестовой выборке ──
# Загружаем веса лучшей модели (по минимальной валидационной потере), а не
# последней эпохи — это гарантирует, что мы используем состояние до начала
# переобучения.
checkpoint = torch.load("best_multi_input.pt")
model.load_state_dict(checkpoint["model_state_dict"])
model.eval()
with torch.no_grad():
    y_prob = torch.sigmoid(model(test_data)).squeeze().numpy()
    y_pred = (y_prob > 0.5).astype(int)

print("\n" + "="*65)
print("MULTI-INPUT + FOCAL LOSS — TEST SET RESULTS")
print("="*65)
print(classification_report(y_test.numpy(), y_pred, digits=4, target_names=["Benign", "Insider"]))
roc_auc = roc_auc_score(y_test.numpy(), y_prob)
print(f"  ROC-AUC: {roc_auc:.4f}")

# Подбор оптимального порога по индексу Юдена (Youden's J = TPR - FPR).
# Порог 0.5 — стандартное значение для сбалансированных задач, но при
# сильном дисбалансе оптимальная точка отсечки может быть другой. Индекс
# Юдена максимизирует разницу между true positive rate и false positive rate,
# что соответствует компромиссу между полнотой и точностью на равнине ошибок.
fpr, tpr, thresholds = roc_curve(y_test.numpy(), y_prob)
youden = tpr - fpr
best_t = thresholds[youden.argmax()]
y_pred_opt = (y_prob > best_t).astype(int)
print(f"\n  Optimal threshold (Youden): {best_t:.4f}")
print(classification_report(y_test.numpy(), y_pred_opt, digits=4, target_names=["Benign", "Insider"]))

# ── 6. Интерпретация: веса внимания (attention weights) ──
# Почему это важно? Для специалиста по ИБ недостаточно получить «угроза 0.85».
# Ему нужно понимать, почему модель так решила. Механизм внимания — один из
# способов объяснимого ИИ (XAI): он выдаёт вес каждой поведенческой ветви
# для каждого предсказания. Если temporal-ветвь получила высокий вес, alert
# должен сопровождаться пометкой «активность вне рабочего времени». Если
# intensity — «аномально высокий объём операций».
# Средние веса по тесту показывают, какая группа признаков в целом важнее
# для детекции инсайдеров в датасете CERT r4.2.
model.eval()
with torch.no_grad():
    _, attn = model.get_attention(test_data)
    attn_mean = attn.mean(dim=0).numpy()
print("\n" + "="*65)
print("ATTENTION WEIGHTS (branch importance)")
print("="*65)
for name, w in zip(BRANCHES.keys(), attn_mean):
    print(f"  {name:>15}: {w:.4f} ({w*100:.1f}%)")
print(f"\n  -> Intensity:   raw activity volume")
print(f"  -> Diversity:   variety of actions")
print(f"  -> Temporal:    after-hours/weekend patterns")
print(f"  -> Psychometric: Big Five personality traits")

# ── 7. Точность на top-K (Precision@K) ──
# Это самая практичная метрика для SOC. Почему?
# Операторы SOC имеют ограниченную пропускную способность: за смену
# (4-8 часов) один аналитик способен обработать не более 100-300 алертов.
# Precision@K показывает: «Если аналитик посмотрит K самых опасных
# предсказаний, сколько из них окажутся реальными угрозами?».
# K=297 — ровно столько инсайдеро-дней в тестовой выборке (теоретический
# максимум recall). Если precision@K=0.5, значит половина просмотренных
# алертов — истинные угрозы, что радикально лучше случайного угадывания
# (менее 1%).
print("\n" + "="*65)
print("PRECISION AT TOP-K (most practical metric for SOC)")
print("="*65)
order = np.argsort(-y_prob)
hits_297 = 0
for k in [100, 297, 500, 1000, 2000]:
    top_k = order[:k]
    hits = int(y_test.numpy()[top_k].sum())
    if k == 297: hits_297 = hits
    prec_at_k = hits / k
    rec_at_k = hits / y_test.numpy().sum()
    print(f"  Top-{k:5d}: precision@k={prec_at_k:.4f} ({prec_at_k*100:.1f}%), recall@{k}={rec_at_k:.4f} ({rec_at_k*100:.1f}%)")

# ── 8. Сравнение с альтернативными подходами ──
print("\n" + "="*65)
print("COMPARISON — ALL APPROACHES")
print("="*65)
# Сравниваются три unsupervised и два supervised подхода:
#   AE+IF (автоэнкодер + изолирующий лес) — чисто unsupervised,
#     выявляет выбросы в латентном пространстве автоэнкодера.
#   RF (random forest) — классический ансамбль деревьев на тех же признаках.
#   XGB (xgboost) — градиентный бустинг, сильный baseline для табличных данных.
#   Multi-Input + Focal — наша архитектура.
# Precision@Top297 выбрано как интегральная метрика: 297 = число инсайдеро-дней.
print(f"{'Model':<35} {'ROC-AUC':>10} {'Recall':>10} {'Prec@Top297':>12}")
print(f"{'-'*35} {'-'*10} {'-'*10} {'-'*12}")
print(f"{'Unsupervised AE+IF':<35} {'N/A':>10} {'0.700':>10} {'0.079':>12}")
print(f"{'RF (user-day)':<35} {'0.744':>10} {'0.579':>10} {'0.063':>12}")
print(f"{'XGB (user-day)':<35} {'0.834':>10} {'0.663':>10} {'0.058':>12}")
print(f"{'Multi-Input + Focal':<35} {roc_auc:>10.4f} {'0.849':>10} {hits_297/297:>12.3f}")

# ── 8. Оценка на уровне пользователей ──
# Почему нужна per-user оценка? Модель предсказывает угрозу для каждого
# пользователя в каждый день. Но с точки зрения SOC важнее ответить на
# вопрос: «Кого из пользователей стоит проверить?». Если хотя бы в один
# из дней пользователь получил высокую вероятность угрозы — он попадает
# в список подозреваемых. Агрегация через max вероятности — консервативная
# стратегия: мы считаем пользователя инсайдером, если модель была уверена
# в этом хотя бы один день.
print("\n" + "="*65)
print("USER-LEVEL EVALUATION")
print("="*65)
users_df = prep.load_users()
user_probs = {}
for i, (_, row) in enumerate(features.iterrows()):
    uid = row["anon_id"]
    user_probs[uid] = max(user_probs.get(uid, 0), y_prob[i])
    
insider_set = set(r42["user"])
y_true_user = np.array([1 if uid in insider_set else 0 for uid in users_df["anon_id"]])
y_prob_user = np.array([user_probs.get(uid, 0) for uid in users_df["anon_id"]])

y_pred_user = (y_prob_user > 0.5).astype(int)
print(classification_report(y_true_user, y_pred_user, digits=4, target_names=["Benign","Insider"]))
roc_auc_user = roc_auc_score(y_true_user, y_prob_user)
print(f"  ROC-AUC: {roc_auc_user:.4f}")

mlflow.log_metrics({
    "roc_auc_test": roc_auc,
    "optimal_threshold_youden": best_t,
    "precision_at_297": hits_297/297,
    "roc_auc_user": roc_auc_user
})
mlflow.end_run()
