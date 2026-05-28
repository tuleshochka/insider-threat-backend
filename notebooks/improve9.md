# Глубокий анализ системы HST-UEBA v8/v9 и скрипта подготовки данных

## 📊 Общая оценка архитектуры

Система реализует иерархический подход: **событие → день → окно (7 дней)**, что концептуально верно для задачи обнаружения инсайдерских угроз. Однако есть ряд существенных проблем и точек роста.

---

## 🔴 Критические проблемы

### 1. Скрипт подготовки данных (`prepare_nlp_dataset.py`)

#### Проблема утечки разметки (Data Leakage) в Suspicious Trimming
```python
# ТЕКУЩИЙ КОД (строки 238-261)
for ts, hour, token_id, is_mal in events:  # ← is_mal используется в цикле!
    score = 0
    if token_id in [4, 5, 10, 11, 12, 13, 15, 17]:
        score += 3
    if hour < 8 or hour > 18:
        score += 1
    scores.append(score)
```

**Проблема:** Хотя `is_mal` напрямую не используется в вычислении `score`, итерация `for ts, hour, token_id, is_mal in events` создаёт контекст, где скор опирается на **те же токены, которые являются маркерами атак** (USB, внешний email, HTTP_UPLOAD). Алгоритм тримминга косвенно выбирает окно вокруг наиболее "подозрительных" событий, что завышает метрики.

```python
# ИСПРАВЛЕНИЕ: скор должен быть независим от разметки
def compute_trim_score(token_id, hour):
    """Только эвристики, не коррелирующие с разметкой"""
    score = 0
    if token_id in [4, 5, 10, 11, 12, 13, 15, 17]:  # аномальные типы
        score += 3
    if hour < 8 or hour > 18:  # нерабочее время
        score += 1
    return score
# Важно: вычислять ДО формирования меток, без доступа к is_mal
```

#### Непоследовательность разбиения пользователей
```python
# В prepare_nlp_dataset.py (строка 172)
np.random.seed(42)
shuffled_users = unique_users.copy()
np.random.shuffle(shuffled_users)
train_users = set(shuffled_users[:train_split])  # только 70%

# В notebook (строки разбиения)
np.random.seed(42)
# Та же логика, НО unique_users берётся из users (всего массива окон)
# а не из отсортированного списка уникальных пользователей
```

**Риск:** Если порядок `unique_users` отличается между скриптом и ноутбуком, пользователи train/val/test могут не совпадать, что нарушает изоляцию выборок.

#### Z-score нормализация для unseen пользователей
```python
# Строка 221-223
safe_std = np.maximum(std, 1.0)  # ← жёсткий порог 1.0 без обоснования
dev_z = (daily_raw - mean) / safe_std
dev_z = np.clip(dev_z, -10.0, 10.0)
```

**Проблемы:**
- `safe_std = np.maximum(std, 1.0)` — порог 1.0 произвольный; для пользователей с малым числом дней std будет нулевым и заменяется на 1.0, что даёт некорректные z-score
- Для unseen пользователей используется `global_mean/global_std`, вычисленный как среднее средних, а не средневзвешенное — это статистически некорректно

```python
# УЛУЧШЕНИЕ: взвешенное среднее по числу наблюдений
weighted_means = []
weighted_stds = []
for user, feats_list in train_user_features.items():
    n = len(feats_list)
    weighted_means.append((n, np.mean(feats_list, axis=0)))
    weighted_stds.append((n, np.std(feats_list, axis=0)))

total_n = sum(w for w, _ in weighted_means)
global_mean = sum(w * m for w, m in weighted_means) / total_n
# Аналогично для std через формулу объединённой дисперсии
```

#### Отсутствие проверки временной корректности окон
```python
# В notebook при формировании скользящих окон
for i in range(len(indices) - WINDOW_SIZE + 1):
    w_idx = indices[i:i+WINDOW_SIZE]
```

**Проблема:** Нет проверки, что дни в окне идут подряд без пропусков. Если пользователь отсутствовал 3 дня (отпуск, командировка), окно объединит несмежные периоды, что создаёт артефакты.

```python
# ИСПРАВЛЕНИЕ
for i in range(len(indices) - WINDOW_SIZE + 1):
    w_dates = g_dates[i:i+WINDOW_SIZE]
    # Проверяем непрерывность (допускаем пропуск выходных)
    date_diffs = np.diff([pd.Timestamp(d) for d in w_dates])
    max_gap = max(d.days for d in date_diffs)
    if max_gap > 3:  # пропуск более 3 дней — пропускаем окно
        continue
```

---

### 2. Архитектура модели (`insider_threat_nlp_training`)

#### Некорректная обработка BiLSTM hidden states
```python
# Строки в forward()
if h_n.shape[0] == 2:
    day_repr = torch.cat([h_n[0], h_n[1]], dim=1)
else:
    day_repr = torch.cat([h_n[-2], h_n[-1]], dim=1)
```

**Проблема:** При `num_lstm_layers=2` и `bidirectional=True`, `h_n.shape[0] = 4` (2 слоя × 2 направления). Код берёт `h_n[-2]` и `h_n[-1]` — только последний слой, но формулировка условия `if h_n.shape[0] == 2` никогда не выполняется при многослойном BiLSTM с `num_lstm_layers=2`. Это скрытый баг.

```python
# ИСПРАВЛЕНИЕ: явно извлекаем последний слой BiLSTM
num_directions = 2  # bidirectional
# h_n shape: (num_layers * num_directions, batch, hidden//2)
# Последний слой: индексы [-2] и [-1] для forward и backward
last_forward = h_n[-2]   # последний слой, прямое направление
last_backward = h_n[-1]  # последний слой, обратное направление
day_repr = torch.cat([last_forward, last_backward], dim=1)
# Это правильно, но условие нужно переписать чётче:
day_repr = torch.cat([h_n[-(num_directions):][0], 
                      h_n[-(num_directions):][1]], dim=1)
```

#### Потенциальный information leak в Ablation Study
```python
# Строка в ablation
ab_model.eval()
# Оценка на val_loader — ВСЕХ валидационных пользователей
# Но ablation обучается только на 15 пользователях!
```

**Проблема:** Ablation Study обучается на 15 пользователях из train, но валидируется на полном val_loader. Это нарушает сравнимость результатов — модель видит пользователей, которых никогда не обучалась, и результаты ablation некорректны.

#### Дублирование inference на test set
```python
# В секции 5 (Multiple Seeds)
test_probs, test_true = [], []
# ... первый inference на test

# В секции 7 (финальная оценка)
test_probs, test_true = [], []
# ... ВТОРОЙ inference на test (те же переменные)
```

Это не баг, но увеличивает время выполнения и создаёт путаницу.

---

## 🟡 Существенные недостатки

### 3. Скрипт подготовки данных

#### Словарь токенов слишком мал
```python
TOKEN_MAP = {
    "<PAD>": 0, "<UNK>": 1, "LOGON": 2, ..., "HTTP_DOWNLOAD": 18
}
VOCAB_SIZE = 19  # только 19 токенов
```

**Проблема:** 19 токенов — крайне мало. Теряется информация о:
- Конкретных хостах/ресурсах (доступ к критичным папкам)
- Типах файлов (`.pdf`, `.docx` vs `.exe`)
- Количестве получателей email (1 получатель vs 50)
- Объёме данных (размер файла, объём HTTP трафика)

```python
# УЛУЧШЕНИЕ: расширить словарь с учётом контекста
TOKEN_MAP_EXTENDED = {
    # Существующие токены...
    # Добавить:
    "FILE_BULK_COPY": 19,        # копирование >10 файлов за раз
    "EMAIL_SEND_EXT_BULK": 20,   # email >5 получателей
    "HTTP_UPLOAD_LARGE": 21,     # загрузка >10МБ
    "LOGON_AFTER_HOURS": 22,     # вход после 20:00
    "LOGON_WEEKEND": 23,         # вход в выходные
    "USB_LARGE_TRANSFER": 24,    # >100МБ на USB
}
```

#### Определение внешних доменов — хардкод
```python
# Строка 93
INTERNAL_DOMAINS = {"@dtaa.com", "@dtaa.org"}
```

Домены захардкожены, что делает скрипт непереносимым. Нужно вынести в конфиг:

```python
# config.yaml или в начало скрипта
INTERNAL_DOMAINS = set(os.environ.get(
    "INTERNAL_DOMAINS", "@dtaa.com,@dtaa.org"
).split(","))
```

#### Отсутствие нормализации TSLE перед сохранением
```python
# В скрипте: tsle сохраняется как сырые интервалы в секундах
tsle_s.append(0.0 if prev_ts is None else (ts - prev_ts))
```

Интервалы могут быть от 0 до 86400 секунд. В ноутбуке применяется `log1p` и z-нормализация, но это создаёт двойную зависимость — нормализация должна быть в скрипте или только в ноутбуке.

---

### 4. Архитектура модели

#### Только 10 эпох обучения
```python
NUM_EPOCHS = 10
PATIENCE = 3
```

С `PATIENCE=3` обучение может остановиться на 4-й эпохе. Для задачи с сильным дисбалансом классов нужно больше:

```python
NUM_EPOCHS = 30
PATIENCE = 5
# + добавить warmup scheduler
scheduler = torch.optim.lr_scheduler.OneCycleLR(
    optimizer, max_lr=1e-3, 
    steps_per_epoch=len(train_loader), epochs=NUM_EPOCHS
)
```

#### Фиксированный порог классификации из validation → test
```python
# Порог ищется на val, применяется на test — это корректно
# НО: порог ищется через F2, а не через PR-AUC
for t in np.linspace(0.01, 0.99, 100):
    f2 = fbeta_score(val_labels, preds, beta=2, ...)
```

Для систем безопасности (UEBA) более правильно искать порог через **Recall@FixedFPR** (например, 1% FPR), что соответствует реальным операционным требованиям SOC.

#### Attentive Pooling имеет конфликт интересов
```python
class AttentivePooling(nn.Module):
    def forward(self, trans_out):
        attn_scores = torch.softmax(self.attention(trans_out).squeeze(-1), dim=1)
        weighted = (trans_out * attn_scores.unsqueeze(-1)).sum(dim=1)
        last_day = trans_out[:, -1, :]
        w = torch.sigmoid(self.last_day_weight)
        return (1.0 - w) * weighted + w * last_day
```

**Проблема:** `self.attention` (обучаемый) и `self.last_day_weight` (фиксированный по смыслу) конкурируют. Если attention научится давать высокий вес последнему дню, `w * last_day` будет дублировать его вклад.

#### Occlusion XAI — слишком медленный для production
```python
def calculate_occlusion_importance(...):
    for day in range(WINDOW_SIZE):  # 7 инференсов на каждый пример
        # ...
```

7 инференсов на пример × N примеров = очень медленно. Для диплома ОК, но нужно упомянуть ограничение.

---

## 🟢 Конкретные рекомендации по улучшению

### Улучшение 1: Исправление разбиения пользователей (критично)

```python
# prepare_nlp_dataset.py — сохранять информацию о разбиении
split_info = {
    'train_users': list(train_users),
    'val_users': list(val_users_set),  # добавить
    'test_users': list(test_users_set),  # добавить
    'seed': 42
}
# Сохранять вместе с .npz как отдельный JSON
with open(OUTPUT_FILE.replace('.npz', '_split.json'), 'w') as f:
    json.dump(split_info, f)

# В ноутбуке — загружать готовое разбиение
with open('cert_nlp_sequences_split.json') as f:
    split_info = json.load(f)
train_users = np.array(split_info['train_users'])
```

### Улучшение 2: Добавить признаки дрейфа поведения (Behavioral Drift)

```python
# В скрипте подготовки: скользящее среднее за 30 дней
def compute_behavioral_drift(user_day_events, window=30):
    """
    Для каждого дня вычисляет отклонение от 30-дневной скользящей базы
    Это более чувствительно к постепенному изменению поведения
    """
    drift_features = {}
    for user in unique_users:
        user_days = sorted([d for u, d in user_day_events if u == user])
        rolling_counts = []
        for i, day in enumerate(user_days):
            # 30-дневное окно до текущего дня (не включая его!)
            window_days = user_days[max(0, i-window):i]
            if window_days:
                baseline = np.mean([
                    raw_features.get((user, d), np.zeros(4)) 
                    for d in window_days
                ], axis=0)
                current = raw_features.get((user, day), np.zeros(4))
                drift = current - baseline  # абсолютный дрейф
            else:
                drift = np.zeros(4)
            drift_features[(user, day)] = drift
    return drift_features
```

### Улучшение 3: Добавить контекстные эмбеддинги ролей пользователей

```python
# В HierarchicalTransformer
class HierarchicalTransformerV9(nn.Module):
    def __init__(self, ..., num_roles=10):
        # Добавить эмбеддинг роли пользователя (IT, Finance, HR, etc.)
        self.role_embedding = nn.Embedding(num_roles + 1, hidden_dim // 4)
        # Обновить day_proj для учёта роли
        self.day_proj = nn.Linear(hidden_dim + 4 + hidden_dim // 4, hidden_dim)
```

### Улучшение 4: Правильная метрика для SOC-контекста

```python
# Добавить метрику MTTD (Mean Time To Detect)
def compute_mttd(test_true, test_probs, win_dates, threshold):
    """
    Среднее время от начала атаки до первого обнаружения
    Ключевая метрика для систем ИБ
    """
    mttd_values = []
    # Группируем по пользователям
    for user in np.unique(win_u[test_idx]):
        user_mask = win_u[test_idx] == user
        user_labels = test_true[user_mask]
        user_probs = test_probs[user_mask]
        user_dates = win_dates[test_idx][user_mask]
        
        if user_labels.sum() == 0:
            continue
        
        # Находим первый день атаки
        attack_start = None
        for i, (label, date) in enumerate(zip(user_labels, user_dates)):
            if label == 1:
                attack_start = pd.Timestamp(date[-1])
                break
        
        # Находим первое обнаружение
        for i, (prob, date) in enumerate(zip(user_probs, user_dates)):
            if prob >= threshold:
                detection_time = pd.Timestamp(date[-1])
                if attack_start:
                    mttd_values.append(
                        (detection_time - attack_start).days
                    )
                break
    
    return np.mean(mttd_values) if mttd_values else float('inf')
```

### Улучшение 5: Добавить Graph-based признаки (коммуникационный граф)

```python
# В скрипте подготовки: анализ графа коммуникаций
def build_email_graph_features(email_events_by_user):
    """
    Признаки из графа email-коммуникаций:
    - степень вершины (число уникальных контактов)
    - новые контакты за день (не из исторического графа)
    - число внешних контактов относительно нормы
    """
    import networkx as nx
    G = nx.DiGraph()
    graph_features = {}
    
    for user, events in email_events_by_user.items():
        for ts, recipient, token_id in events:
            if token_id == 15:  # EMAIL_SEND_EXT
                G.add_edge(user, recipient, 
                          timestamp=ts, weight=1)
    
    for user in G.nodes():
        # Центральность как признак аномальности
        centrality = nx.degree_centrality(G)[user]
        out_degree = G.out_degree(user)
        graph_features[user] = {
            'centrality': centrality,
            'out_degree': out_degree,
            'external_contacts': len([
                n for n in G.successors(user) 
                if not n.endswith('@dtaa.com')
            ])
        }
    return graph_features
```

### Улучшение 6: Исправить Ablation Study

```python
# Выделить отдельный ablation-val из ablation_users
ablation_users_train = train_users[:12]
ablation_users_val = train_users[12:15]  # не пересекается с основным val

ablation_val_idx = np.where(np.isin(win_u, ablation_users_val))[0]
ablation_val_loader = DataLoader(
    HierarchicalDataset(...win_X[ablation_val_idx]...),
    batch_size=BATCH_SIZE, shuffle=False
)
# Теперь оценивать на ablation_val_loader, а не на val_loader
```

### Улучшение 7: Добавить онлайн-обновление базовых профилей

```python
class OnlineUserProfileUpdater:
    """
    Экспоненциальное скользящее среднее для обновления профилей
    в продакшн-режиме без переобучения модели
    """
    def __init__(self, alpha=0.05):
        self.alpha = alpha  # скорость обновления
        self.profiles = {}
    
    def update(self, user: str, new_features: np.ndarray):
        if user not in self.profiles:
            self.profiles[user] = {
                'mean': new_features.copy(),
                'variance': np.ones_like(new_features)
            }
        else:
            old_mean = self.profiles[user]['mean']
            # EMA обновление
            self.profiles[user]['mean'] = (
                (1 - self.alpha) * old_mean + self.alpha * new_features
            )
            # Обновление дисперсии через Welford's algorithm
            delta = new_features - old_mean
            self.profiles[user]['variance'] = (
                (1 - self.alpha) * self.profiles[user]['variance'] 
                + self.alpha * delta**2
            )
    
    def get_zscore(self, user: str, features: np.ndarray) -> np.ndarray:
        if user not in self.profiles:
            return np.zeros_like(features)
        mean = self.profiles[user]['mean']
        std = np.sqrt(self.profiles[user]['variance']) + 1e-8
        return np.clip((features - mean) / std, -10, 10)
```

---

## 📋 Сводная таблица проблем и приоритетов

| # | Проблема | Критичность | Компонент | Трудозатраты |
|---|----------|-------------|-----------|--------------|
| 1 | Несогласованное разбиение пользователей между скриптом и ноутбуком | 🔴 Критично | Оба | 2 часа |
| 2 | Баг в извлечении hidden state BiLSTM при 2 слоях | 🔴 Критично | Ноутбук | 30 мин |
| 3 | Ablation Study оценивается на несовместимом val | 🟠 Высокая | Ноутбук | 1 час |
| 4 | Отсутствие проверки непрерывности дат в окнах | 🟠 Высокая | Ноутбук | 1 час |
| 5 | Z-score нормализация с порогом std=1.0 без обоснования | 🟠 Высокая | Скрипт | 30 мин |
| 6 | Малый словарь токенов (19 токенов) | 🟡 Средняя | Скрипт | 3 часа |
| 7 | Хардкод внутренних доменов | 🟡 Средняя | Скрипт | 30 мин |
| 8 | Только 10 эпох, PATIENCE=3 | 🟡 Средняя | Ноутбук | 30 мин |
| 9 | Отсутствие метрики MTTD | 🟡 Средняя | Ноутбук | 2 часа |
| 10 | Нет онлайн-обновления профилей | 🟢 Низкая | Скрипт | 4 часа |

---

## 🎯 Рекомендации для диплома

### Что обязательно исправить (для защиты):
1. **Согласовать разбиение** — сохранять split.json из скрипта и загружать в ноутбук
2. **Исправить BiLSTM баг** — явно указать извлечение последнего слоя
3. **Добавить MTTD метрику** — это ключевая метрика для систем безопасности
4. **Зафиксировать порядок данных** — добавить сортировку по (user, date) перед сохранением

### Что добавить для сильной защиты:
1. **Сравнение с LSTM-only baseline** — показать вклад Transformer
2. **Анализ false positives по дням недели** — понедельник/пятница vs среда
3. **ROC-кривая по сценариям** — не только общая, но и per-scenario
4. **Калибровка вероятностей** — Platt Scaling или Isotonic Regression для более надёжных порогов

### Формулировка для диплома:
> Система HST-UEBA реализует иерархическую двухуровневую архитектуру: BiLSTM для кодирования суточных поведенческих паттернов на уровне токенизированных событий и Transformer Encoder для анализа 7-дневных поведенческих траекторий. Ключевым отличием от существующих подходов является интеграция Z-score отклонений от персонализированных поведенческих базелайнов непосредственно в архитектуру модели, а также применение Causal Occlusion XAI для верификации причинно-следственных связей между аномальными днями и итоговым решением классификатора.