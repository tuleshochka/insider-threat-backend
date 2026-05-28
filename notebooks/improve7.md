# Глубокий анализ системы HST-UEBA v7

## 1. ОБЩАЯ АРХИТЕКТУРА РЕШЕНИЯ

Ваше решение состоит из двух ключевых компонентов:
- **Скрипт подготовки данных** (`prepare_nlp_dataset.py`) — ETL-конвейер из сырых логов CERT r5.2 в токенизированные последовательности
- **Обучающий ноутбук** — иерархическая модель BiLSTM → Transformer с двойным XAI

Подход хорошо структурирован, но имеет ряд существенных проблем — от багов до архитектурных ограничений. Ниже разбираю всё по уровням.

---

## 2. АНАЛИЗ СКРИПТА ПОДГОТОВКИ ДАННЫХ

### 2.1. Что сделано хорошо ✅

| Аспект | Описание |
|--------|----------|
| Токенизация | 19 семантических токенов покрывают основные типы активности |
| Микро-признаки | hour, is_working_hours, TSLE — отличный набор временных фич |
| Сортировка | Критически важная сортировка по timestamp внутри дня |
| Обрезка | Берутся ПОСЛЕДНИЕ события при превышении MAX_SEQ_LEN — правильный выбор для обнаружения угроз |
| USB-разделение | Файловые операции с USB и без неё различаются — важно для сценариев хищения |

### 2.2. Проблемы и баги 🐛

**Проблема 1: Мёртвый код — `malicious_user_dates` не используется**
```python
malicious_user_dates = set()  # Строка 30
# ... заполняется в цикле, но НИГДЕ не используется при формировании labels
```
Переменная заполняется, но метка `has_malicious` вычисляется через `malicious_event_ids` на уровне отдельных событий. Это правильно, но `malicious_user_dates` — мёртвый код. Его можно было бы использовать для **верификации** или для **дополнительной метки на уровне дня** (мягкая метка).

**Проблема 2: Потеря информации при обрезке вредоносных дней**
```python
if len(seq) > MAX_SEQ_LEN:
    seq = seq[-MAX_SEQ_LEN:]  # Берём последние 200 событий
```
Если вредоносное событие произошло в начале дня и за ним следует > 200 нормальных событий, **вредоносный токен будет обрезан**, но метка `has_malicious` всё равно будет = 1. Модель получит «чистую» последовательность с положительной меткой → **шум в обучении**.

**Исправление:**
```python
# Перед обрезкой проверяем, остались ли вредоносные события
if len(seq) > MAX_SEQ_LEN:
    # Проверяем, не теряем ли malicious-события
    kept_events = events[-MAX_SEQ_LEN:]
    has_malicious_after_trim = any(e[3] for e in kept_events)
    if has_malicious != has_malicious_after_trim:
        # Вредоносное событие было обрезано — сдвигаем окно, 
        # чтобы включить его
        mal_indices = [i for i, e in enumerate(events) if e[3]]
        if mal_indices:
            start = max(0, mal_indices[0] - 10)  # контекст до атаки
            events_window = events[start:start+MAX_SEQ_LEN]
```

**Проблема 3: Нет агрегатных признаков дня**

Текущий формат сохраняет только сырую последовательность токенов. Но для модели были бы полезны **статистические признаки дня**:
- Общее число событий
- Число USB-операций
- Число внешних email
- Число событий вне рабочих часов
- Энтропия распределения токенов

**Проблема 4: Игнорируются данные LDAP и psychometric из CERT r5.2**

Набор данных CERT r5.2 содержит:
- `LDAP/` — информация об отделах, ролях, руководителях
- `psychometric.csv` — психометрические профили сотрудников (OCEAN)

Эти данные **вообще не используются**, хотя они критически важны для контекста: увольняющийся сотрудник (видно из LDAP) + высокий нейротизм (psychometric) = повышенный риск.

**Проблема 5: Примитивная классификация HTTP-трафика**
```python
if any(kw in url for kw in ["upload", "dropbox", "drive.google"...]):
    token = "HTTP_UPLOAD"
```
Это жёстко заданные ключевые слова. Не обнаруживаются:
- Новые облачные хранилища
- Tor/VPN-сайты
- Сайты поиска работы (индикатор увольнения в CERT)

**Улучшение:** Добавить токены `HTTP_JOBSEARCH`, `HTTP_CLOUD_STORAGE`, `HTTP_SUSPICIOUS`, используя расширенные списки и regex.

**Проблема 6: Нет межпользовательских признаков**

Не учитываются:
- Email-коммуникации между пользователями (граф)
- Общие файлы
- Нетипичная PC-активность (вход с чужого компьютера)

### 2.3. Рекомендации по улучшению скрипта

```python
# === ДОБАВИТЬ: Агрегатные признаки дня ===
day_stats = {
    'total_events': len(events),
    'usb_events': sum(1 for e in events if e[2] in [4,5,10,11,12,13]),
    'after_hours_events': sum(1 for e in events if not (8 <= e[1] <= 18)),
    'external_emails': sum(1 for e in events if e[2] == 15),
    'unique_token_types': len(set(e[2] for e in events)),
    'avg_tsle': np.mean([e[0] for e in events[1:]] - [e[0] for e in events[:-1]]) 
        if len(events) > 1 else 0,
}
# Сохранить как отдельный массив X_day_stats
```

```python
# === ДОБАВИТЬ: N-gram токены для контекста ===
# Вместо одиночных токенов, создать bigram-пары
bigrams = []
for i in range(len(seq)-1):
    bigram = f"{seq[i]}_{seq[i+1]}"
    bigrams.append(BIGRAM_MAP.get(bigram, 0))
```

---

## 3. АНАЛИЗ ОБУЧАЮЩЕГО НОУТБУКА

### 3.1. Что сделано хорошо ✅

| Компонент | Оценка |
|-----------|--------|
| Split по пользователям | ✅ Строгое разделение без утечки данных |
| Z-нормализация TSLE по train | ✅ Правильно, нет data leakage |
| Focal Loss (α=0.85, γ=2.0) | ✅ Адекватный выбор для сильного дисбаланса |
| Pack_padded_sequence | ✅ Эффективная обработка переменной длины |
| Маскирование Transformer | ✅ Корректная фильтрация пустых дней |
| F2-порог на validation | ✅ Правильный подход — Recall важнее Precision |
| Сохранение нормализации | ✅ Критично для деплоя |
| Dual XAI | ✅ Attention + Occlusion — хорошая комбинация |

### 3.2. Критические проблемы архитектуры 🔴

**Проблема 1: Параметр `last_day_weight` не ограничен**
```python
self.last_day_weight = nn.Parameter(torch.tensor(0.3))
# ...
return (1.0 - self.last_day_weight) * weighted + self.last_day_weight * last_day
```
Значение `last_day_weight` может стать **отрицательным или > 1** через градиенты, что приведёт к нестабильности.

**Исправление:**
```python
def forward(self, trans_out):
    w = torch.sigmoid(self.last_day_weight)  # Ограничиваем [0, 1]
    attn_scores = torch.softmax(self.attention(trans_out).squeeze(-1), dim=1)
    weighted = (trans_out * attn_scores.unsqueeze(-1)).sum(dim=1)
    last_day = trans_out[:, -1, :]
    return (1.0 - w) * weighted + w * last_day
```

**Проблема 2: BiLSTM без dropout (1 слой)**
```python
self.day_lstm = nn.LSTM(lstm_input_dim, hidden_dim//2, 
                         batch_first=True, bidirectional=True)
# dropout в LSTM работает только при num_layers > 1
```
**Исправление:**
```python
self.day_lstm = nn.LSTM(lstm_input_dim, hidden_dim//2, 
                         batch_first=True, bidirectional=True,
                         num_layers=2, dropout=0.2)
# И обновить извлечение h_n:
day_repr = torch.cat([h_n[-2], h_n[-1]], dim=1)  # последние forward/backward
```

**Проблема 3: Нет LayerNorm перед Transformer**
```python
seq_repr = day_repr.view(B, W, -1)
seq_repr = seq_repr + self.dow_embedding(dow) + self.pos_encoder
# Прямо подаётся в Transformer без нормализации
```
Transformer чувствителен к масштабу входных данных. BiLSTM выходы + embeddings + positional могут быть в разных масштабах.

**Исправление:**
```python
self.input_norm = nn.LayerNorm(hidden_dim)
# ...
seq_repr = self.input_norm(seq_repr + self.dow_embedding(dow) + self.pos_encoder)
```

**Проблема 4: Классификатор слишком простой**
```python
self.classifier = nn.Sequential(
    nn.Linear(hidden_dim, 64),
    nn.ReLU(),
    nn.Dropout(0.3),
    nn.Linear(64, 1)
)
```
128→64→1 — это один скрытый слой. Для сложной задачи обнаружения инсайдеров недостаточно.

**Улучшение:**
```python
self.classifier = nn.Sequential(
    nn.Linear(hidden_dim, hidden_dim),
    nn.GELU(),
    nn.Dropout(0.3),
    nn.Linear(hidden_dim, 64),
    nn.GELU(),
    nn.Dropout(0.2),
    nn.Linear(64, 1)
)
```

**Проблема 5: Позиционное кодирование — обучаемое, маленькое**
```python
self.pos_encoder = nn.Parameter(torch.randn(1, window_size, hidden_dim) * 0.02)
```
Масштаб 0.02 слишком мал — позиционная информация будет «утоплена» в шуме. Для окна 7 дней лучше использовать **синусоидальное кодирование** или увеличить масштаб.

```python
# Синусоидальное позиционное кодирование
def sinusoidal_positional_encoding(window_size, hidden_dim):
    pe = torch.zeros(window_size, hidden_dim)
    position = torch.arange(0, window_size).unsqueeze(1).float()
    div_term = torch.exp(torch.arange(0, hidden_dim, 2).float() * 
                         -(np.log(10000.0) / hidden_dim))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe.unsqueeze(0)
```

### 3.3. Проблемы обучения 🟡

**Проблема 6: Нет learning rate warmup**
```python
optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
```
Начальный lr=1e-3 может быть слишком агрессивным для Transformer.

**Улучшение:**
```python
from torch.optim.lr_scheduler import OneCycleLR

scheduler = OneCycleLR(
    optimizer, 
    max_lr=1e-3,
    epochs=NUM_EPOCHS,
    steps_per_epoch=len(train_loader),
    pct_start=0.1,  # 10% warmup
    anneal_strategy='cos'
)
```

**Проблема 7: Нет аугментации данных для миноритарного класса**

При экстремальном дисбалансе (доля атак < 1%) Focal Loss помогает, но недостаточно.

```python
# Аугментация: временной сдвиг окна для позитивных примеров
def augment_positive_windows(win_X, win_y, win_u, ...):
    pos_idx = np.where(win_y == 1)[0]
    augmented = []
    for idx in pos_idx:
        # Случайная перестановка порядка событий внутри дня
        aug = win_X[idx].copy()
        random_day = np.random.randint(0, WINDOW_SIZE-1)  # не трогаем последний
        np.random.shuffle(aug[random_day])
        augmented.append(aug)
    return np.concatenate([win_X, np.array(augmented)])
```

**Проблема 8: 15 эпох — потенциально мало**
```python
NUM_EPOCHS = 15
PATIENCE = 4
```
С Early Stopping на 4 эпохи модель может остановиться на 8-9 эпохе, не достигнув оптимума. Рекомендуется увеличить:
```python
NUM_EPOCHS = 30
PATIENCE = 6
```

### 3.4. Проблемы оценки 🟠

**Проблема 9: Баг в коде чтения сценариев**
```python
scen_path = os.path.join(scen_path, fname) if 'fname' in locals() else os.path.join(ANSWERS_DIR, scen)
```
Это **явный баг**: `'fname' in locals()` проверяет наличие переменной из **внешнего** цикла предыдущей итерации, а `scen_path` перезаписывается. При второй итерации `scen_path` уже содержит путь от предыдущей итерации.

**Исправление:**
```python
for scen in scenarios:
    scen_path = os.path.join(ANSWERS_DIR, scen)  # Всегда от корня
    for fname in os.listdir(scen_path):
        ...
```

**Проблема 10: Хардкодированные «репрезентативные» метрики**
```python
rep_scens = [
    ('r5.2-1 (хищение через USB)', 45, 8, 0.8491),  # Откуда эти числа?
    ...
]
```
Эти числа **не вычислены моделью**, а захардкожены. Для дипломной работы это **недопустимо** — комиссия может задать вопрос о происхождении.

**Проблема 11: XAI на 3 примерах не репрезентативен**

3 True Positives и 3 False Positives — это слишком мало для выводов. Нужно:
```python
N_EXAMPLES = 50  # Минимум для статистической значимости
# Агрегировать occlusion importances по всем TP/FP
mean_tp_importance = np.mean([imp for _, _, imp in tps_found], axis=0)
std_tp_importance = np.std([imp for _, _, imp in tps_found], axis=0)
```

**Проблема 12: Бейслайны не обучены, а взяты из воздуха**

Метрики Isolation Forest, Autoencoder и т.д. захардкожены. Для честного сравнения необходимо **обучить эти модели на тех же данных** с тем же split'ом пользователей.

### 3.5. Проблемы скользящих окон

**Проблема 13: Огромный дисбаланс после формирования окон**

Если у пользователя 365 дней и 2 дня с атаками, формируется ~359 окон, из которых только 2 помечены как атака. Это соотношение ~180:1.

**Улучшение — Разреженная выборка нормальных окон:**
```python
# Для каждого пользователя-неатакующего оставляем не все окна, а подвыборку
MAX_NORMAL_WINDOWS_PER_USER = 50
for user, group in df.groupby('user'):
    ...
    normal_windows = [(i, w) for i, w in enumerate(windows) if w_label == 0]
    if len(normal_windows) > MAX_NORMAL_WINDOWS_PER_USER:
        normal_windows = random.sample(normal_windows, MAX_NORMAL_WINDOWS_PER_USER)
```

**Проблема 14: Окна не учитывают пропуски дней**

Если пользователь не работал в субботу/воскресенье, но его записи идут [Пт, Пн, Вт...], скользящее окно создаст окно [Чт, Пт, Пн, Вт, Ср, Чт, Пт] — без выходных. Это может быть проблемой, так как «7 последовательных записей» ≠ «7 календарных дней».

**Улучшение:**
```python
# Проверяем непрерывность дат в окне
dates_in_window = group['date'].values[i:i+WINDOW_SIZE]
date_range = (dates_in_window[-1] - dates_in_window[0]).days
if date_range > WINDOW_SIZE + 3:  # Допускаем пропуск до 3 дней
    continue  # Слишком большой разрыв — пропускаем окно
```

---

## 4. РЕКОМЕНДУЕМЫЕ СИСТЕМНЫЕ УЛУЧШЕНИЯ

### 4.1. Расширение набора признаков

```
Текущие:     токен, час, рабочие часы, TSLE, день недели
Добавить:    
├── Персональный контекст
│   ├── Роль (из LDAP: admin/developer/manager)
│   ├── Отдел
│   ├── Дата увольнения (если указана)
│   └── Психометрический профиль (OCEAN)
├── Поведенческие аномалии
│   ├── Отклонение от персонального базового поведения
│   ├── Число событий vs. медиана по отделу
│   └── Z-score числа USB-операций за день
├── Сетевые признаки
│   ├── Число уникальных получателей email
│   ├── Новые домены в HTTP (не встречались ранее)
│   └── Вход с нетипичного ПК
└── Темпоральные паттерны
    ├── Время первого LOGON / последнего LOGOFF
    ├── Длительность сессии
    └── Отношение after-hours к in-hours событий
```

### 4.2. Улучшение архитектуры модели

```
Предложенная архитектура v8:

                    ┌──────────────────┐
                    │  Персональные    │
                    │  признаки (LDAP) │
                    └────────┬─────────┘
                             │
Day Events ──► BiLSTM ──►┌───┴───┐
                         │ Concat │──► LayerNorm ──► Transformer ──►
Day Stats  ──► MLP   ──►└───────┘                                  │
                                                     Attentive     │
                                                     Pooling ◄─────┘
                                                        │
                                                   Classifier
                                                        │
                                                    P(insider)
```

```python
class HierarchicalTransformerV8(nn.Module):
    def __init__(self, ...):
        super().__init__()
        # Day encoder (как есть)
        ...
        
        # NEW: Day statistics encoder
        self.day_stats_mlp = nn.Sequential(
            nn.Linear(num_day_features, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, hidden_dim)
        )
        
        # NEW: User context encoder
        self.user_embed = nn.Sequential(
            nn.Linear(num_user_features, hidden_dim // 4),
            nn.GELU(),
        )
        
        # Fusion
        self.fusion = nn.Linear(hidden_dim + hidden_dim + hidden_dim // 4, hidden_dim)
        self.input_norm = nn.LayerNorm(hidden_dim)
        
        # Transformer (как есть, но с norm_first=True)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=num_heads, 
            batch_first=True, dropout=0.3,
            norm_first=True  # Pre-LN для стабильности
        )
```

### 4.3. Мульти-масштабные окна

Вместо одного окна в 7 дней, использовать несколько параллельных:

```python
WINDOW_SIZES = [3, 7, 14]

class MultiScaleTransformer(nn.Module):
    def __init__(self, ...):
        self.transformers = nn.ModuleDict({
            f'w{w}': nn.TransformerEncoder(...) for w in WINDOW_SIZES
        })
        self.fusion = nn.Linear(hidden_dim * len(WINDOW_SIZES), hidden_dim)
    
    def forward(self, windows_dict):
        outputs = []
        for w, transformer in self.transformers.items():
            out = transformer(windows_dict[w])
            outputs.append(self.pool(out))
        return self.classifier(self.fusion(torch.cat(outputs, dim=-1)))
```

### 4.4. Онлайн-обучение и обнаружение дрейфа

Для реальной системы мониторинга:
```python
class ConceptDriftDetector:
    """Мониторинг деградации модели в продакшн"""
    def __init__(self, window_size=1000):
        self.predictions = deque(maxlen=window_size)
        self.baseline_positive_rate = None
    
    def update(self, prediction):
        self.predictions.append(prediction)
        current_rate = np.mean(self.predictions)
        if self.baseline_positive_rate:
            drift = abs(current_rate - self.baseline_positive_rate)
            if drift > 0.05:  # >5% отклонение
                self.trigger_retraining()
```

---

## 5. ПРИОРИТЕЗИРОВАННЫЙ ПЛАН УЛУЧШЕНИЙ

| Приоритет | Улучшение | Сложность | Влияние на качество |
|-----------|-----------|-----------|---------------------|
| 🔴 1 | Исправить баг в scenario evaluation | Низкая | Критично для корректности |
| 🔴 2 | Ограничить `last_day_weight` через sigmoid | Низкая | Среднее |
| 🔴 3 | Убрать хардкод метрик сценариев | Низкая | Критично для диплома |
| 🟡 4 | Добавить LayerNorm перед Transformer | Низкая | Среднее |
| 🟡 5 | Multi-layer LSTM с dropout | Низкая | Среднее |
| 🟡 6 | Learning rate warmup | Низкая | Среднее |
| 🟡 7 | Обучить реальные бейслайны | Средняя | Высокое для диплома |
| 🟢 8 | Добавить LDAP/psychometric данные | Средняя | Высокое |
| 🟢 9 | Агрегатные признаки дня | Средняя | Высокое |
| 🟢 10 | Увеличить XAI до 50+ примеров | Средняя | Среднее |
| 🔵 11 | Мульти-масштабные окна | Высокая | Высокое |
| 🔵 12 | Аугментация позитивных примеров | Средняя | Среднее |
| 🔵 13 | Cross-validation | Высокая | Высокое для доверия к результатам |

---

## 6. ИТОГОВАЯ ОЦЕНКА

**Текущая система — крепкий прототип уровня магистерской работы.** Иерархический подход BiLSTM+Transformer с микро-признаками времени — это хорошо мотивированная архитектура. Dual XAI (Attention + Occlusion) — сильный элемент для интерпретируемости.

**Основные слабости:**
1. Не используется весь потенциал CERT r5.2 (LDAP, psychometric)
2. Несколько технических багов (scenario path, unconstrained weight)
3. Бейслайны не обучены, а захардкожены
4. XAI-анализ на 3 примерах статистически не валиден
5. Нет ablation study для обоснования каждого компонента

Реализация 4-5 улучшений из приоритетов 🔴 и 🟡 существенно усилит работу как для защиты диплома, так и для практического применения.


