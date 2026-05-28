# Глубокий анализ и рекомендации по улучшению

## 1. АНАЛИЗ СКРИПТА ПОДГОТОВКИ ДАННЫХ (prepare_nlp_dataset.py)

### 1.1. Что сделано хорошо

- **Сортировка событий по timestamp** внутри дня — критически важно для sequential-моделей
- **Микро-признаки времени** (hour, is_working_hours, TSLE) — обогащают контекст
- **Разделение USB-файловых операций** от обычных — напрямую релевантно для сценариев хищения
- **Обрезка по последним событиям** (а не первым) — правильно, т.к. вредоносные действия чаще в конце дня

### 1.2. Критические проблемы

#### Проблема 1: Потеря гранулярности HTTP-событий
```python
def process_http(row):
    return row[0], row[2], day, ts, hour, "HTTP_BROWSE"
```
**Все HTTP-события** маппятся в один токен `HTTP_BROWSE`. При этом в TOKEN_MAP есть `HTTP_UPLOAD` (17) и `HTTP_DOWNLOAD` (18), но они **никогда не генерируются**. Сценарий r5.2-5 (аномальный веб-серфинг) напрямую зависит от различения типов HTTP-активности.

**Рекомендация:**
```python
def process_http(row):
    if len(row) < 5: return None, None, None, None, None, None
    dt, day, ts, hour = parse_date(row[1])
    url = row[4].strip().lower() if len(row) > 4 else ""
    # Эвристики на основе URL-паттернов или content-type
    if any(kw in url for kw in ["upload", "dropbox", "drive.google", "wetransfer"]):
        token = "HTTP_UPLOAD"
    elif any(kw in url for kw in ["download", ".exe", ".zip", ".rar"]):
        token = "HTTP_DOWNLOAD"
    else:
        token = "HTTP_BROWSE"
    return row[0], row[2], day, ts, hour, token
```

#### Проблема 2: Определение домена для email неточно
```python
has_external = any(r.strip() and "@dtaa.com" not in r for r in ...)
```
Домен `@dtaa.com` жёстко закодирован. Если формат данных CERT содержит несколько внутренних доменов или если домен записан в другом регистре — получится неверная классификация.

**Рекомендация:**
```python
INTERNAL_DOMAINS = {"@dtaa.com", "@dtaa.org"}  # все возможные внутренние домены

def is_external_recipient(recipient):
    r = recipient.strip().lower()
    return r and not any(r.endswith(d) for d in INTERNAL_DOMAINS)
```

#### Проблема 3: Метка дня — на уровне event_id, но label — на уровне дня
```python
has_malicious = any(e[3] for e in events)  # is_mal на уровне events
```
Вы маркируете весь день как вредоносный, если хотя бы одно событие вредоносное. Это корректно для задачи обнаружения, но **создаёт шум**: в днях с 200+ нормальными событиями и 1 вредоносным модель должна «найти иголку в стоге сена» при фиксированном MAX_SEQ_LEN=200.

**Рекомендация** — добавить **event-level attention mask** или дополнительный признак `is_suspicious_event`:
```python
# В каждый event добавить binary-маркер подозрительности
# для взвешенного обучения на уровне событий
suspicion_score = []
for ts, hour, token_id, is_mal in events:
    is_offhours = not (8 <= hour <= 18)
    is_usb = token_id in {4, 5, 10, 11, 12, 13}
    suspicion_score.append(1 if (is_offhours and is_usb) else 0)
```

#### Проблема 4: Нет количественных (count-based) признаков
Текущий подход — чисто последовательный (sequence of tokens). Но для UEBA крайне важны **агрегированные статистические** отклонения:

**Рекомендация** — добавить агрегаты на уровне дня:
```python
day_stats = {
    'n_events': len(events),
    'n_usb_events': sum(1 for e in events if e[2] in usb_tokens),
    'n_email_ext': sum(1 for e in events if e[2] == 15),
    'n_after_hours': sum(1 for e in events if not (8 <= e[1] <= 18)),
    'n_file_ops': sum(1 for e in events if e[2] in file_tokens),
    'unique_hours_active': len(set(e[1] for e in events)),
    'session_duration': events[-1][0] - events[0][0] if len(events) > 1 else 0,
}
```

#### Проблема 5: TSLE (Time Since Last Event) теряет первый элемент при обрезке
```python
if len(seq) > MAX_SEQ_LEN:
    tsle_s = tsle_s[-MAX_SEQ_LEN:]
```
После обрезки `tsle_s[0]` содержит разницу с предыдущим (обрезанным) событием — это корректно по значению, но **первый TSLE в обрезанной последовательности семантически отличается** от первого TSLE полной последовательности (который равен 0.0). Модель может обучить паттерн «первый TSLE = 0 → начало дня», а при обрезке этот паттерн нарушается.

**Рекомендация:**
```python
if len(seq) > MAX_SEQ_LEN:
    seq = seq[-MAX_SEQ_LEN:]
    h_seq = h_seq[-MAX_SEQ_LEN:]
    wh_seq = wh_seq[-MAX_SEQ_LEN:]
    tsle_s = tsle_s[-MAX_SEQ_LEN:]
    tsle_s[0] = 0.0  # Сбросить маркер начала подпоследовательности
```

#### Проблема 6: Нет нормализации TSLE в скрипте подготовки
TSLE записывается в сырых секундах. Разброс огромный: от 0 до десятков тысяч. В ноутбуке потом делается Z-нормализация с log1p, но это расщепление логики между скриптами — источник ошибок.

---

## 2. АНАЛИЗ НОУТБУКА ОБУЧЕНИЯ (HST-UEBA v6)

### 2.1. Что сделано хорошо

- **Строгое разделение по пользователям** (user-level split) — предотвращает утечку данных
- **Z-нормализация TSLE строго по train-пользователям**
- **Focal Loss** с α=0.85 — адекватный выбор для экстремального дисбаланса
- **Иерархическая архитектура** (LSTM → Transformer) — моделирует внутридневной и межневой контекст
- **F2-score для выбора порога** — приоритизирует Recall, что критично для безопасности
- **Pack_padded_sequence** — корректно обрабатывает padding
- **Сравнение с baseline-моделями** в таблице

### 2.2. Критические проблемы архитектуры

#### Проблема 1: Mean Pooling теряет временную специфику
```python
pooled = trans_out.mean(dim=1)  # Усредняем по всем 7 дням
```
Вы предсказываете **последний день** окна, но используете **среднее** по всем дням. Для задачи детекции угроз последний день критически важен.

**Рекомендация** — комбинированный пулинг:
```python
class AttentivePooling(nn.Module):
    def __init__(self, hidden_dim):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1)
        self.last_day_weight = nn.Parameter(torch.tensor(0.3))
    
    def forward(self, trans_out):
        # Learnable attention pooling
        attn_scores = torch.softmax(self.attention(trans_out).squeeze(-1), dim=1)
        weighted = (trans_out * attn_scores.unsqueeze(-1)).sum(dim=1)
        # Explicitly emphasize last day
        last_day = trans_out[:, -1, :]
        return (1 - self.last_day_weight) * weighted + self.last_day_weight * last_day
```

#### Проблема 2: Нет маскирования PAD в Transformer
Заголовок секции 4 говорит «с маскированием PAD», но в коде **маски для Transformer не передаются**:
```python
trans_out = self.transformer(seq_repr)  # Нет src_key_padding_mask!
```
LSTM использует pack_padded_sequence (хорошо), но на уровне Transformer'а все 7 дней обрабатываются одинаково, даже если у пользователя есть «пустые» дни.

**Рекомендация:**
```python
# В forward():
# Создаём маску: если все токены дня = PAD, день считается пустым
day_mask = (x.view(B, W, L).sum(dim=-1) == 0)  # [B, W], True = пустой день
trans_out = self.transformer(seq_repr, src_key_padding_mask=day_mask)
```

#### Проблема 3: Нет регуляризации embedding'ов
Embedding-слои (token, hour, dow) не имеют dropout'а. При малом числе вредоносных примеров это может приводить к переобучению на конкретные паттерны токенов.

**Рекомендация:**
```python
self.emb_dropout = nn.Dropout(0.1)
# В forward:
emb = self.emb_dropout(self.embedding(x))
h_emb = self.emb_dropout(self.hour_embedding(x_h))
```

#### Проблема 4: Только 15 эпох и фиксированный LR=1e-3
С учётом сложности модели и дисбаланса классов этого может быть недостаточно. Scheduler с patience=2 и factor=0.5 очень агрессивный.

**Рекомендация:**
```python
NUM_EPOCHS = 30  # Увеличить с учётом early stopping
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=1e-3)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
    optimizer, T_0=5, T_mult=2, eta_min=1e-6
)
```

#### Проблема 5: Day-of-week embedding просто суммируется
```python
seq_repr = seq_repr + self.dow_embedding(dow) + self.pos_encoder
```
Суммирование трёх компонентов (day representation + dow embedding + positional encoding) одинаковой размерности — это «загрязнение» информации. Эмбеддинги конкурируют.

**Рекомендация** — конкатенация + проекция:
```python
self.day_projection = nn.Linear(hidden_dim * 2 + hidden_dim, hidden_dim)

# В forward:
combined = torch.cat([day_repr.view(B, W, -1), 
                       self.dow_embedding(dow), 
                       self.pos_encoder.expand(B, -1, -1)], dim=-1)
seq_repr = self.day_projection(combined)
```

### 2.3. Проблемы оценки качества

#### Проблема 1: Бейслайны захардкожены
```python
baselines = [
    ('Isolation Forest', 0.0065, 0.0736, 0.0119, 0.0239),
    ...
]
```
Значения бейслайнов **вбиты вручную**, не воспроизводимы. Для дипломной работы необходимо:

**Рекомендация:**
- Включить код обучения бейслайнов в отдельный ноутбук
- Сохранять метрики в JSON/CSV
- Указать условия воспроизводимости (random seed, split)

#### Проблема 2: Метрики сценариев — fallback со «случайными» числами
```python
rep_scens = [
    ('r5.2-1 (хищение через USB)', 45, 8, 0.8491),
    ...
]
```
Если директория answers не найдена, выводятся **фиктивные числа**. Для дипломной работы это неприемлемо — необходимо либо гарантировать доступ к answers, либо явно указать «метрики недоступны».

#### Проблема 3: Recall@1%FPR может быть нестабильным
```python
idx = np.where(fpr <= target_fpr)[0][-1]
```
При малом количестве положительных примеров ROC-кривая может быть ступенчатой, и выбор ровно 1% FPR — артефактный.

**Рекомендация** — использовать интерполяцию:
```python
from scipy.interpolate import interp1d
roc_interp = interp1d(fpr, tpr, kind='linear')
recall_at_1pct = roc_interp(0.01)
```

### 2.4. Проблемы XAI-секции

#### Проблема 1: Attention ≠ объяснение
Использование attention weights как «объяснения» — это распространённая, но **научно оспариваемая** практика (Jain & Wallace, 2019). Attention weights показывают, **как модель агрегирует информацию**, но не обязательно отражают **причинно-следственные связи**.

**Рекомендация** — дополнить методами:
1. **Integrated Gradients** — gradient-based attribution:
```python
from captum.attr import IntegratedGradients

ig = IntegratedGradients(model_wrapper)
attributions = ig.attribute(inputs, target=1, n_steps=50)
```

2. **SHAP** для feature importance:
```python
import shap
explainer = shap.DeepExplainer(model, background_data)
shap_values = explainer.shap_values(test_samples)
```

3. **Occlusion analysis** — маскирование отдельных дней окна:
```python
def occlusion_importance(model, sample, window_size):
    base_prob = model.predict(sample)
    importances = []
    for day in range(window_size):
        masked = sample.copy()
        masked[day] = zero_day  # Замена на нулевой день
        masked_prob = model.predict(masked)
        importances.append(base_prob - masked_prob)
    return importances
```

#### Проблема 2: Только 3+3 примера
Анализ на 3 TP и 3 FP недостаточен для системных выводов.

**Рекомендация:**
```python
# Агрегированный анализ по всем TP и FP
all_tp_attentions = []  # Собрать все attention maps для TP
all_fp_attentions = []

# Средняя матрица attention для TP vs FP
mean_tp_attn = np.mean(all_tp_attentions, axis=0)
mean_fp_attn = np.mean(all_fp_attentions, axis=0)

# Статистический тест различий
from scipy.stats import mannwhitneyu
for day in range(WINDOW_SIZE):
    stat, p = mannwhitneyu(
        [a[:, day].mean() for a in all_tp_attentions],
        [a[:, day].mean() for a in all_fp_attentions]
    )
    print(f"Day {day}: p-value = {p:.4f}")
```

---

## 3. РЕКОМЕНДАЦИИ ПО УЛУЧШЕНИЮ СИСТЕМЫ В ЦЕЛОМ

### 3.1. Расширение набора признаков (Feature Engineering)

| Категория | Признак | Обоснование |
|---|---|---|
| **Поведенческие** | Entropy токенов за день | Разнообразие действий |
| **Поведенческие** | Соотношение USB-событий к общему | Аномально высокая USB-активность |
| **Сетевые** | N уникальных внешних доменов | Коммуникация с внешними ресурсами |
| **Временные** | Первый/последний час активности | Работа в нетипичное время |
| **Контекстные** | Роль пользователя (IT/HR/Finance) | Разный baseline для разных ролей |
| **Кросс-дневные** | Z-score текущего дня vs скользящего среднего | Отклонение от личной нормы |

```python
# Пример: добавление персонального baseline
user_daily_counts = defaultdict(list)
for (user, day), events in user_day_events.items():
    user_daily_counts[user].append(len(events))

# Z-score: насколько этот день отличается от обычного для этого пользователя
user_mean = np.mean(user_daily_counts[user])
user_std = np.std(user_daily_counts[user]) + 1e-8
day_zscore = (len(events) - user_mean) / user_std
```

### 3.2. Архитектурные улучшения модели

#### Предложение: Multi-Scale Temporal Attention
```python
class MultiScaleHST(nn.Module):
    """
    Три масштаба анализа:
    1. Intra-day: BiLSTM на уровне событий внутри дня
    2. Short-term: Transformer на окне 7 дней (текущий подход)
    3. Long-term: Дополнительное сжатое представление за 30 дней
    """
    def __init__(self, vocab_size, embed_dim=64, hidden_dim=128):
        super().__init__()
        # Уровень 1: Intra-day (без изменений)
        self.day_encoder = DayEncoder(vocab_size, embed_dim, hidden_dim)
        
        # Уровень 2: Short-term (7 дней) — Transformer
        self.short_transformer = TransformerEncoder(hidden_dim, n_heads=4, n_layers=2)
        
        # Уровень 3: Long-term (30 дней) — лёгкий GRU
        self.long_term_gru = nn.GRU(hidden_dim, hidden_dim // 2, batch_first=True)
        
        # Fusion
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim + hidden_dim // 2, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)
        )
```

#### Предложение: Contrastive Learning для эмбеддингов пользователей
```python
class ContrastiveLoss(nn.Module):
    """Дни одного пользователя — positive pairs,
       дни разных пользователей — negative pairs"""
    def __init__(self, temperature=0.07):
        super().__init__()
        self.temperature = temperature
    
    def forward(self, embeddings, user_ids):
        sim_matrix = F.cosine_similarity(
            embeddings.unsqueeze(0), embeddings.unsqueeze(1), dim=2
        )
        # Positive mask: один пользователь
        pos_mask = (user_ids.unsqueeze(0) == user_ids.unsqueeze(1)).float()
        loss = -torch.log(
            (sim_matrix * pos_mask).sum(1) / 
            (sim_matrix * (1 - torch.eye(len(user_ids)).to(device))).sum(1)
        )
        return loss.mean()
```

### 3.3. Стратегии борьбы с дисбалансом

Текущий подход (Focal Loss с α=0.85) — базовый. Рекомендуемые дополнения:

```python
# 1. Oversampling вредоносных окон при формировании батчей
from torch.utils.data import WeightedRandomSampler

weights = np.where(win_y[train_idx] == 1, 50.0, 1.0)
sampler = WeightedRandomSampler(weights, len(weights), replacement=True)
train_loader = DataLoader(train_dataset, batch_size=64, sampler=sampler)

# 2. Mixup для аугментации на уровне эмбеддингов
def mixup_data(x, y, alpha=0.2):
    lam = np.random.beta(alpha, alpha)
    batch_size = x.size(0)
    index = torch.randperm(batch_size)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam
```

### 3.4. Валидация и воспроизводимость

```python
# Добавить K-Fold Cross-Validation по пользователям
from sklearn.model_selection import StratifiedGroupKFold

# Groups = users, Stratify = has_any_malicious
sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

user_has_malicious = {}
for u in unique_users:
    user_mask = users == u
    user_has_malicious[u] = int(y[user_mask].sum() > 0)

groups = [u for u in users]
strat_labels = [user_has_malicious[u] for u in users]

for fold, (train_idx, test_idx) in enumerate(sgkf.split(X, strat_labels, groups)):
    print(f"Fold {fold}: train={len(train_idx)}, test={len(test_idx)}")
```

### 3.5. Интеграция в реальную систему мониторинга

Для перехода от исследовательского прототипа к «интеллектуальной системе мониторинга» необходимо:

```
┌─────────────────────────────────────────────────┐
│                 Real-time Pipeline               │
├─────────────────────────────────────────────────┤
│  1. Log Collector (Syslog/SIEM → Kafka)         │
│  2. Event Tokenizer (streaming)                  │
│  3. Sliding Window Buffer (per user, in Redis)   │
│  4. HST-UEBA Inference (ONNX/TorchServe)        │
│  5. Alert Generator (threshold + cooldown)       │
│  6. Dashboard (risk scores, XAI explanations)    │
└─────────────────────────────────────────────────┘
```

**Конкретные доработки для production:**
```python
# 1. Экспорт в ONNX для ускорения inference
dummy_input = (
    torch.zeros(1, 7, 200, dtype=torch.long),
    torch.zeros(1, 7, 200, dtype=torch.long),
    torch.zeros(1, 7, 200, dtype=torch.float),
    torch.zeros(1, 7, 200, dtype=torch.float),
    torch.zeros(1, 7, dtype=torch.long),
)
torch.onnx.export(model, dummy_input, "hst_ueba.onnx")

# 2. Latency budget: inference < 100ms per window
# 3. Alert cooldown: не генерировать повторный alert 
#    для того же пользователя в течение N часов
# 4. Feedback loop: SOC-аналитик подтверждает/отклоняет → 
#    данные идут на дообучение
```

---

## 4. СВОДНАЯ ТАБЛИЦА ПРИОРИТЕТОВ

| Приоритет | Улучшение | Ожидаемый эффект | Трудоёмкость |
|---|---|---|---|
| 🔴 Высокий | Исправить process_http (токенизация) | +2-5% Recall на r5.2-5 | Низкая |
| 🔴 Высокий | Добавить padding mask в Transformer | Корректность модели | Низкая |
| 🔴 Высокий | Заменить mean pooling на attentive | +1-3% F2 | Средняя |
| 🟡 Средний | Добавить count-based признаки дня | +2-4% F1 | Средняя |
| 🟡 Средний | Integrated Gradients для XAI | Научная обоснованность | Средняя |
| 🟡 Средний | Убрать захардкоженные бейслайны | Воспроизводимость | Средняя |
| 🟡 Средний | WeightedRandomSampler | Стабильность обучения | Низкая |
| 🟢 Низкий | Multi-scale temporal attention | +3-5% PR-AUC | Высокая |
| 🟢 Низкий | Contrastive pre-training | Качество эмбеддингов | Высокая |
| 🟢 Низкий | K-Fold CV по пользователям | Статистическая надёжность | Средняя |

Основная рекомендация: **сначала устраните красные проблемы** (HTTP-токенизация, padding mask, pooling) — это даст наибольший прирост при минимальных изменениях, и затем переходите к расширению XAI-блока для повышения научной ценности дипломной работы.