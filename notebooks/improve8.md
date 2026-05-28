Да, стало **заметно лучше**, чем v7: структура научнее, появились `X_dev`, проверки согласованности, LayerNorm, GELU, multiple seeds, абляция. Но в текущем варианте всё ещё есть несколько **критичных методологических и технических проблем**, которые я бы обязательно исправил до финального запуска и включения в ВКР.

Ниже — честный разбор.

---

# 1. Что стало лучше

## 1.1. `X_dev` / User Behavioral Baselines — сильное улучшение

Добавление признаков:

```python
[event_count, usb_count, email_count, http_count]
```

и их преобразование в z-score отклонения — это правильный шаг для UEBA.

Теперь модель получает не только последовательность событий, но и информацию:

> «Насколько поведение этого дня отличается от типичного профиля пользователя или глобального профиля».

Это хорошо ложится на тему ВКР:

> «интеллектуальная система мониторинга и анализа поведения пользователей».

Особенно удачно, что `X_dev` добавляется на уровне дня:

```python
dev_features = x_dev.view(B * W, 4)
day_repr = torch.cat([day_repr, dev_features], dim=-1)
day_repr = self.day_proj(day_repr)
```

То есть модель не просто получает агрегаты окна, а видит отклонение **каждого дня**.

---

## 1.2. Assertions на порядок пользователей и дат — правильно

```python
assert len(X_dev) == len(X)
assert np.all(users == users_dev)
assert np.all(dates == dates_dev)
```

Это хорошая инженерная практика. Для `.npz` с несколькими массивами такие проверки нужны обязательно.

---

## 1.3. LayerNorm перед Transformer — правильно

```python
seq_repr = self.layer_norm(seq_repr)
```

Это стабилизирует масштаб после суммы:

```python
day_repr + dow_embedding + pos_encoder
```

Для Transformer это полезно.

---

## 1.4. GELU вместо ReLU — аккуратное улучшение

```python
nn.GELU()
```

Хороший выбор. GELU часто работает мягче, чем ReLU, особенно в архитектурах с Transformer.

---

## 1.5. Multiple Seeds — большой плюс для ВКР

```python
SEEDS = [42, 100, 2026]
```

И вывод:

```python
mean ± std
```

Это усиливает научную достоверность. Комиссии обычно нравится, когда результат не представлен как один случайный запуск.

---

# 2. Главная проблема: Target Leakage в обрезке всё ещё остался

Самая важная проблема — в скрипте подготовки данных.

Сейчас ты всё ещё используешь ground truth при обрезке последовательности:

```python
mal_indices = [idx for idx, e in enumerate(events) if e[3]]
```

и затем выбираешь окно так, чтобы туда попали вредоносные события.

Это по-прежнему **label-aware trimming**.

## Почему это плохо

Да, ты правильно описала Contextual Detection:

> `[Day 1..7] -> Label(Day 7)` — это корректно для UEBA.

Но это **другая проблема**.

Использование текущего дня как входа — нормально.

А вот использование знания:

> «какие именно события внутри дня являются вредоносными»

для формирования входной последовательности — уже утечка разметки.

В реальной системе мониторинга у тебя нет `mal_indices`. Система не знает заранее, какое событие вредоносное, иначе модель не нужна.

---

## Что сейчас получается

Для положительного дня модель получает не случайный или естественный фрагмент активности, а специально выбранный кусок, где вредоносное событие гарантированно присутствует.

Для тестовой выборки это особенно опасно:

```python
events_trimmed = events[start : start + MAX_SEQ_LEN]
```

где `start` вычислен с использованием `mal_indices`.

Это может завысить качество модели.

---

## Как исправить

### Вариант 1 — лучший для научной корректности

Заменить label-aware trimming на label-independent suspicious trimming.

То есть выбирать окно не по `is_mal`, а по наблюдаемым признакам риска:

- USB;
- внешние email;
- HTTP upload;
- активность вне рабочих часов;
- большое количество файловых операций.

Пример:

```python
SUSPICIOUS_TOKENS = {
    TOKEN_MAP["USB_CONNECT"],
    TOKEN_MAP["USB_DISCONNECT"],
    TOKEN_MAP["FILE_OPEN_USB"],
    TOKEN_MAP["FILE_WRITE_USB"],
    TOKEN_MAP["FILE_COPY_USB"],
    TOKEN_MAP["FILE_DELETE_USB"],
    TOKEN_MAP["EMAIL_SEND_EXT"],
    TOKEN_MAP["HTTP_UPLOAD"],
}

def suspicious_score(event):
    ts, hour, token_id, is_mal = event

    score = 0

    if token_id in SUSPICIOUS_TOKENS:
        score += 3

    if hour < 8 or hour > 18:
        score += 1

    return score


def trim_events_label_independent(events, max_len, context_ratio=0.1):
    if len(events) <= max_len:
        return events

    scores = [suspicious_score(e) for e in events]

    if max(scores) > 0:
        anchor = int(np.argmax(scores))
        context_len = int(max_len * context_ratio)

        start = anchor - context_len
        start = max(0, min(len(events) - max_len, start))

        return events[start:start + max_len]

    return events[-max_len:]
```

Тогда в скрипте:

```python
events_trimmed = trim_events_label_independent(events, MAX_SEQ_LEN)
```

И только потом строить:

```python
seq, h_seq, wh_seq, tsle_s
```

---

### Вариант 2 — если хочешь оставить текущую логику

Тогда в дипломе нужно честно назвать это не обычным препроцессингом, а:

> oracle-assisted event selection

или

> supervised trimming for training dataset construction

Но я бы не советовал. На защите это могут раскритиковать.

---

# 3. Второй критичный момент: выбор лучшей модели по test PR-AUC

В блоке multiple seeds сейчас:

```python
if test_pr_auc > best_overall_pr_auc:
    best_overall_pr_auc = test_pr_auc
    best_model_state_global = best_model_state.copy()
    best_thr_global = best_thr
```

Это методологически неправильно.

Ты выбираешь лучший seed по тестовой выборке. Это превращает test set в часть процедуры выбора модели.

## Почему это плохо

Тестовая выборка должна использоваться только один раз — для финальной оценки.

Если ты выбираешь лучший запуск по `test_pr_auc`, то метрики становятся оптимистичными.

---

## Как исправить

Выбирать лучший запуск нужно по validation PR-AUC:

```python
if best_pr_auc > best_overall_val_pr_auc:
    best_overall_val_pr_auc = best_pr_auc
    best_model_state_global = copy.deepcopy(best_model_state)
    best_thr_global = best_thr
```

То есть:

```python
best_overall_val_pr_auc = 0.0
```

и дальше:

```python
if best_pr_auc > best_overall_val_pr_auc:
    best_overall_val_pr_auc = best_pr_auc
    best_model_state_global = copy.deepcopy(best_model_state)
    best_thr_global = best_thr
```

А `test_pr_auc` использовать только для отчёта.

---

# 4. Третья критичная проблема: `state_dict().copy()` всё ещё shallow copy

У тебя осталось:

```python
best_model_state = model.state_dict().copy()
```

и:

```python
best_model_state_global = best_model_state.copy()
```

Это поверхностная копия. Тензоры внутри могут продолжать ссылаться на изменяемые параметры модели.

Нужно:

```python
import copy
best_model_state = copy.deepcopy(model.state_dict())
```

и:

```python
best_model_state_global = copy.deepcopy(best_model_state)
```

Это обязательно исправить.

---

# 5. TSLE-нормализация всё ещё считает padding

Сейчас:

```python
train_day_mask = np.isin(users, train_users)
tsle_mean = X_tsle_log[train_day_mask].mean()
tsle_std = X_tsle_log[train_day_mask].std() + 1e-8
```

Проблема: `X_tsle_log` содержит много padding-нулей. Они попадают в среднее и std.

Лучше:

```python
train_day_mask = np.isin(users, train_users)
nonpad_mask = X != 0

tsle_mask = train_day_mask[:, None] & nonpad_mask

tsle_mean = X_tsle_log[tsle_mask].mean()
tsle_std = X_tsle_log[tsle_mask].std() + 1e-8
```

Это более корректно.

---

# 6. Baselines заявлены, но фактически не обучаются

В тексте v8 написано, что есть сравнение с бейслайнами. Но в ноутбуке бейслайны по-прежнему читаются из JSON или подставляются вручную:

```python
baseline_file = 'baseline_metrics.json'
if os.path.exists(baseline_file):
    baselines = json.load(f)
else:
    baselines = [
        {'name': 'Logistic Regression', ...},
        {'name': 'LSTM-only', ...},
        {'name': 'XGBoost', ...}
    ]
```

Это проблема.

Если ты пишешь, что бейслайны обучаются на том же split, они должны реально обучаться в ноутбуке.

Сейчас это выглядит как несоответствие между заявлением и реализацией.

## Что сделать

Либо:

### Вариант A — реализовать обучение бейслайнов

Добавить реальные блоки:

- Logistic Regression;
- XGBoost;
- LSTM-only.

Либо:

### Вариант B — убрать раздел бейслайнов

И оставить только:

- multiple seeds;
- ablation study;
- сценарный анализ;
- XAI.

Для ВКР можно обойтись без внешних бейслайнов, если есть сильная абляция.

Но нельзя оставлять fallback-метрики, которые не были получены в текущем эксперименте.

---

# 7. Репрезентативные метрики сценариев всё ещё нельзя оставлять

В сценарном блоке осталось:

```python
else:
    print('Директория answers не найдена. Выводим репрезентативные метрики сценариев для диплома:')
    rep_scens = [
        ('r5.2-1 (хищение через USB)', 45, 8, 0.8491),
        ...
    ]
```

Это лучше удалить.

Для диплома это опасно. Если комиссия или руководитель спросит:

> «Откуда эти числа?»

будет сложно защищать.

Лучше:

```python
else:
    print('Директория answers не найдена. Анализ по сценариям невозможен.')
    print('Для расчета Recall по сценариям необходимо загрузить папку answers.')
```

---

# 8. Сценарный анализ исправлен частично

Ты исправила баг:

```python
scen_path = os.path.join(ANSWERS_DIR, scen)
```

Это хорошо.

Но сценарий всё ещё определяется только по пользователю:

```python
user_scenarios[u] = scen
```

Это грубовато.

Лучше определять по `(user, date)`.

Сейчас может быть ситуация:

- пользователь встречается в одном сценарии;
- но окно относится к другому дню;
- модель считает сценарий по пользователю, а не по конкретному дню атаки.

Для более строгого анализа нужно сохранить даты окон:

```python
win_dates = []
...
win_dates.append(group['date'].values[i:i+WINDOW_SIZE])
```

И в answers читать:

```python
user_day_scenarios[(user, day)] = scen
```

А потом:

```python
u = win_u[test_idx][i]
last_day = pd.to_datetime(win_dates[test_idx][i][-1]).strftime("%Y-%m-%d")
scen = user_day_scenarios.get((u, last_day), "unknown")
```

---

# 9. Абляционный анализ стал лучше, но не совсем честный

Сейчас абляция:

```python
ablation_users = train_users[:15]
```

и 3 эпохи.

Это нормально как быстрый sanity check, но для диплома слабовато.

## Основная проблема

Твой `v7 baseline` в абляции уже использует `X_dev`:

```python
self.day_proj = nn.Linear(128 + 4, 128)
...
day_repr = torch.cat([day_repr, x_dev.view(B * W, 4)], dim=-1)
```

То есть это не настоящий v7 baseline.

Если ты хочешь показать вклад `X_dev`, нужна отдельная конфигурация:

| Конфигурация | X_dev | LayerNorm | GELU | 2-layer LSTM | Sigmoid |
|---|---:|---:|---:|---:|---:|
| v7 baseline | ❌ | ❌ | ❌ | ❌ | ❌ |
| + X_dev | ✅ | ❌ | ❌ | ❌ | ❌ |
| + LayerNorm | ✅ | ✅ | ❌ | ❌ | ❌ |
| + GELU | ✅ | ✅ | ✅ | ❌ | ❌ |
| + 2-layer LSTM | ✅ | ✅ | ✅ | ✅ | ❌ |
| Full v8 | ✅ | ✅ | ✅ | ✅ | ✅ |

И в `AblationModel` нужно уметь отключать `X_dev`.

Например:

```python
if self.use_dev:
    day_repr = torch.cat([day_repr, x_dev.view(B * W, 4)], dim=-1)
    day_repr = self.day_proj(day_repr)
else:
    day_repr = self.day_proj_no_dev(day_repr)
```

---

# 10. Sigmoid weight лучше инициализировать через logit

Сейчас:

```python
self.last_day_weight = nn.Parameter(torch.tensor(0.0))
...
w = torch.sigmoid(self.last_day_weight)
```

Это означает:

```python
w = 0.5
```

Если ты хочешь начальный вес последнего дня 0.3, нужно:

```python
init_w = 0.3
self.last_day_weight = nn.Parameter(torch.logit(torch.tensor(init_w)))
```

Тогда:

```python
torch.sigmoid(self.last_day_weight) == 0.3
```

Сейчас это не ошибка, но важно понимать: модель стартует с веса последнего дня 0.5, а не 0.3.

---

# 11. `X_dev` нужно клиппировать

В скрипте:

```python
dev_z = (daily_raw - mean) / (std + 1e-8)
```

Если `std` близок к нулю, z-score может стать огромным.

Особенно для редких признаков:

- USB;
- external email;
- HTTP upload.

Например, если пользователь почти никогда не отправляет внешние письма, `std ≈ 0`, а в один день появляется 1 external email, z-score может быть гигантским.

Лучше:

```python
std = np.maximum(std, 1.0)
dev_z = (daily_raw - mean) / std
dev_z = np.clip(dev_z, -10, 10)
```

или:

```python
dev_z = np.clip(dev_z, -5, 5)
```

Для нейросети это сильно повысит стабильность.

---

# 12. Для `X_dev` лучше использовать `log1p`-счётчики

Сейчас используются сырые counts:

```python
[event_count, usb_count, email_count, http_count]
```

Но event_count может быть большим и иметь тяжёлый хвост.

Лучше:

```python
raw = np.array([
    np.log1p(event_count),
    np.log1p(usb_count),
    np.log1p(email_count),
    np.log1p(http_count)
], dtype=np.float32)
```

И уже по этим значениям считать mean/std.

Это снизит влияние экстремальных дней.

---

# 13. В checkpoint не сохраняются параметры Behavioral Baselines

Сейчас сохраняется:

```python
save_dict = {
    'model_state_dict': model.state_dict(),
    'vocab_size': VOCAB_SIZE,
    'window_size': WINDOW_SIZE,
    'best_threshold': best_thr,
    'best_pr_auc': best_overall_pr_auc,
    'tsle_mean': tsle_mean,
    'tsle_std': tsle_std,
}
```

Но для реального backend-инференса теперь нужны ещё параметры для `X_dev`.

Иначе backend не сможет правильно посчитать z-score отклонения.

Нужно сохранять хотя бы:

```python
'dev_feature_names': ['event_count', 'usb_count', 'email_ext_count', 'http_upload_count'],
'global_mean': global_mean,
'global_std': global_std,
'dev_clip': 10,
```

Но `global_mean/global_std` сейчас есть только в скрипте подготовки данных, а в ноутбук они не передаются.

Нужно добавить их в `metadata`:

```python
metadata = {
    ...
    "dev_feature_names": ["event_count", "usb_count", "email_ext_count", "http_upload_count"],
    "global_mean": global_mean.tolist(),
    "global_std": global_std.tolist(),
}
```

А в ноутбуке:

```python
DEV_GLOBAL_MEAN = np.array(metadata["global_mean"])
DEV_GLOBAL_STD = np.array(metadata["global_std"])
```

И сохранить в `.pth`.

---

# 14. Нужно добавить проверку распределения атак по split

Сейчас split случайный по пользователям. Нужно обязательно вывести:

```python
def split_stats(name, idx):
    labels = win_y[idx]
    print(
        f"{name}: windows={len(idx)}, "
        f"positives={labels.sum()}, "
        f"positive_rate={labels.mean():.6f}"
    )

split_stats("Train", train_idx)
split_stats("Val", val_idx)
split_stats("Test", test_idx)
```

Если в test мало атак, метрики будут нестабильны.

Для сценариев тоже желательно проверить:

```python
print("Positive windows in test:", win_y[test_idx].sum())
```

---

# 15. Кривые обучения сейчас фактически не работают

В блоке 6:

```python
axes[0].plot(train_losses if 'train_losses' in locals() else [0])
```

Но в multiple seeds ты не сохраняешь `train_losses`, `val_losses`, `val_pr_aucs`.

В итоге графики могут показывать `[0]`.

Лучше сохранять кривые для лучшего seed или последнего seed:

```python
history = {
    'train_loss': [],
    'val_loss': [],
    'val_pr_auc': []
}
```

И внутри эпох:

```python
history['train_loss'].append(epoch_train_loss)
history['val_loss'].append(epoch_val_loss)
history['val_pr_auc'].append(pr_auc)
```

Для лучшего запуска:

```python
best_history_global = copy.deepcopy(history)
```

---

# 16. В скрипте нужно проверять `day`, `ts`, `hour`

Сейчас:

```python
event_id, user, day, ts, hour, token = processor(row)
if user is None: continue
```

Но если дата не распарсилась, будет:

```python
day = None
ts = None
hour = None
```

и событие попадёт в:

```python
user_day_events[(user, day)]
```

Лучше:

```python
if user is None or day is None or ts is None or hour is None:
    continue
```

---

# 17. Общий вердикт

## Стало лучше?

Да, существенно.

Текущая v8 уже выглядит как более зрелая версия:

- есть поведенческие отклонения `X_dev`;
- архитектура стабильнее;
- есть multiple seeds;
- есть ablation study;
- исправлен `scen_path`;
- добавлены assertions;
- постановка Contextual Detection описана лучше.

---

## Но до финальной версии я бы обязательно исправил 6 вещей

### Критично

1. **Убрать label-aware trimming через `mal_indices`** или заменить на label-independent suspicious trimming.
2. **Не выбирать лучший seed по test PR-AUC** — выбирать только по validation.
3. **Заменить `state_dict().copy()` на `copy.deepcopy(...)`.**
4. **Убрать fake fallback-метрики сценариев и бейслайнов.**

### Очень желательно

5. **Исправить TSLE-нормализацию с исключением padding.**
6. **Сохранить параметры `X_dev` baseline в metadata/checkpoint.**

---

# 18. Короткая оценка текущей версии

| Компонент | Оценка |
|---|---:|
| Архитектура HST-UEBA v8 | Хорошо |
| Интеграция `X_dev` | Очень хорошо |
| Contextual Detection описание | Хорошо |
| Multiple Seeds | Хорошо |
| Ablation Study | Средне, нужно честнее |
| Baselines | Пока плохо, не обучаются реально |
| Scenario analysis | Лучше, но есть fake fallback |
| Preprocessing leakage | Главная проблема |
| Готовность к ВКР | После правок — высокая |

---

Если исправить перечисленные критичные моменты, особенно **trimming leakage** и **выбор модели по test**, версия v8 будет уже достаточно сильной и методологически защищённой для дипломной работы.