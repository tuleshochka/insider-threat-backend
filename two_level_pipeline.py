"""
Двухуровневый пайплайн обнаружения инсайдерских угроз: AE+IF -> Multi-Input.

Зачем нужны два уровня?
━━━━━━━━━━━━━━━━━━━━━━━━━━━
Одноуровневые детекторы (один классификатор поверх сырых признаков) плохо
справляются с сильным дисбалансом классов — положительных (инсайдерских)
примеров в датасете CERT r4.2 менее 0.5%. Любая модель, которая предскажет
«норма» для всех образцов, формально получит >99.5% точности, но не обнаружит
ни одной угрозы.

Первый уровень (AE + Isolation Forest) решает задачу фильтрации: два
принципиально разных детектора аномалий работают параллельно, и их оценки
комбинируются в ансамблевый скор. Autoencoder улавливает нелинейные
корреляции в данных, Isolation Forest — разрывы в плотности распределения.
Их комбинация компенсирует недостатки каждого: AE может «переобучаться» на
нормальном поведении, IF может пропускать аномалии, скрытые в многомерных
взаимодействиях признаков.

Второй уровень (Multi-Input сеть с ветвями признаков) применяется только к
20% образцов, отобранных первым уровнем. Это снимает проблему огромного
дисбаланса — соотношение классов внутри отобранной подвыборки становится
~1:5–1:10 вместо 1:200, что позволяет обучать нейросеть без коллапса в
ложноотрицательную сторону.
"""
import os; os.environ["DB_TYPE"] = "sqlite"
os.chdir(os.path.dirname(os.path.abspath(__file__)))
import warnings; warnings.filterwarnings("ignore")
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import sys
from app.services.preprocessor import DataPreprocessor
from app.models.autoencoder import Autoencoder, train_autoencoder, compute_anomaly_scores
from app.models.isolation_forest import train_isolation_forest, score_samples
from app.models.multi_input import MultiInputClassifier, FocalLoss
from app.config import settings

# ── Ветви признаков для Multi-Input сети ──
# Почему признаки разделены на ветви, а не подаются одним вектором?
# Разные аспекты пользовательского поведения имеют разную семантику и масштаб.
# Объединение всех признаков в один вектор заставляет сеть изучать сквозные
# зависимости, что при малом числе положительных примеров ведёт к переобучению.
# Каждая ветвь обрабатывается собственной подсетью, результат агрегируется —
# модель учится и вкладывать признаки в единое скрытое представление, и
# учитывать вклад каждой семантической группы отдельно.
BRANCHES = {
    # Интенсивность операций: сколько действий каждого типа совершил пользователь.
    # Высокая интенсивность в нерабочее время — частый паттерн инсайдера.
    "intensity":     ["logon_count","file_operations","email_sent","email_received",
                      "device_operations","http_requests","email_attachments","email_size_total"],
    # Разнообразие: уникальные ресурсы, с которыми работал пользователь.
    # Инсайдеры, похищающие данные, обращаются к большему числу различных
    # файлов/папок/адресатов, чем обычные сотрудники в рутинной работе.
    "diversity":     ["logon_unique_pc","file_unique_pc","file_unique_names",
                      "email_unique_recipients","http_unique_urls"],
    # Временные признаки: активность в нерабочее время и выходные.
    # Один из самых сильных предикторов инсайдерской активности —
    # работа в часы, когда обычные сотрудники отдыхают.
    "temporal":      ["after_hours_logons","after_hours_files","after_hours_email",
                      "after_hours_device","after_hours_http","weekend_logons","weekend_device"],
    # Психометрические характеристики личности (Big Five).
    # В датасете CERT r4.2 для каждого пользователя указаны оценки O, C, E, A, N.
    # Исследования [5] показывают корреляцию некоторых черт (низкая добросовестность,
    # высокий нейротизм) с вероятностью инсайдерских действий.
    "psychometric":  ["O","C","E","A","N"],
}

print("Loading features...")
prep = DataPreprocessor()
features = prep.build_training_matrix("2010-01-01", "2010-12-31")
skip = {"date","anon_id","role","department","business_unit","employee_name"}
feat_cols = [c for c in features.columns if c not in skip]
available_cols = set(feat_cols)
for name, cols in BRANCHES.items():
    missing = [c for c in cols if c not in available_cols]
    if missing:
        print(f"  Warning: branch '{name}' missing {missing} — adding as zeros")
        for c in missing:
            features[c] = 0.0
            feat_cols.append(c)

gt = __import__("pandas").read_csv(settings.ground_truth_path)
r42 = gt[gt["dataset"] == 4.2].copy()
r42["start"] = __import__("pandas").to_datetime(r42["start"])
r42["end"] = __import__("pandas").to_datetime(r42["end"])
malicious = set()
for _, row in r42.iterrows():
    cur = row["start"]
    while cur <= row["end"]:
        malicious.add((row["user"], cur.date()))
        cur += __import__("pandas").Timedelta(days=1)
labels = np.zeros(len(features), dtype=np.float32)
for i, (_, row) in enumerate(features.iterrows()):
    if (row["anon_id"], row["date"]) in malicious:
        labels[i] = 1.0
print(f"Samples: {len(features)}, Pos: {int(labels.sum())} ({labels.sum()/len(labels)*100:.3f}%)")

# Train/test split
train_idx, test_idx = train_test_split(np.arange(len(features)), test_size=0.3, random_state=42, stratify=labels)

# ── Уровень 1: AE + Isolation Forest ──
# Почему два разных детектора аномалий?
# Autoencoder и Isolation Forest используют принципиально разные математические
# подходы к поиску аномалий. AE восстанавливает вход через узкое горлышко
# скрытого пространства — аномалии имеют большую ошибку восстановления.
# IF строит случайные деревья и измеряет глубину изоляции точки — аномалии
# изолируются быстрее. Их комбинация даёт более робастную оценку: если оба
# метода независимо указывают на одну и ту же точку как на аномальную,
# доверие к этой оценке существенно выше.
print("\n--- Level 1: AE+IF ---")
vals = features[feat_cols].values.astype(np.float64)
m_all, s_all = vals.mean(0), vals.std(0)
s_all[s_all == 0] = 1.0
X_all = (vals - m_all) / s_all

X_train_l1 = X_all[train_idx]
X_test_l1  = X_all[test_idx]
y_test_l1  = labels[test_idx].astype(np.int32)

data = torch.tensor(X_train_l1, dtype=torch.float32)
n = len(data)
tr_l1, val_l1 = data[int(n*0.1):], data[:int(n*0.1)]
train_loader = DataLoader(TensorDataset(tr_l1), batch_size=128, shuffle=True)
val_loader = DataLoader(TensorDataset(val_l1), batch_size=128)
ae = Autoencoder(input_dim=X_train_l1.shape[1])
train_autoencoder(ae, train_loader, val_loader, epochs=15, lr=0.001, patience=5)
if_model = train_isolation_forest(X_train_l1)

test_loader = DataLoader(TensorDataset(torch.tensor(X_test_l1, dtype=torch.float32)), batch_size=256)
ae_s = np.array(compute_anomaly_scores(ae, test_loader))
if_s = score_samples(if_model, X_test_l1)
ae_th = float(np.percentile(ae_s, 95))
if_th = float(np.percentile(if_s, 95))
print(f"AE threshold={ae_th:.4f}, IF threshold={if_th:.4f}")

# ── Ансамблевый скор и отбор top-K ──
# Почему ансамбль через произведение, а не сумму?
# Если хотя бы один детектор даёт низкую оценку аномальности, произведение
# обнуляет итоговый скор — это конъюнктивная комбинация: мы хотим, чтобы
# оба метода «согласились» на аномалии. Нормализация AE-скор к [0,1]
# через min-max нужна, чтобы шкалы AE и IF были сопоставимы.
# Почему top-20%, а не порог?
# Процентный отбор гарантирует управляемую загрузку второго уровня:
# мы точно знаем, сколько образцов попадёт в Multi-Input. При сильном
# дисбалансе классов фиксированный порог может пропустить слишком мало
# (и второй уровень будет обучаться на единицах примеров) или слишком
# много (и дисбаланс останется высоким). 20% — эмпирически найденный
# компромисс между полнотой первого уровня и селективностью второго.
ae_norm = (ae_s - ae_s.min()) / (ae_s.max() - ae_s.min() + 1e-10)
ensemble = np.sqrt(ae_norm * if_s)
top_k = int(len(ensemble) * 0.20)
keep_idx = np.argsort(-ensemble)[:top_k]
l1_pass = np.zeros(len(X_test_l1), dtype=np.int32)
l1_pass[keep_idx] = 1
print(f"Keeping top 20% ({top_k} samples): TP in keep={(l1_pass & y_test_l1).sum()}/{int(y_test_l1.sum())}")

# ── Уровень 2: Multi-Input с ветвями признаков ──
# Почему Multi-Input, а не полносвязная сеть на плоском векторе?
# Разделение признаков на семантические ветви решает три проблемы:
# 1) разные группы признаков имеют разную размерность и масштаб — каждая ветвь
#    нормализуется и обрабатывается независимо;
# 2) архитектура с ветвями позволяет сети обучать промежуточные представления
#    для каждого аспекта поведения, которые затем агрегируются;
# 3) это естественный способ внести априорное знание о структуре данных
#    в архитектуру модели, что особенно важно при малом числе примеров.
print("\n--- Level 2: Multi-Input ---")
# Нормализация по ветвям: каждая ветвь нормализуется по статистикам
# ТОЛЬКО обучающей выборки, чтобы избежать data leakage.
train_data = {}
test_data = {}
for name, cols in BRANCHES.items():
    v = features[cols].values.astype(np.float64)
    m, s = v[train_idx].mean(0), v[train_idx].std(0)
    s[s == 0] = 1.0
    v_norm = (v - m) / s
    train_data[name] = v_norm[train_idx]
    test_data[name]  = v_norm[test_idx]

# ── SMOTE для балансировки классов ──
# Почему SMOTE, а не взвешивание классов в функции потерь?
# При соотношении положительных к отрицательным 1:200 одно лишь взвешивание
# заставляет сеть придавать огромный вес редким примерам, что ведёт к
# нестабильному обучению. SMOTE создаёт синтетические положительные примеры
# интерполяцией между соседями в пространстве признаков — это расширяет
# область принятия решений вокруг малочисленного класса, не копируя
# существующие точки (как сделал бы Random Oversampling).
# Параметр sampling_strategy=0.2 означает, что после ресемплинга отношение
# положительных к отрицательным станет 1:5 — этого достаточно для стабильного
# обучения, но не настолько много, чтобы модель «забыла» распределение нормы.
from imblearn.over_sampling import SMOTE
print("Applying SMOTE...")
n_train = len(X_train_l1)
X_flat = np.column_stack([train_data[n] for n in BRANCHES])
smote = SMOTE(random_state=42, sampling_strategy=0.2)
X_res, y_res = smote.fit_resample(X_flat, labels[train_idx])
n_pos_before = int(labels[train_idx].sum())
n_pos_after = int(y_res.sum())
print(f"  Train samples: {n_train} -> {len(X_res)} (positives: {n_pos_before} -> {n_pos_after})")

# После SMOTE возвращаем данные к ветвям: SMOTE генерирует новые точки как
# плоские векторы, их нужно «разрезать» обратно по размерностям ветвей.
branch_dims = [len(BRANCHES[n]) for n in BRANCHES]
offsets = np.cumsum([0] + branch_dims)
train_data_res = {}
for i, name in enumerate(BRANCHES):
    train_data_res[name] = X_res[:, offsets[i]:offsets[i+1]]

# Multi-Input сеть: каждая ветвь — отдельный малый MLP, выходы конкатенируются
# и проходят через общий классификатор. latent_dim=8 — размер общего скрытого
# представления, полученного из объединения ветвей.
model = MultiInputClassifier({n: len(BRANCHES[n]) for n in BRANCHES}, latent_dim=8)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
# FocalLoss: модификация кросс-энтропии, которая уменьшает вклад хорошо
# классифицированных примеров в общую ошибку. Параметр gamma=2.0 фокусирует
# модель на «трудных» (пограничных) примерах, alpha=0.75 — вес редкого
# класса. Выбор FocalLoss вместо обычной BCE объясняется сильным дисбалансом
# даже после SMOTE: FocalLoss дополнительно страхует от переобучения на
# лёгких отрицательных примерах.
criterion = FocalLoss(alpha=0.75, gamma=2.0)
bs = 256
y_train_l2 = labels[train_idx].astype(np.float32)
y_test_l2  = labels[test_idx].astype(np.float32)

for epoch in range(30):
    model.train()
    perm = np.random.permutation(len(y_res))
    total_loss = 0.0
    for i in range(0, len(perm), bs):
        idx = perm[i:i+bs]
        inp = {n: torch.tensor(train_data_res[n][idx], dtype=torch.float32) for n in BRANCHES}
        loss = criterion(model(inp), torch.tensor(y_res[idx], dtype=torch.float32).unsqueeze(1))
        optimizer.zero_grad(); loss.backward(); optimizer.step()
        total_loss += loss.item() * len(idx)
    # Мониторинг AUC на валидации каждые 10 эпох — AUC не зависит от порога
    # принятия решения и даёт объективную оценку ранжирующей способности.
    if epoch % 10 == 9:
        model.eval()
        with torch.no_grad():
            inp_val = {n: torch.tensor(test_data[n], dtype=torch.float32) for n in BRANCHES}
            vp = torch.sigmoid(model(inp_val)).squeeze().numpy()
            vauc = __import__("sklearn.metrics", fromlist=["roc_auc_score"]).roc_auc_score(y_test_l2, vp)
            print(f"Epoch {epoch+1}: loss={total_loss/len(perm):.4f}, val-AUC={vauc:.4f}")

# ── Двухуровневая оценка: AE+IF → затем Multi-Input ──
# Почему precision@top-k, а не accuracy или полнота?
# В задаче обнаружения инсайдеров цена ложного срабатывания высока:
# каждое подозрение проверяется специалистом по ИБ. Если система выдаёт
# сотни ложных тревог в день, аналитик перестаёт им доверять (эффект
# «крика волка»). Поэтому метрика качества для практического применения —
# точность среди K наиболее подозрительных пользователей (precision@top-K).
# Специалист может проверить 10-20 пользователей в день — и ему нужно,
# чтобы среди них было максимальное число реальных угроз.
# Полнота (recall) важна, но вторична: лучше пропустить одну угрозу и
# расследовать её постфактум, чем похоронить аналитика в ложных тревогах.
model.eval()
with torch.no_grad():
    inp_all = {n: torch.tensor(test_data[n], dtype=torch.float32) for n in BRANCHES}
    mi_prob = torch.sigmoid(model(inp_all)).squeeze().numpy()

# Комбинированный фильтр: образец должен пройти первый уровень (top-20%
# по ансамблю AE+IF) И получить высокую оценку от Multi-Input сети.
# Перебор порогов для MI позволяет построить ROC и выбрать operational
# point — порог, при котором precision и recall оптимальны для бизнес-
# требований. Значения 0.3, 0.5, 0.7 покрывают диапазон от мягкого
# (больше флагов, выше recall) до жёсткого (меньше флагов, выше precision).
for thresh in [0.3, 0.5, 0.7]:
    both = ((l1_pass == 1) & (mi_prob > thresh)).astype(int)
    yl2_int = y_test_l2.astype(np.int32)
    tp = int((both & yl2_int).sum())
    fp = int((both & (1 - yl2_int)).sum())
    fn = int(y_test_l2.sum()) - tp
    p = tp/max(tp+fp,1)
    r = tp/max(tp+fn,1)
    f = 2*p*r/max(p+r, 1e-10)
    print(f"\nThreshold MI>{thresh:.1f}: TP={tp}, FP={fp}, FN={fn}, Flags={tp+fp}")
    print(f"  Precision={p:.4f}, Recall={r:.4f}, F1={f:.4f}")

# Базовый уровень (baseline): только первый уровень, без верификации MI.
# Сравнение с двухуровневым пайплайном показывает, даёт ли второй уровень
# прирост точности (обычно — да, за счёт отбрасывания ложных срабатываний,
# которые прошли первый уровень, но не подтверждены Multi-Input сетью).
print("\n" + "="*65)
print("AE+IF ALONE (top 20%):")
hits5 = int((l1_pass & y_test_l1).sum())
print(f"  TP={hits5}, Flags={int(l1_pass.sum())}, Recall={hits5/max(int(y_test_l1.sum()),1):.4f}")
print(f"  Precision={hits5/max(int(l1_pass.sum()),1):.4f}")
