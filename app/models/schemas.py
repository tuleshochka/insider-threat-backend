"""
Pydantic-схемы для валидации запросов и ответов API.

Зачем нужны отдельные схемы, а не возвращаются ORM-модели напрямую?
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Pydantic выполняет три функции: (1) автоматическая валидация типов на
входе в API — если клиент передал строку вместо числа, FastAPI вернёт
понятную ошибку 422; (2) сериализация — поля, существующие в БД, но
не нужные клиенту (хэши паролей, служебные флаги), просто не включаются
в схему; (3) документирование — по этим классам генерируется OpenAPI-схема
для Swagger UI. from_attributes=True разрешает создавать Pydantic-объект
напрямую из SQLAlchemy-модели.
"""
from datetime import datetime
from typing import Any
from pydantic import BaseModel, Field


# ── Пользователь (ответ) ──
class UserOut(BaseModel):
    id: int                           # первичный ключ в БД
    anon_id: str                      # анонимизированный идентификатор пользователя в датасете CERT
    role: str | None                  # должность (например, IT Admin, Sales Manager)
    department: str | None            # отдел — для группового анализа аномалий
    business_unit: str | None         # бизнес-юнит — более крупная единица группировки
    is_active: bool                   # флаг активности: уволенные пользователи исключаются из мониторинга
    risk_score: float = 0.0           # агрегированный уровень риска (0-1), вычисляется по истории аномалий

    model_config = {"from_attributes": True}


class UserListResponse(BaseModel):
    items: list[UserOut]   # список пользователей на текущей странице
    total: int             # общее количество (для пагинации)
    page: int              # номер текущей страницы
    size: int              # размер страницы


# ── Профиль пользователя (настройки детектора) ──
class ProfileOut(BaseModel):
    id: int                              # первичный ключ
    user_id: int                         # внешний ключ к таблице users
    anomaly_threshold: float | None      # индивидуальный порог срабатывания (если None — используется глобальный)
    trained_at: datetime | None          # дата последнего обучения модели для этого пользователя
    model_version: str | None            # версия модели (для отслеживания изменений и отката)

    model_config = {"from_attributes": True}


# ── Признак аномалии (один элемент объяснения) ──
class FeatureOut(BaseModel):
    feature_name: str     # название признака (например, "after_hours_logons")
    feature_value: float  # значение признака у данного пользователя
    shap_value: float     # вклад признака в итоговую оценку (положительный — повышает аномальность)

    model_config = {"from_attributes": True}


# ── Аномалия (зафиксированное событие-подозрение) ──
class AnomalyOut(BaseModel):
    id: int                         # первичный ключ
    user_id: int                    # к какому пользователю относится
    score: float                    # оценка аномальности от модели (0-1)
    threshold: float                # порог, при котором сработало (для понимания «насколько близко к границе»)
    status: str                     # статус расследования: new / under_review / confirmed / rejected / false_positive
    detected_at: datetime           # момент автоматического обнаружения
    reviewed_at: datetime | None = None  # момент, когда специалист ИБ проверил
    features: list[FeatureOut] = []      # SHAP-объяснение: какие признаки повлияли на решение

    model_config = {"from_attributes": True}


# ── Запрос на ревью аномалии ──
class AnomalyReview(BaseModel):
    status: str  # новое состояние: "confirmed" — подтверждена как угроза, "rejected" — ложное срабатывание
    comment: str | None = None


# ── Обучение модели ──
class TrainResponse(BaseModel):
    task_id: str    # UUID задачи (для отслеживания через TrainStatus)
    message: str    # человекочитаемое сообщение о старте


class TrainStatus(BaseModel):
    task_id: str                # UUID задачи, переданный из TrainResponse
    status: str                 # текущее состояние: running | done | failed
    progress: float             # прогресс в процентах (0-100) для отображения progress bar на фронтенде
    message: str = ""           # дополнительное сообщение (ошибка, предупреждение)


# ── Здоровье сервиса ──
class HealthOut(BaseModel):
    status: str = "ok"        # ok / degraded / down
    version: str = "1.0.0"    # семантическая версия для отслеживания совместимости


class ActorContextOut(BaseModel):
    name: str
    role: str
    permissions: list[str]


class IncidentCreate(BaseModel):
    anomaly_id: int
    title: str | None = None
    severity: str = Field(default="high", pattern="^(low|medium|high|critical)$")
    assignee: str | None = None
    summary: str | None = None


class IncidentOut(BaseModel):
    id: int
    anomaly_id: int | None
    user_id: int | None
    title: str
    severity: str
    status: str
    assignee: str | None
    summary: str | None
    exported: bool
    created_by: str
    created_at: datetime
    updated_at: datetime | None

    model_config = {"from_attributes": True}


class IncidentStatusUpdate(BaseModel):
    status: str = Field(pattern="^(open|in_progress|resolved|closed)$")


class AuditLogOut(BaseModel):
    id: int
    actor: str
    actor_role: str
    action: str
    object_type: str
    object_id: str | None
    result: str
    details: dict[str, Any] | None
    created_at: datetime

    model_config = {"from_attributes": True}


class SystemSettingOut(BaseModel):
    key: str
    value: dict[str, Any]
    description: str | None = None
    updated_by: str | None = None
    updated_at: datetime | None = None

    model_config = {"from_attributes": True}


class SystemSettingUpdate(BaseModel):
    value: dict[str, Any]
    description: str | None = None


class DatasetReleaseOut(BaseModel):
    id: str
    path: str
    recommended: bool
    purpose: str
    schema_notes: list[str]


class TrainingPlanOut(BaseModel):
    execution_target: str
    baseline_release: str
    recommended_releases: list[str]
    excluded_by_default: list[str]
    storage_format: str
    feature_grain: str
    required_artifacts: list[str]
    validation_strategy: list[str]
    releases: list[DatasetReleaseOut]


# ── Схемы для авторизации системных пользователей ──
class SystemUserCreate(BaseModel):
    username: str
    password: str
    role: str = Field(default="observer", pattern="^(admin|specialist|observer)$")


class SystemUserOut(BaseModel):
    id: int
    username: str
    role: str
    is_active: bool
    created_at: datetime

    model_config = {"from_attributes": True}


class SystemUserRoleUpdate(BaseModel):
    role: str = Field(pattern="^(admin|specialist|observer)$")


class Token(BaseModel):
    access_token: str
    token_type: str = "bearer"


class TokenData(BaseModel):
    username: str | None = None

