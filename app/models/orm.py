"""SQLAlchemy ORM models."""
from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text, JSON
from sqlalchemy.orm import relationship
from app.models.db import Base


class User(Base):
    """
    Пользователи корпоративной сети.
    Хранит обезличенные учётные записи из датасета CERT r4.2.
    """
    __tablename__ = "users"

    id = Column(Integer, primary_key=True, index=True)                  # Внутренний суррогатный ключ
    anon_id = Column(String(50), unique=True, nullable=False, index=True)
    # unique=True — гарантирует, что один и тот же пользователь не заведён дважды.
    # В датасете CERT пользователи идентифицируются строкой (например, 'CJL0053'),
    # поэтому накладываем уникальность для целостности данных.
    # index=True — ускоряет поиск по этому полю, по которому происходит join с логами.
    role = Column(String(50))                                           # Роль в организации (например, IT-специалист, менеджер)
    department = Column(String(100))                                    # Отдел (например, инженерный, HR)
    business_unit = Column(String(50))                                  # Бизнес-единица для группировки пользователей
    is_active = Column(Boolean, default=True)                           # Флаг активности: уволенные помечаются как False
    created_at = Column(DateTime, default=datetime.utcnow)              # Дата заведения записи в системе

    profiles = relationship("Profile", back_populates="user", cascade="all, delete-orphan")
    anomalies = relationship("Anomaly", back_populates="user", cascade="all, delete-orphan")
    # cascade="all, delete-orphan" — при удалении пользователя автоматически
    # удаляются его профили и аномалии. Это согласованность данных: записи
    # без владельца не имеют смысла.


class Profile(Base):
    """
    Эталонные профили нормального поведения пользователя.
    Строятся с помощью модели изоляции (Isolation Forest) по историческим данным.
    """
    __tablename__ = "profiles"

    id = Column(Integer, primary_key=True, index=True)                  # Внутренний суррогатный ключ
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # Внешний ключ к User. ondelete="CASCADE" — удаление пользователя
    # каскадно удаляет его профиль (см. cascade на уровне ORM).
    profile_data = Column(JSON)
    # JSON — потому что профиль содержит разнородные метрики: квартили рабочих часов,
    # типичные действия, частоту логинов, средний объём переданных данных.
    # Набор метрик может меняться при добавлении новых источников (логи, почта, устройство).
    # Реляционная схема (отдельная таблица на каждую метрику) была бы негибкой:
    # при каждом расширении пришлось бы делать миграцию с ALTER TABLE.
    # JSON-поле позволяет хранить произвольную структуру без изменения схемы БД.
    anomaly_threshold = Column(Float)                                   # Порог аномальности для данного пользователя
    trained_at = Column(DateTime, default=datetime.utcnow)              # Дата последнего обучения профиля
    model_version = Column(String(20))                                  # Версия модели (для воспроизводимости и сравнения)

    user = relationship("User", back_populates="profiles")


class Anomaly(Base):
    """
    Зафиксированные аномалии в поведении пользователя.
    Каждая запись — результат сравнения сессии пользователя с его профилем.
    """
    __tablename__ = "anomalies"

    id = Column(Integer, primary_key=True, index=True)                  # Внутренний суррогатный ключ
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=False)
    # Внешний ключ к User: аномалия всегда привязана к конкретному пользователю.
    score = Column(Float, nullable=False)                               # Численная оценка аномальности (чем выше, тем подозрительнее)
    threshold = Column(Float, nullable=False)                           # Порог, при котором событие считается аномалией
    status = Column(String(20), default="pending")                      # Статус: pending (ожидает), confirmed (подтверждена), rejected (отклонена)
    # Статус описан строковым перечислением, а не булевым флагом (is_confirmed),
    # потому что жизненный цикл аномалии включает три состояния:
    #   1) pending — система обнаружила, специалист ещё не проверил;
    #   2) confirmed — после ручной проверки признана реальной угрозой;
    #   3) rejected — ложное срабатывание, не требующее реакции.
    # Двух значений (да/нет) недостаточно: pending позволяет различать
    # нерассмотренные записи и сознательно отклонённые.
    detected_at = Column(DateTime, default=datetime.utcnow)             # Момент обнаружения аномалии
    reviewed_at = Column(DateTime, nullable=True)                       # Момент проверки специалистом (null, пока не рассмотрено)

    user = relationship("User", back_populates="anomalies")
    features = relationship("Feature", back_populates="anomaly", cascade="all, delete-orphan")
    incidents = relationship("Incident", back_populates="anomaly")
    # При удалении аномалии каскадно удаляются признаки, объясняющие её.
    # Это предотвращает «висящие» записи признаков без родительской аномалии.


class Feature(Base):
    """
    Признаки, повлиявшие на решение модели.
    Хранит вклад каждого признака (SHAP-значение) для объяснимости аномалии.
    """
    __tablename__ = "features"

    id = Column(Integer, primary_key=True, index=True)                  # Внутренний суррогатный ключ
    anomaly_id = Column(Integer, ForeignKey("anomalies.id", ondelete="CASCADE"), nullable=False)
    # Внешний ключ к Anomaly. Признак существует только в контексте конкретной аномалии.
    # Один и тот же признак (например, 'login_time') может иметь разные значения
    # и SHAP-вклады для разных аномалий, поэтому связь «многие к одному»
    # с Anomaly (а не с AnomalyType). ondelete="CASCADE" гарантирует,
    # что при удалении аномалии все её признаки удалятся автоматически.
    feature_name = Column(String(100))                                  # Название признака (например, 'logon_frequency_outlier')
    feature_value = Column(Float)                                       # Фактическое числовое значение признака в данной сессии
    shap_value = Column(Float)                                          # Вклад признака в итоговую оценку аномальности (SHAP)
    # Хранение SHAP-значений необходимо для объяснимости: специалист по ИБ
    # видит не только общий балл аномалии, но и то, какие именно признаки
    # (например, «вход в нерабочее время» или «копирование на USB-накопитель»)
    # внесли наибольший вклад. Это требование принципа Explainable AI (XAI).

    anomaly = relationship("Anomaly", back_populates="features")


class Event(Base):
    """
    События активности пользователей в реальном времени.
    Служит для эмуляции потока и сбора сырых данных перед их агрегацией.
    """
    __tablename__ = "events"

    id = Column(Integer, primary_key=True, index=True)
    anon_id = Column(String(50), index=True, nullable=False)            # Анонимизированный ID пользователя
    event_type = Column(String(50), nullable=False, index=True)         # Тип события (logon, file, http, email, device)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)   # Время наступления события
    details = Column(JSON)                                              # Полезная нагрузка события (URL, имя файла и т.д.)
    processed = Column(Boolean, default=False, index=True)              # Флаг обработки (агрегировано ли событие)


class AnomalyFeedback(Base):
    """
    Обратная связь от аналитика ИБ по выявленной аномалии.
    Позволяет дообучать модель на основе подтвержденных/отклоненных инцидентов.
    """
    __tablename__ = "anomaly_feedback"

    id = Column(Integer, primary_key=True, index=True)
    anomaly_id = Column(Integer, ForeignKey("anomalies.id", ondelete="CASCADE"), nullable=False, unique=True)
    # unique=True: на одну аномалию должен быть один итоговый вердикт
    analyst_name = Column(String(100))                                  # Имя или ID аналитика, принявшего решение
    decision = Column(String(20), nullable=False)                       # Решение: 'confirmed', 'false_positive'
    comment = Column(Text, nullable=True)                               # Комментарий специалиста
    created_at = Column(DateTime, default=datetime.utcnow)

    anomaly = relationship("Anomaly", backref="feedback")


class Incident(Base):
    """
    Инцидент, созданный специалистом ИБ по итогам проверки аномалии.
    В корпоративном режиме это отдельная сущность расследования: она может
    быть экспортирована во внешнюю SIEM/SOAR или систему заявок.
    """
    __tablename__ = "incidents"

    id = Column(Integer, primary_key=True, index=True)
    anomaly_id = Column(Integer, ForeignKey("anomalies.id", ondelete="SET NULL"), nullable=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="SET NULL"), nullable=True, index=True)
    title = Column(String(200), nullable=False)
    severity = Column(String(20), default="medium", index=True)
    status = Column(String(30), default="open", index=True)
    assignee = Column(String(100), nullable=True)
    summary = Column(Text, nullable=True)
    exported = Column(Boolean, default=False)
    created_by = Column(String(100), nullable=False)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)
    updated_at = Column(DateTime, default=datetime.utcnow)

    anomaly = relationship("Anomaly", back_populates="incidents")
    user = relationship("User")


class AuditLog(Base):
    """
    Журнал действий пользователей системы.
    Нужен для корпоративного контроля: кто смотрел карточки, менял статусы,
    создавал инциденты и изменял настройки.
    """
    __tablename__ = "audit_logs"

    id = Column(Integer, primary_key=True, index=True)
    actor = Column(String(100), nullable=False, index=True)
    actor_role = Column(String(30), nullable=False, index=True)
    action = Column(String(100), nullable=False, index=True)
    object_type = Column(String(50), nullable=False, index=True)
    object_id = Column(String(80), nullable=True, index=True)
    result = Column(String(30), default="success", index=True)
    details = Column(JSON, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow, index=True)


class SystemSetting(Base):
    """
    Настройки корпоративного режима: пороги риска, эскалация и политика
    обработки срабатываний. Значения хранятся в JSON для гибкости прототипа.
    """
    __tablename__ = "system_settings"

    id = Column(Integer, primary_key=True, index=True)
    key = Column(String(100), unique=True, nullable=False, index=True)
    value = Column(JSON, nullable=False)
    description = Column(Text, nullable=True)
    updated_by = Column(String(100), nullable=True)
    updated_at = Column(DateTime, default=datetime.utcnow)
