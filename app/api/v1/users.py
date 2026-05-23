"""User endpoints: list, detail, profile."""
from fastapi import APIRouter, Depends, HTTPException, Query
from sqlalchemy.orm import Session
from sqlalchemy import func

from app.models.db import get_db
from app.models.orm import User, Profile, Anomaly
from app.models.schemas import UserOut, UserListResponse, ProfileOut

router = APIRouter()


# GET-эндпоинт для получения списка пользователей с сортировкой по уровню
# риска (убывание). Метод GET — идемпотентный, не изменяет состояние,
# что соответствует принципам REST (чтение коллекции).
#
# Пагинация через page/size:
# - API может вернуть тысячи пользователей; без пагинации один запрос
#   создаст чрезмерную нагрузку на БД, сеть и браузер.
# - page (>=1) и size (1-100) с валидацией через ge=1, le=100 защищают
#   от некорректных значений: page=0 или size=1000 привели бы к ошибкам
#   БД или timeout'ам на стороне клиента.
# - Query(...) с default=1/20 делает параметры необязательными для
#   клиента, но задаёт разумные значения по умолчанию.
#
# Сортировка по risk_score (убывание) на стороне Python, а не SQL:
# risk_score вычисляется динамически через _compute_current_risk,
# который не является колонкой таблицы. Сортировка в Python — осознанный
# компромисс для точности ранжирования: специалисту по ИБ важно видеть
# самых рискованных пользователей первыми.
#
# response_model=UserListResponse оборачивает массив в конверт с total,
# чтобы клиент знал общее количество записей и мог отобразить пагинатор.
@router.get("", response_model=UserListResponse)
def list_users(
    page: int = Query(1, ge=1),
    size: int = Query(20, ge=1, le=100),
    db: Session = Depends(get_db),
):
    """Paginated list of users with current risk score."""
    total = db.query(func.count(User.id)).scalar()
    offset = (page - 1) * size
    users = db.query(User).offset(offset).limit(size).all()

    items = []
    for u in users:
        score = _compute_current_risk(db, u.id)
        items.append(UserOut(
            id=u.id,
            anon_id=u.anon_id,
            role=u.role,
            department=u.department,
            business_unit=u.business_unit,
            is_active=u.is_active,
            risk_score=score,
        ))

    items.sort(key=lambda x: x.risk_score, reverse=True)
    return UserListResponse(items=items, total=total, page=page, size=size)


# GET-эндпоинт для получения одного пользователя по ID.
# Метод GET — чтение конкретного ресурса (REST: /users/{user_id}).
# ID передаётся в пути, а не в query, потому что он идентифицирует
# уникальный ресурс, а не фильтрует коллекцию.
#
# HTTPException(404) возвращается, если пользователь не найден.
# Это корректный HTTP-статус для отсутствующего ресурса. Альтернатива —
# возвращать null с 200, но 404 точнее семантически: клиент (фронтенд)
# может обработать 404 в catch-блоке и показать «Пользователь не найден»,
# не проверяя тело ответа на null.
@router.get("/{user_id}", response_model=UserOut)
def get_user(user_id: int, db: Session = Depends(get_db)):
    """Single user detail."""
    u = db.query(User).filter(User.id == user_id).first()
    if not u:
        raise HTTPException(404, "User not found")
    score = _compute_current_risk(db, u.id)
    return UserOut(
        id=u.id,
        anon_id=u.anon_id,
        role=u.role,
        department=u.department,
        business_unit=u.business_unit,
        is_active=u.is_active,
        risk_score=score,
    )


# GET-эндпоинт для получения профиля пользователя (результат обучения
# модели для конкретного пользователя). Выделен в отдельный подресурс
# (/users/{user_id}/profile), потому что профиль — самостоятельная
# сущность, создаваемая в процессе обучения модели.
# Вложенный URL (/users/{id}/profile) отражает иерархию: профиль
# принадлежит пользователю.
#
# Выбирается последняя запись (ORDER BY trained_at DESC LIMIT 1) —
# у пользователя может быть несколько профилей после переобучений,
# клиенту нужен самый свежий.
#
# 404 возвращается, если профиля пока нет (обучение не запускалось).
@router.get("/{user_id}/profile", response_model=ProfileOut)
def get_profile(user_id: int, db: Session = Depends(get_db)):
    """Latest profile for a user."""
    p = db.query(Profile).filter(Profile.user_id == user_id).order_by(Profile.trained_at.desc()).first()
    if not p:
        raise HTTPException(404, "Profile not found")
    return ProfileOut(
        id=p.id,
        user_id=p.user_id,
        anomaly_threshold=p.anomaly_threshold,
        trained_at=p.trained_at,
        model_version=p.model_version,
    )


# Вспомогательная функция (не эндпоинт), вычисляющая текущий уровень
# риска пользователя как оценку (score) последней непросмотренной
# аномалии со статусом "pending".
#
# Выбор pending-аномалии, а не confirmed: риск отражает потенциальную
# угрозу, которую ещё не оценил специалист. После подтверждения или
# отклонения аномалия перестаёт влиять на риск.
#
# Если аномалий нет — возвращается 0.0 (отсутствие риска).
# Альтернатива — вернуть None, но числовой 0 удобнее для сортировки
# и отображения на дашборде (прогресс-бар от 0).
def _compute_current_risk(db: Session, user_id: int) -> float:
    """Return the score of the latest pending anomaly, or 0."""
    latest = (
        db.query(Anomaly)
        .filter(Anomaly.user_id == user_id, Anomaly.status == "pending")
        .order_by(Anomaly.detected_at.desc())
        .first()
    )
    return latest.score if latest else 0.0
