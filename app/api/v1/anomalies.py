"""Anomaly endpoints: list, detail, review."""
from datetime import datetime
from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session
from app.api.v1.security import Actor, get_actor, require_permission
from app.models.db import get_db
from app.models.orm import Anomaly, AnomalyFeedback, Feature
from app.models.schemas import AnomalyOut, AnomalyReview, FeatureOut
from app.services.audit import write_audit

router = APIRouter()


# GET-эндпоинт для получения списка аномалий с опциональной фильтрацией.
# Метод GET — идемпотентное чтение коллекции.
#
# Параметры status и user_id — query-параметры (не path), потому что
# они опциональны и служат для фильтрации коллекции, а не для
# идентификации конкретного ресурса. По REST-канону фильтры — это
# query string, а не часть пути.
#
# Лимит 100 записей — защита от случайного запроса без фильтров,
# который мог бы вернуть миллионы строк и вызвать OOM на сервере.
# Клиент, которому нужны все записи, должен добавить пагинацию
# (может быть реализована в будущем).
#
# Сортировка по detected_at DESC гарантирует, что самые свежие
# (и, вероятно, наиболее актуальные) аномалии отображаются первыми.
#
# Для каждой аномалии подгружаются связанные признаки (features)
# со значениями SHAP — это данные, необходимые фронтенду для
# отрисовки объяснения причины аномалии (waterfall-диаграмма).
@router.get("", response_model=list[AnomalyOut])
def list_anomalies(
    status: str | None = None,
    user_id: int | None = None,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    """List anomalies, optionally filtered by status or user."""
    require_permission(actor, "anomalies:read")
    q = db.query(Anomaly)
    if status:
        q = q.filter(Anomaly.status == status)
    if user_id:
        q = q.filter(Anomaly.user_id == user_id)
    anomalies = q.order_by(Anomaly.detected_at.desc()).limit(100).all()

    result = []
    for a in anomalies:
        features = [
            FeatureOut(feature_name=f.feature_name, feature_value=f.feature_value, shap_value=f.shap_value)
            for f in a.features
        ]
        result.append(_anomaly_to_out(a, features))
    return result


# GET-эндпоинт для получения детальной информации об одной аномалии.
# Метод GET — чтение конкретного ресурса.
# ID аномалии — path-параметр, так как это идентификатор ресурса.
#
# 404 возвращается при отсутствии аномалии — клиент (веб-интерфейс
# специалиста по ИБ) должен понимать, что ссылка на аномалию устарела
# или недействительна.
#
# Вместе с аномалией возвращаются признаки (features) с SHAP-значениями.
# Это критически важно для объяснимости — специалист должен видеть,
# какие именно действия пользователя привели к аномалии.
@router.get("/{anomaly_id}", response_model=AnomalyOut)
def get_anomaly(
    anomaly_id: int,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    """Anomaly detail with SHAP features."""
    require_permission(actor, "anomalies:read")
    a = db.query(Anomaly).filter(Anomaly.id == anomaly_id).first()
    if not a:
        raise HTTPException(404, "Anomaly not found")
    features = [
        FeatureOut(feature_name=f.feature_name, feature_value=f.feature_value, shap_value=f.shap_value)
        for f in a.features
    ]
    return _anomaly_to_out(a, features)


# PATCH-эндпоинт для ревью аномалии: подтвердить или отклонить.
# Метод PATCH выбран, потому что изменяется ТОЛЬКО поле status
# (частичное обновление ресурса). PUT заменил бы всю запись целиком,
# что избыточно. Статус-код 200 (не 204) — возвращаем обновлённый
# ресурс, чтобы клиент мог сразу отобразить изменения без
# дополнительного GET-запроса.
#
# Валидация status на стороне API (строка 52) — защита от передачи
# некорректных значений на уровне приложения, ДО записи в БД.
# 422 Unprocessable Entity — стандартный HTTP-статус для ошибок
# валидации данных. Альтернатива — довериться CHECK-констрейнту БД,
# но тогда ошибка будет 500 (Internal Server Error), что хуже для
# диагностики.
#
# reviewed_at проставляется сервером (datetime.utcnow()), а не берётся
# из тела запроса, — это системное поле, клиент не должен его
# подделывать. Это принцип доверенного сервера: временные метки
# всегда устанавливаются на бэкенде.
@router.patch("/{anomaly_id}", response_model=AnomalyOut)
def review_anomaly(
    anomaly_id: int,
    body: AnomalyReview,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    """Confirm or reject an anomaly."""
    require_permission(actor, "anomalies:review")
    if body.status not in ("confirmed", "rejected"):
        raise HTTPException(422, "Status must be 'confirmed' or 'rejected'")
    a = db.query(Anomaly).filter(Anomaly.id == anomaly_id).first()
    if not a:
        raise HTTPException(404, "Anomaly not found")
    a.status = body.status
    a.reviewed_at = datetime.utcnow()
    feedback = db.query(AnomalyFeedback).filter(AnomalyFeedback.anomaly_id == a.id).first()
    decision = "confirmed" if body.status == "confirmed" else "false_positive"
    if feedback:
        feedback.analyst_name = actor.name
        feedback.decision = decision
        feedback.comment = body.comment
        feedback.created_at = datetime.utcnow()
    else:
        db.add(
            AnomalyFeedback(
                anomaly_id=a.id,
                analyst_name=actor.name,
                decision=decision,
                comment=body.comment,
            )
        )
    write_audit(
        db,
        actor,
        "anomaly.review",
        "anomaly",
        a.id,
        details={"status": body.status, "comment_present": bool(body.comment)},
    )
    db.commit()
    features = [
        FeatureOut(feature_name=f.feature_name, feature_value=f.feature_value, shap_value=f.shap_value)
        for f in a.features
    ]
    return _anomaly_to_out(a, features)


# Вспомогательная функция сборки ответа AnomalyOut.
# Вынесена в отдельную функцию, чтобы не дублировать код между
# list_anomalies, get_anomaly и review_anomaly (DRY).
# Принимает готовый список FeatureOut, а не запрашивает его внутри, —
# это даёт вызывающему коду контроль над стратегией загрузки
# связанных сущностей (eager/lazy loading).
def _anomaly_to_out(a: Anomaly, features: list) -> AnomalyOut:
    """Build AnomalyOut response."""
    return AnomalyOut(
        id=a.id,
        user_id=a.user_id,
        score=a.score,
        threshold=a.threshold,
        status=a.status,
        detected_at=a.detected_at,
        reviewed_at=a.reviewed_at,
        features=features,
    )
