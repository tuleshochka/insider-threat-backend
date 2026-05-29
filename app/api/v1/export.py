"""Data export endpoints for external training."""
import io
import csv
from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import StreamingResponse
from sqlalchemy.orm import Session
from app.models.db import get_db
from app.models.orm import Anomaly, AnomalyFeedback, Feature
from app.api.v1.security import Actor, get_actor, require_permission

router = APIRouter()

@router.get("/csv", response_class=StreamingResponse)
def export_anomalies_csv(
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    """
    Эндпоинт для выгрузки размеченных данных (аномалий) в формате CSV.
    Предназначен для последующего дообучения ML-модели во внешней среде
    силами специалиста по информационной безопасности.
    Включает метки (решения), проставленные аналитиками ИБ (confirmed / false_positive).
    """
    # Доступ разрешен только тем, у кого есть права на чтение аномалий (или экспорт)
    require_permission(actor, "anomalies:read")

    # Получаем все аномалии вместе с их отзывами
    anomalies = db.query(Anomaly).all()

    # Формируем CSV в памяти
    output = io.StringIO()
    writer = csv.writer(output)
    
    # Заголовки: основные поля аномалии и метка от специалиста
    writer.writerow([
        "id", "user_id", "score", "threshold", "status",
        "detected_at", "reviewed_at", "analyst_decision", "analyst_comment"
    ])

    for a in anomalies:
        feedback = db.query(AnomalyFeedback).filter(AnomalyFeedback.anomaly_id == a.id).first()
        decision = feedback.decision if feedback else "unreviewed"
        comment = feedback.comment if feedback else ""
        
        writer.writerow([
            a.id,
            a.user_id,
            a.score,
            a.threshold,
            a.status,
            a.detected_at.isoformat() if a.detected_at else "",
            a.reviewed_at.isoformat() if a.reviewed_at else "",
            decision,
            comment
        ])

    # Сбрасываем позицию указателя в начало
    output.seek(0)
    
    # Возвращаем потоковый ответ с заголовками для скачивания файла
    return StreamingResponse(
        iter([output.getvalue()]),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=labeled_anomalies_export.csv"}
    )
