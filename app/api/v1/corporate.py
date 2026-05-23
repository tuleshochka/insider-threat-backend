"""Corporate workflow endpoints: actor context, incidents, audit and settings."""
from datetime import datetime

from fastapi import APIRouter, Depends, HTTPException
from sqlalchemy.orm import Session

from app.api.v1.security import Actor, ROLE_PERMISSIONS, get_actor, require_permission
from app.models.db import get_db
from app.models.orm import Anomaly, AuditLog, Incident, SystemSetting
from app.models.schemas import (
    ActorContextOut,
    AuditLogOut,
    IncidentCreate,
    IncidentOut,
    IncidentStatusUpdate,
    SystemSettingOut,
    SystemSettingUpdate,
)
from app.services.audit import write_audit

router = APIRouter()

DEFAULT_SETTINGS = {
    "risk_policy": {
        "value": {
            "queue_threshold": 0.65,
            "high_threshold": 0.82,
            "critical_threshold": 0.92,
            "auto_escalate_critical": True,
        },
        "description": "Пороги попадания событий в очередь и правила эскалации.",
    },
    "source_policy": {
        "value": {
            "required_sources": ["logon", "file", "email", "device", "http"],
            "optional_sources": ["decoy_file", "users", "psychometric"],
        },
        "description": "Набор источников событий для корпоративного профиля поведения.",
    },
}


@router.get("/me", response_model=ActorContextOut)
def me(actor: Actor = Depends(get_actor)):
    return ActorContextOut(
        name=actor.name,
        role=actor.role,
        permissions=sorted(actor.permissions),
    )


@router.get("/roles")
def roles():
    return {role: sorted(perms) for role, perms in ROLE_PERMISSIONS.items()}


@router.get("/incidents", response_model=list[IncidentOut])
def list_incidents(
    status: str | None = None,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    require_permission(actor, "incidents:read")
    q = db.query(Incident)
    if status:
        q = q.filter(Incident.status == status)
    return q.order_by(Incident.created_at.desc()).limit(100).all()


@router.post("/incidents", response_model=IncidentOut)
def create_incident(
    body: IncidentCreate,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    require_permission(actor, "incidents:create")
    anomaly = db.query(Anomaly).filter(Anomaly.id == body.anomaly_id).first()
    if not anomaly:
        raise HTTPException(404, "Anomaly not found")

    title = body.title or f"Проверка подозрительной активности #{anomaly.id}"
    incident = Incident(
        anomaly_id=anomaly.id,
        user_id=anomaly.user_id,
        title=title,
        severity=body.severity,
        assignee=body.assignee,
        summary=body.summary,
        created_by=actor.name,
    )
    db.add(incident)
    write_audit(
        db,
        actor,
        "incident.create",
        "incident",
        details={"anomaly_id": anomaly.id, "severity": body.severity},
    )
    db.commit()
    db.refresh(incident)
    return incident


@router.patch("/incidents/{incident_id}", response_model=IncidentOut)
def update_incident_status(
    incident_id: int,
    body: IncidentStatusUpdate,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    require_permission(actor, "incidents:update")
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(404, "Incident not found")
    incident.status = body.status
    incident.updated_at = datetime.utcnow()
    write_audit(db, actor, "incident.status_update", "incident", incident.id, details={"status": body.status})
    db.commit()
    db.refresh(incident)
    return incident


@router.post("/incidents/{incident_id}/export", response_model=IncidentOut)
def export_incident(
    incident_id: int,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    require_permission(actor, "incidents:update")
    incident = db.query(Incident).filter(Incident.id == incident_id).first()
    if not incident:
        raise HTTPException(404, "Incident not found")
    incident.exported = True
    incident.updated_at = datetime.utcnow()
    write_audit(db, actor, "incident.export", "incident", incident.id, details={"target": "json_stub"})
    db.commit()
    db.refresh(incident)
    return incident


@router.get("/audit", response_model=list[AuditLogOut])
def list_audit(
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    require_permission(actor, "audit:read")
    return db.query(AuditLog).order_by(AuditLog.created_at.desc()).limit(200).all()


@router.get("/settings", response_model=list[SystemSettingOut])
def list_settings(
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    require_permission(actor, "settings:read")
    _ensure_settings(db)
    return db.query(SystemSetting).order_by(SystemSetting.key.asc()).all()


@router.patch("/settings/{key}", response_model=SystemSettingOut)
def update_setting(
    key: str,
    body: SystemSettingUpdate,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    require_permission(actor, "settings:write")
    _ensure_settings(db)
    setting = db.query(SystemSetting).filter(SystemSetting.key == key).first()
    if not setting:
        raise HTTPException(404, "Setting not found")
    setting.value = body.value
    if body.description is not None:
        setting.description = body.description
    setting.updated_by = actor.name
    setting.updated_at = datetime.utcnow()
    write_audit(db, actor, "settings.update", "system_setting", key, details={"value": body.value})
    db.commit()
    db.refresh(setting)
    return setting


def _ensure_settings(db: Session) -> None:
    existing = {row.key for row in db.query(SystemSetting.key).all()}
    changed = False
    for key, payload in DEFAULT_SETTINGS.items():
        if key in existing:
            continue
        db.add(SystemSetting(key=key, value=payload["value"], description=payload["description"]))
        changed = True
    if changed:
        db.commit()
