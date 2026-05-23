"""Audit helpers for corporate workflow actions."""
from typing import Any

from sqlalchemy.orm import Session

from app.api.v1.security import Actor
from app.models.orm import AuditLog


def write_audit(
    db: Session,
    actor: Actor,
    action: str,
    object_type: str,
    object_id: str | int | None = None,
    result: str = "success",
    details: dict[str, Any] | None = None,
) -> AuditLog:
    entry = AuditLog(
        actor=actor.name,
        actor_role=actor.role,
        action=action,
        object_type=object_type,
        object_id=str(object_id) if object_id is not None else None,
        result=result,
        details=details,
    )
    db.add(entry)
    return entry
