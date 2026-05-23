"""Lightweight role context for the prototype corporate workflow."""
from dataclasses import dataclass

from fastapi import Header, HTTPException


ROLE_PERMISSIONS: dict[str, set[str]] = {
    "security_specialist": {
        "anomalies:read",
        "anomalies:review",
        "incidents:read",
        "incidents:create",
        "training:read",
    },
    "lead": {
        "anomalies:read",
        "anomalies:review",
        "incidents:read",
        "incidents:create",
        "incidents:update",
        "audit:read",
        "settings:read",
        "settings:write",
        "training:read",
        "training:manage",
    },
    "observer": {
        "anomalies:read",
        "incidents:read",
        "audit:read",
        "settings:read",
        "training:read",
    },
}


@dataclass(frozen=True)
class Actor:
    name: str
    role: str
    permissions: set[str]


def get_actor(
    x_actor_name: str | None = Header(default="demo.specialist"),
    x_actor_role: str | None = Header(default="security_specialist"),
) -> Actor:
    """Resolve a demo actor from headers.

    The production version should replace this with SSO/LDAP integration.
    For the home test stand headers keep role scenarios easy to reproduce.
    """
    role = x_actor_role or "security_specialist"
    if role not in ROLE_PERMISSIONS:
        raise HTTPException(403, f"Unknown role: {role}")
    return Actor(
        name=x_actor_name or "demo.specialist",
        role=role,
        permissions=ROLE_PERMISSIONS[role],
    )


def require_permission(actor: Actor, permission: str) -> None:
    if permission not in actor.permissions:
        raise HTTPException(403, f"Permission required: {permission}")
