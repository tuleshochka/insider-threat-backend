"""Role-based access control with real JWT authentication."""
from dataclasses import dataclass
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from sqlalchemy.orm import Session

from app.config import settings
from app.models.db import get_db
from app.models.orm import SystemUser

# Define OAuth2 scheme pointing to our login endpoint
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/api/v1/auth/login")

ROLE_PERMISSIONS: dict[str, set[str]] = {
    "admin": {
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
        "users:manage",
    },
    "specialist": {
        "anomalies:read",
        "anomalies:review",
        "incidents:read",
        "incidents:create",
        "incidents:update",
        "settings:read",
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
    token: str = Depends(oauth2_scheme),
    db: Session = Depends(get_db),
) -> Actor:
    """Resolve current actor using JWT authentication."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Не удалось проверить учетные данные",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token, settings.secret_key, algorithms=[settings.algorithm])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
    except JWTError:
        raise credentials_exception
    
    user = db.query(SystemUser).filter(SystemUser.username == username).first()
    if user is None or not user.is_active:
        raise credentials_exception
        
    role = user.role
    if role not in ROLE_PERMISSIONS:
        raise HTTPException(403, f"Неизвестная роль: {role}")
        
    return Actor(
        name=user.username,
        role=role,
        permissions=ROLE_PERMISSIONS[role],
    )


def require_permission(actor: Actor, permission: str) -> None:
    if permission not in actor.permissions:
        raise HTTPException(403, f"Недостаточно прав. Требуется: {permission}")
