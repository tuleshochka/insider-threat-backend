"""System users management endpoints (Admin only)."""
from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.orm import Session

from app.models.db import get_db
from app.models.orm import SystemUser
from app.models.schemas import SystemUserCreate, SystemUserOut, SystemUserRoleUpdate
from app.core.security import get_password_hash
from app.api.v1.security import get_actor, require_permission, Actor
from app.services.audit import write_audit

router = APIRouter()


@router.get("", response_model=list[SystemUserOut])
def list_system_users(
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    """Получение списка всех зарегистрированных системных пользователей (требует прав администратора)."""
    require_permission(actor, "users:manage")
    return db.query(SystemUser).order_by(SystemUser.created_at.desc()).all()


@router.post("", response_model=SystemUserOut, status_code=status.HTTP_201_CREATED)
def create_system_user(
    user_in: SystemUserCreate,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    """Регистрация нового системного пользователя (требует прав администратора)."""
    require_permission(actor, "users:manage")
    
    existing = db.query(SystemUser).filter(SystemUser.username == user_in.username).first()
    if existing:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь с таким именем уже существует",
        )
        
    hashed_password = get_password_hash(user_in.password)
    user = SystemUser(
        username=user_in.username,
        hashed_password=hashed_password,
        role=user_in.role,
        is_active=True,
    )
    db.add(user)
    db.commit()
    db.refresh(user)
    
    write_audit(
        db=db,
        actor=actor,
        action="create_system_user",
        object_type="SystemUser",
        object_id=user.id,
        details={"username": user.username, "role": user.role},
    )
    db.commit()
    
    return user


@router.patch("/{user_id}/role", response_model=SystemUserOut)
def update_system_user_role(
    user_id: int,
    role_update: SystemUserRoleUpdate,
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    """Изменение роли системного пользователя (требует прав администратора)."""
    require_permission(actor, "users:manage")
    
    user = db.query(SystemUser).filter(SystemUser.id == user_id).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
        
    if user.username == actor.name:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Вы не можете изменить роль самому себе",
        )
        
    old_role = user.role
    user.role = role_update.role
    db.add(user)
    
    write_audit(
        db=db,
        actor=actor,
        action="update_system_user_role",
        object_type="SystemUser",
        object_id=user.id,
        details={"username": user.username, "old_role": old_role, "new_role": user.role},
    )
    db.commit()
    db.refresh(user)
    
    return user
