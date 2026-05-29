"""Authentication endpoints."""
from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from app.models.db import get_db
from app.models.orm import SystemUser
from app.models.schemas import Token, SystemUserOut
from app.core.security import verify_password, create_access_token
from app.api.v1.security import get_actor, Actor

router = APIRouter()


@router.post("/login", response_model=Token)
def login(
    form_data: OAuth2PasswordRequestForm = Depends(),
    db: Session = Depends(get_db),
):
    """Аутентификация пользователя и получение JWT-токена."""
    user = db.query(SystemUser).filter(SystemUser.username == form_data.username).first()
    if not user or not verify_password(form_data.password, user.hashed_password):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Неверное имя пользователя или пароль",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    if not user.is_active:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Пользователь заблокирован",
        )
        
    access_token = create_access_token(subject=user.username)
    return Token(access_token=access_token)


@router.get("/me", response_model=SystemUserOut)
def get_current_user_profile(
    actor: Actor = Depends(get_actor),
    db: Session = Depends(get_db),
):
    """Получение профиля текущего авторизованного пользователя."""
    user = db.query(SystemUser).filter(SystemUser.username == actor.name).first()
    if not user:
        raise HTTPException(status_code=404, detail="Пользователь не найден")
    return user
