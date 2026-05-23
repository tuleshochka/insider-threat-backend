"""
Движок и фабрика сессий базы данных. Поддерживает исключительно PostgreSQL.

Все настройки подключения передаются через переменные окружения и настройки приложения.
"""
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, declarative_base
from app.config import settings

DATABASE_URL = (
    f"postgresql+psycopg2://{settings.db_user}:{settings.db_password}"
    f"@{settings.db_host}:{settings.db_port}/{settings.db_name}"
)
engine = create_engine(DATABASE_URL, pool_pre_ping=True, pool_size=5)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base = declarative_base()


def get_db():
    """FastAPI-зависимость: yield сессии БД, автоматическое закрытие."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()
