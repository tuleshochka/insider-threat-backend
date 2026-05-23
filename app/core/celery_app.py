import os
from celery import Celery
from app.config import settings

# Использование PostgreSQL в качестве брокера задач (через SQLAlchemy transport) 
# и бэкенда результатов. Это исключает необходимость в дополнительном Redis.
db_user = settings.db_user
db_password = settings.db_password
db_host = settings.db_host
db_port = settings.db_port
db_name = settings.db_name

# Для брокера Celery используется префикс sqla+, для бэкенда результатов — db+
broker_url = f"sqla+postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"
result_backend = f"db+postgresql://{db_user}:{db_password}@{db_host}:{db_port}/{db_name}"

celery_app = Celery(
    "ueba_worker",
    broker=broker_url,
    backend=result_backend
)

celery_app.conf.update(
    task_serializer="json",
    accept_content=["json"],
    result_serializer="json",
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
)

# Автоматический поиск задач в модуле 'app.tasks'
celery_app.autodiscover_tasks(["app.tasks"])

