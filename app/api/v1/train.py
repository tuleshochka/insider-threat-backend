"""Training endpoints: trigger and poll."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from app.models.db import get_db
from app.models.schemas import TrainResponse, TrainStatus
from app.tasks.train_task import start_training, get_task_state

router = APIRouter()


# POST-эндпоинт для запуска обучения модели.
# Метод POST выбран, потому что запрос НЕ идемпотентен — каждый вызов
# создаёт новую фоновую задачу с уникальным task_id. GET здесь не подходит
# по двум причинам: GET не должен иметь побочных эффектов (создание задачи),
# и тело запроса у GET не является стандартизированным.
#
# Статус-код 202 (Accepted) — корректный выбор для асинхронных операций:
# он сообщает клиенту, что запрос принят, но результат ещё не готов.
# Клиент должен опрашивать статус через GET /{task_id}. Это паттерн
# «polling» — простой и надёжный способ реализации асинхронности без
# WebSocket или Server-Sent Events, оправданный здесь тем, что обучение
# модели может длиться минуты.
#
# response_model=TrainResponse гарантирует, что клиент всегда получит
# task_id и сообщение в едином формате.
@router.post("", response_model=TrainResponse, status_code=202)
def trigger_training(db: Session = Depends(get_db)):
    """Start model training as a background task."""
    task_id = start_training()
    return TrainResponse(task_id=task_id, message="Training started")


# GET-эндпоинт для опроса статуса фоновой задачи обучения.
# Метод GET — идемпотентный, не меняет состояние задачи, только читает его.
# Это позволяет безопасно повторять запрос (например, при обрыве сети).
#
# Параметр task_id — часть пути (path parameter), а не query-параметр,
# потому что он идентифицирует конкретный ресурс (задачу). По REST-канону
# идентификатор ресурса — это часть URL, а не строка запроса.
#
# response_model=TrainStatus определяет поля, которые получит клиент:
# status (running/completed/failed), progress (0-100) и message.
# FastAPI автоматически исключит поля, которых нет в модели, — это
# защищает от утечки внутреннего состояния задачи.
@router.get("/{task_id}", response_model=TrainStatus)
def training_status(task_id: str):
    """Poll training progress."""
    state = get_task_state(task_id)
    return TrainStatus(
        task_id=task_id,
        status=state["status"],
        progress=state["progress"],
        message=state.get("message", ""),
    )
