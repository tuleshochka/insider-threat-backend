"""Background training task management.
#
# Почему фоновый поток (background thread):
#   Обучение модели автоэнкодера + Isolation Forest занимает от 30 секунд до
#   нескольких минут. Если запускать синхронно внутри HTTP-запроса (например,
#   POST /api/train), клиент получит timeout. Фоновый поток позволяет
#   вернуть task_id немедленно и опрашивать статус через GET /api/train/status.
#
# Почему in-memory dict (а не Redis / БД):
#   На этапе разработки и отладки хранение состояния задач в оперативной
#   памяти — самое быстрое решение: не требует поднимать Redis, не создаёт
#   лишних таблиц в SQLite. Для продакшена словарь надо заменить на
#   Redis (или другую разделяемую БД), чтобы состояние переживало
#   перезапуск веб-сервера и было доступно из разных воркеров.
#
# Почему threading.Lock:
#   HTTP-сервер (FastAPI) обрабатывает запросы в нескольких потоках.
#   Одновременный доступ к _tasks из разных потоков (один пишет статус,
#   другой читает) без блокировки может привести к race condition —
#   чтение «грязных» данных или потеря обновления. Lock гарантирует
#   атомарность чтения-записи для _tasks.
#
# Почему _update вынесена в отдельную функцию и импортируется из trainer.py:
#   Trainer.run() выполняется в фоновом потоке и должен сообщать прогресс
#   (сколько эпох пройдено, текущая loss). Если бы метод update() был
#   только внутри train_task, trainer.py пришлось бы завязываться на
#   этот модуль циклически. Вынесение _update в отдельную чистую функцию
#   позволяет trainer.py делать простой import и вызывать её для
#   обновления статуса, не создавая кольцевых зависимостей.
"""
import uuid
import threading
from typing import Any

from app.models.db import SessionLocal
from app.services.trainer import Trainer

# In-memory task store (dev only; replace with Redis in production)
_tasks: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def start_training() -> str:
    """Queue training in background thread."""
    task_id = uuid.uuid4().hex[:12]
    with _lock:
        _tasks[task_id] = {"status": "running", "progress": 0.0, "message": "Starting..."}
    thread = threading.Thread(target=_run_training, args=(task_id,), daemon=True)
    thread.start()
    return task_id


def get_task_state(task_id: str) -> dict[str, Any]:
    """Return current task state."""
    with _lock:
        return _tasks.get(task_id, {"status": "not_found", "progress": 0.0, "message": "Unknown task"})


def _run_training(task_id: str):
    """Run training pipeline with progress tracking."""
    db = SessionLocal()
    try:
        _update(task_id, 0, "Initialising...")
        trainer = Trainer(db, task_id=task_id)

        result = trainer.run(
            start="2010-01-01",
            end="2010-12-31",
        )

        _update(task_id, 100, f"Done. Loss: {result['train_loss']:.6f}")
    except Exception as e:
        _update(task_id, 0, f"Failed: {e}")
        import traceback
        traceback.print_exc()
    finally:
        db.close()


def _update(task_id: str, progress: float, message: str):
    """Thread-safe update of task state. Also importable from trainer.py."""
    with _lock:
        if task_id in _tasks:
            _tasks[task_id] = {
                "status": "running" if progress < 100 else "done",
                "progress": progress,
                "message": message,
            }

