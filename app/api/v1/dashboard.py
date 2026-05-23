"""Dashboard aggregation endpoint."""
from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import func
from pydantic import BaseModel

from app.models.db import get_db
from app.models.orm import Anomaly, Incident, User

router = APIRouter()


# DashboardOut — Pydantic-модель, объявленная локально, а не в schemas.py,
# потому что она используется ТОЛЬКО в этом эндпоинте. Вынос в schemas.py
# был бы избыточным: при изменении дашборда пришлось бы править два файла.
# Однако если в будущем появится второй эндпоинт, возвращающий такую же
# структуру, модель стоит перенести в schemas.py (DRY).
class DashboardOut(BaseModel):
    total_users: int
    total_anomalies: int
    pending_anomalies: int
    confirmed_anomalies: int
    rejected_anomalies: int
    open_incidents: int
    critical_incidents: int
    max_score: float


# GET-эндпоинт для агрегированных статистик главной страницы дашборда.
# Метод GET — идемпотентное чтение, подходит для аналитической панели,
# которую фронтенд будет опрашивать при загрузке страницы.
#
# Все запросы выполняются через func.count и func.max — агрегатные
# функции SQL, которые считаются на стороне БД, а не в Python.
# Это эффективнее, чем загрузить все записи в память и считать их
# средствами Python: SQL-агрегаты работают за O(log n) или O(1)
# с индексами, а не за O(n) полного сканирования.
#
# Поле max_score — float(max_score) — явное приведение типа на случай,
# если func.max вернёт Decimal (зависит от диалекта SQLite/PostgreSQL).
# Без приведения Pydantic может выбросить ошибку валидации.
#
# total_users=0 — временное значение-заглушка. Заполнение будет
# реализовано после завершения модуля импорта данных пользователей
# из LDAP/CSV. Заглушка позволяет фронтенду уже сейчас отображать
# карточку «Всего пользователей», не дожидаясь финальной реализации
# импорта.
@router.get("", response_model=DashboardOut)
def dashboard(db: Session = Depends(get_db)):
    """Aggregated statistics for the main dashboard."""
    total_anomalies = db.query(func.count(Anomaly.id)).scalar()
    pending = db.query(func.count(Anomaly.id)).filter(Anomaly.status == "pending").scalar()
    confirmed = db.query(func.count(Anomaly.id)).filter(Anomaly.status == "confirmed").scalar()
    rejected = db.query(func.count(Anomaly.id)).filter(Anomaly.status == "rejected").scalar()
    total_users = db.query(func.count(User.id)).scalar()
    open_incidents = db.query(func.count(Incident.id)).filter(Incident.status.in_(["open", "in_progress"])).scalar()
    critical_incidents = db.query(func.count(Incident.id)).filter(Incident.severity == "critical").scalar()
    max_score = db.query(func.max(Anomaly.score)).scalar() or 0.0

    return DashboardOut(
        total_users=total_users,
        total_anomalies=total_anomalies,
        pending_anomalies=pending,
        confirmed_anomalies=confirmed,
        rejected_anomalies=rejected,
        open_incidents=open_incidents,
        critical_incidents=critical_incidents,
        max_score=float(max_score),
    )
