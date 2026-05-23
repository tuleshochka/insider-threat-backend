"""API v1 router. Aggregates all endpoint modules."""
from fastapi import APIRouter
from app.api.v1 import anomalies, corporate, dashboard, events, health, train, training_plan, users

api_router = APIRouter()

# Каждый модуль вынесен в отдельный файл, чтобы:
# 1) сохранить разделение ответственности — health, users, anomalies,
#    train и dashboard решают разные задачи;
# 2) упростить тестирование — каждый роутер можно тестировать изолированно;
# 3) избежать конфликтов имен между обработчиками разных сущностей.
#
# prefix задаёт единый префикс пути для всех эндпоинтов модуля,
# чтобы не повторять его в каждом хендлере (DRY).
# tags используется для группировки эндпоинтов в автодокументации Swagger
# (интерфейс OpenAPI), что упрощает навигацию для фронтенд-разработчика
# и специалиста по информационной безопасности при изучении API вручную.
#
# Порядок подключения не влияет на маршрутизацию — FastAPI разрешает
# конфликты путей по принципу «специфичный маршрут раньше общего».
api_router.include_router(train.router, prefix="/train", tags=["Training"])
api_router.include_router(users.router, prefix="/users", tags=["Users"])
api_router.include_router(anomalies.router, prefix="/anomalies", tags=["Anomalies"])
api_router.include_router(dashboard.router, prefix="/dashboard", tags=["Dashboard"])
api_router.include_router(health.router, prefix="/health", tags=["Health"])
api_router.include_router(events.router, prefix="/events", tags=["Events"])
api_router.include_router(corporate.router, prefix="/corporate", tags=["Corporate"])
api_router.include_router(training_plan.router, prefix="/training", tags=["Training Plan"])
