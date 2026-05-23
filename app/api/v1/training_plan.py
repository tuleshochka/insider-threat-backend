"""Training plan endpoints for Colab and multi-release CERT experiments."""
from fastapi import APIRouter, Depends

from app.api.v1.security import Actor, get_actor, require_permission
from app.models.schemas import DatasetReleaseOut, TrainingPlanOut

router = APIRouter()


@router.get("/plan", response_model=TrainingPlanOut)
def get_training_plan(actor: Actor = Depends(get_actor)):
    require_permission(actor, "training:read")
    base = "D:/Политех/Мага/Дипломы/М/12841247"
    releases = [
        DatasetReleaseOut(
            id="r4.1",
            path=f"{base}/r4.1",
            recommended=True,
            purpose="Расширение baseline и проверка близкого релиза.",
            schema_notes=["email содержит cc/bcc", "есть file/http/content"],
        ),
        DatasetReleaseOut(
            id="r4.2",
            path=f"{base}/r4.2",
            recommended=True,
            purpose="Baseline для сопоставимости с текущими метриками.",
            schema_notes=["dense needles", "есть users.csv", "основной контрольный релиз"],
        ),
        DatasetReleaseOut(
            id="r5.1",
            path=f"{base}/r5.1",
            recommended=True,
            purpose="Расширение сценариев и признаков устройств.",
            schema_notes=["добавлен decoy_file.csv", "file содержит операции с removable media"],
        ),
        DatasetReleaseOut(
            id="r5.2",
            path=f"{base}/r5.2",
            recommended=True,
            purpose="Тест переноса для r5.1.",
            schema_notes=["добавлен decoy_file.csv", "email содержит activity"],
        ),
        DatasetReleaseOut(
            id="r6.1",
            path=f"{base}/r6.1",
            recommended=True,
            purpose="Обучение на более крупной организации.",
            schema_notes=["http содержит activity", "file содержит copy/delete/open/write"],
        ),
        DatasetReleaseOut(
            id="r6.2",
            path=f"{base}/r6.2",
            recommended=True,
            purpose="Тест переноса для r6.1.",
            schema_notes=["самый крупный HTTP-журнал", "обязательна потоковая обработка"],
        ),
        DatasetReleaseOut(
            id="r1",
            path=f"{base}/r1",
            recommended=False,
            purpose="Стресс-тест или предобучение без учителя.",
            schema_notes=["нет email/file полного формата", "не подходит для основного supervised-набора"],
        ),
        DatasetReleaseOut(
            id="r2",
            path=f"{base}/r2",
            recommended=False,
            purpose="Дополнительная проверка устойчивости.",
            schema_notes=["схема беднее поздних релизов", "есть supplemental email archive"],
        ),
        DatasetReleaseOut(
            id="r3.1",
            path=f"{base}/r3.1",
            recommended=False,
            purpose="Опциональное расширение после адаптации схем.",
            schema_notes=["нет полей user/pc в email как в r4", "может потребовать отдельный mapper"],
        ),
        DatasetReleaseOut(
            id="r3.2",
            path=f"{base}/r3.2",
            recommended=False,
            purpose="Опциональное расширение после адаптации схем.",
            schema_notes=["промежуточная схема", "ценность ниже r4-r6"],
        ),
    ]
    return TrainingPlanOut(
        execution_target="Google Colab",
        baseline_release="r4.2",
        recommended_releases=["r4.1", "r4.2", "r5.1", "r5.2", "r6.1", "r6.2"],
        excluded_by_default=["r1", "r2", "r3.1", "r3.2"],
        storage_format="Parquet агрегаты на уровне dataset + user + date",
        feature_grain="Одна строка на пользователя за день с dataset_id и меткой инцидента",
        required_artifacts=[
            "processed/cert_<release>_user_day.parquet",
            "models/model.pkl или model.joblib",
            "models/scaler.joblib",
            "models/feature_columns.json",
            "reports/metrics.json",
            "reports/confusion_matrix.png",
            "reports/pr_curve.png",
        ],
        validation_strategy=[
            "baseline: обучение и оценка на r4.2 для сопоставимости",
            "multi-release CV: стратифицированная проверка на объединённой выборке",
            "release transfer: обучение на r4.1+r5.1+r6.1, тест на r4.2+r5.2+r6.2",
        ],
        releases=releases,
    )
