"""Микросервис рекомендаций банковских продуктов (FastAPI).

Запуск:
    uvicorn app1:app --host 0.0.0.0 --port 8079
    # или
    python app1.py
    # или через docker compose (см. README.md):
    docker compose -f fastapi/docker-compose.yaml up --build

Эндпоинты:
    POST /predict  — предсказание продукта по профилю клиента;
    GET  /health   — статус сервиса и загруженных артефактов;
    GET  /metrics  — метрики в формате Prometheus (счётчики запросов,
                     гистограмма латентности /predict).

Предобработка вынесена в общий модуль `preprocessing.py` (CODE_REVIEW §P2.14) —
сервис и обучение используют одни и те же функции и константы.

Артефакты (каталог `fastapi/`, переопределяется переменной окружения `ARTIFACTS_DIR`):
    saved_model.pkl            — sklearn-пайплайн (modeling.ipynb → joblib.dump);
    preprocessing_params.json  — параметры предобработки, посчитанные при обучении
                                 (медианы, моды, границы клиппинга, возрастные бины,
                                 когорты дат, агрегаты, словарь продуктов, replacer);
    replacer.json              — словарь сворачивания редких категорий;
    personal_als.parquet       — персональные ALS-рекомендации, индекс — `ncodpers`.

Если `preprocessing_params.json` отсутствует (не переобучали после правок
CODE_REVIEW §P1.10), сервис работает на legacy-константах из `preprocessing.py`
и пишет об этом предупреждение в лог: такие константы могли разойтись
с обучающим запуском.

Переменные окружения: ARTIFACTS_DIR, MODEL_PATH, PARAMS_PATH, REPLACER_PATH,
PERSONAL_RECS_PATH, LOG_LEVEL.
"""

from __future__ import annotations

import logging
import os
import time
from pathlib import Path
from typing import List, Optional, Union

import joblib
import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from pydantic import BaseModel, Field

from preprocessing import PreprocessingParams, load_json, load_personal_recs
from preprocessing import prepare_features as build_features

logging.basicConfig(
    level=os.getenv('LOG_LEVEL', 'INFO'),
    format='%(asctime)s %(levelname)s %(name)s %(message)s',
)
logger = logging.getLogger('bank_recommender')

ARTIFACTS_DIR = Path(os.getenv('ARTIFACTS_DIR', 'fastapi'))
MODEL_PATH = Path(os.getenv('MODEL_PATH', ARTIFACTS_DIR / 'saved_model.pkl'))
PARAMS_PATH = Path(os.getenv('PARAMS_PATH', ARTIFACTS_DIR / 'preprocessing_params.json'))
REPLACER_PATH = Path(os.getenv('REPLACER_PATH', ARTIFACTS_DIR / 'replacer.json'))
PERSONAL_RECS_PATH = Path(os.getenv('PERSONAL_RECS_PATH', ARTIFACTS_DIR / 'personal_als.parquet'))

# --- Метрики Prometheus (опционально: без prometheus_client /metrics отдаёт 503) --
try:
    from prometheus_client import CONTENT_TYPE_LATEST, Counter, Histogram, generate_latest

    REQUEST_COUNT = Counter(
        'bank_recommender_requests_total',
        'Число HTTP-запросов к сервису.',
        ['endpoint', 'status'],
    )
    PREDICT_LATENCY = Histogram(
        'bank_recommender_predict_latency_seconds',
        'Латентность POST /predict (предобработка + predict_proba).',
        buckets=(0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0),
    )
    PREDICTION_COUNT = Counter(
        'bank_recommender_predictions_total',
        'Число предсказаний по кодам классов.',
        ['prediction'],
    )
    METRICS_ENABLED = True
except ImportError:  # pragma: no cover — в прод-окружении зависимость есть
    logger.warning('prometheus_client не установлен — /metrics недоступен')
    REQUEST_COUNT = PREDICT_LATENCY = PREDICTION_COUNT = None
    METRICS_ENABLED = False


def _observe_request(endpoint: str, status_code: int, latency: Optional[float] = None,
                     prediction: Optional[int] = None) -> None:
    """Учитывает запрос в счётчиках Prometheus (no-op без prometheus_client)."""
    if not METRICS_ENABLED:
        return
    REQUEST_COUNT.labels(endpoint=endpoint, status=status_code).inc()
    if latency is not None:
        PREDICT_LATENCY.observe(latency)
    if prediction is not None:
        PREDICTION_COUNT.labels(prediction=prediction).inc()


# --- Загрузка артефактов ----------------------------------------------------
def _load_model():
    """Загружает sklearn-пайплайн; сервис поднимается даже без модели (см. /health)."""
    if not MODEL_PATH.exists():
        logger.warning('Модель не найдена: %s (POST /predict вернёт 503)', MODEL_PATH)
        return None
    try:
        model = joblib.load(MODEL_PATH)
    except Exception:  # noqa: BLE001 — артефакт может быть несовместим с версией sklearn
        logger.exception('Не удалось загрузить модель %s', MODEL_PATH)
        return None
    logger.info('Модель загружена: %s (классы: %s)', MODEL_PATH, getattr(model, 'classes_', 'n/a'))
    return model


PARAMS = PreprocessingParams.from_json(PARAMS_PATH)
if not PARAMS.replacer:
    # replacer отдельно от preprocessing_params.json — старый формат артефактов
    PARAMS.replacer = load_json(REPLACER_PATH) or {}
# Код класса → название продукта (0 — «покупки не ожидается»).
LABEL_MAP = dict(PARAMS.label_map)

MODEL = _load_model()
PERSONAL_RECS = load_personal_recs(PERSONAL_RECS_PATH)


# --- Схемы запроса и ответа -------------------------------------------------
class ClientProfile(BaseModel):
    """Профиль клиента: те же поля, что и в `train_ver2.csv` (см. `columns.txt`).

    Обязателен только идентификатор клиента — остальные поля необязательны и при
    отсутствии заполняются значениями из обучающего запуска (медианы/моды), как
    это делалось на обучении. Лишние поля запроса игнорируются.
    """

    model_config = {'extra': 'ignore'}

    ncodpers: int = Field(..., description='ID клиента (нужен для персональной ALS-рекомендации)')
    fecha_dato: Optional[str] = Field(None, description='Дата среза; в признаках не используется')
    ind_empleado: Optional[str] = None
    pais_residencia: Optional[str] = None
    sexo: Optional[str] = None
    age: Optional[float] = Field(None, description='Возраст, лет; None → медиана обучения')
    fecha_alta: Optional[str] = Field(None, description='Дата первого договора')
    ind_nuevo: Optional[float] = None
    antiguedad: Optional[float] = Field(None, description='Стаж, месяцев; None → медиана обучения')
    indrel: Optional[float] = Field(None, description='Не используется моделью')
    ult_fec_cli_1t: Optional[str] = None
    indrel_1mes: Optional[Union[str, float]] = None
    tiprel_1mes: Optional[str] = None
    indresi: Optional[str] = None
    indext: Optional[str] = Field(None, description='Не используется моделью')
    conyuemp: Optional[str] = None
    canal_entrada: Optional[str] = None
    indfall: Optional[str] = None
    tipodom: Optional[float] = Field(None, description='Не используется моделью')
    cod_prov: Optional[float] = None
    nomprov: Optional[str] = Field(None, description='Не используется моделью')
    ind_actividad_cliente: Optional[float] = None
    renta: Optional[float] = Field(None, description='Доход домохозяйства; None → медиана обучения')
    segmento: Optional[str] = None

    # Флаги владения продуктами (ind_*_ult1): 1 — продукт есть, 0/None — нет.
    ind_ahor_fin_ult1: Optional[float] = None
    ind_aval_fin_ult1: Optional[float] = None
    ind_cco_fin_ult1: Optional[float] = None
    ind_cder_fin_ult1: Optional[float] = None
    ind_cno_fin_ult1: Optional[float] = None
    ind_ctju_fin_ult1: Optional[float] = None
    ind_ctma_fin_ult1: Optional[float] = None
    ind_ctop_fin_ult1: Optional[float] = None
    ind_ctpp_fin_ult1: Optional[float] = None
    ind_deco_fin_ult1: Optional[float] = None
    ind_deme_fin_ult1: Optional[float] = None
    ind_dela_fin_ult1: Optional[float] = None
    ind_ecue_fin_ult1: Optional[float] = None
    ind_fond_fin_ult1: Optional[float] = None
    ind_hip_fin_ult1: Optional[float] = None
    ind_plan_fin_ult1: Optional[float] = None
    ind_pres_fin_ult1: Optional[float] = None
    ind_reca_fin_ult1: Optional[float] = None
    ind_tjcr_fin_ult1: Optional[float] = None
    ind_valo_fin_ult1: Optional[float] = None
    ind_viv_fin_ult1: Optional[float] = None
    ind_nomina_ult1: Optional[float] = None
    ind_nom_pens_ult1: Optional[float] = None
    ind_recibo_ult1: Optional[float] = None


class PredictedProduct(BaseModel):
    code: int = Field(..., description='Код класса модели (0 — покупки не ожидается)')
    product: Optional[str] = Field(None, description='Название продукта')
    probability: float = Field(..., description='Вероятность класса')


class PredictionResponse(BaseModel):
    prediction: int = Field(..., description='Предсказанный код продукта')
    product: Optional[str] = Field(None, description='Название продукта для кода prediction')
    confidence: float = Field(..., description='Вероятность предсказанного класса')
    top_k: List[PredictedProduct] = Field(..., description='Топ продуктов по вероятности')


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    model_path: str
    params_source: str
    personal_recs_loaded: bool


app = FastAPI(
    title='Bank product recommender',
    description='Рекомендация банковского продукта по профилю клиента (см. README.md).',
    version='1.2.0',
)


# --- Предобработка ----------------------------------------------------------
def prepare_features(profile: dict) -> pd.DataFrame:
    """Профиль клиента → матрица признаков (единая логика в `preprocessing`).

    Тонкая обёртка: читает актуальные параметры из `PARAMS` (тесты подменяют
    их через monkeypatch) и превращает ValueError в HTTP 422.
    """
    try:
        return build_features(profile, PARAMS, PERSONAL_RECS)
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=str(exc),
        ) from exc


# --- Эндпоинты --------------------------------------------------------------
@app.get('/health', response_model=HealthResponse, summary='Статус сервиса и артефактов')
def health() -> HealthResponse:
    _observe_request('health', status.HTTP_200_OK)
    return HealthResponse(
        status='ok' if MODEL is not None else 'degraded',
        model_loaded=MODEL is not None,
        model_path=str(MODEL_PATH),
        params_source=PARAMS.source,
        personal_recs_loaded=PERSONAL_RECS is not None,
    )


@app.get('/metrics', summary='Метрики Prometheus')
def metrics() -> Response:
    """Метрики для Prometheus-скрейпинга (см. fastapi/prometheus/prometheus.yml)."""
    if not METRICS_ENABLED:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail='prometheus_client не установлен',
        )
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


@app.post('/predict', response_model=PredictionResponse, summary='Предсказание продукта')
def predict(profile: ClientProfile) -> PredictionResponse:
    """Скорит профиль клиента.

    Обычный `def`, а не `async def`: внутри CPU-bound pandas/featuretools,
    uvicorn сам вынесет вызов в thread pool и не заблокирует event loop
    (CODE_REVIEW §2.3).
    """
    started = time.perf_counter()
    if MODEL is None:
        _observe_request('predict', status.HTTP_503_SERVICE_UNAVAILABLE,
                         time.perf_counter() - started)
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'Модель не найдена ({MODEL_PATH}). Обучение: modeling.ipynb, '
                   'сохранение: joblib.dump(model, "fastapi/saved_model.pkl")',
        )

    features = prepare_features(profile.model_dump())
    probabilities = MODEL.predict_proba(features)[0]
    classes = [int(cls) for cls in MODEL.classes_]

    # Стабильная сортировка по убыванию вероятности: при равных вероятностях классы
    # остаются в порядке `classes_` (возрастание кода), поэтому `prediction` всегда
    # совпадает с первым элементом `top_k` — так же, как вёл бы себя `np.argmax`.
    order = np.argsort(-probabilities, kind='stable')
    ranked = [
        PredictedProduct(
            code=classes[index],
            product=LABEL_MAP.get(classes[index]),
            probability=float(probabilities[index]),
        )
        for index in order
    ]
    best = ranked[0]
    latency = time.perf_counter() - started
    _observe_request('predict', status.HTTP_200_OK, latency, best.code)
    logger.info('predict: ncodpers=%s → %s (%s, p=%.3f)',
                profile.ncodpers, best.code, LABEL_MAP.get(best.code, '?'), best.probability)
    return PredictionResponse(
        prediction=best.code,
        product=best.product,
        confidence=best.probability,
        top_k=ranked[:3],
    )


if __name__ == '__main__':
    import uvicorn

    uvicorn.run(
        app,
        host=os.getenv('THE_HOST', '0.0.0.0'),
        port=int(os.getenv('THE_PORT', 8079)),
    )
