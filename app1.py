"""Микросервис рекомендаций банковских продуктов (FastAPI).

Запуск:
    uvicorn app1:app --host 0.0.0.0 --port 8079
    # или
    python app1.py

Эндпоинты:
    POST /predict — предсказание продукта по профилю клиента;
    GET  /health  — статус сервиса и загруженных артефактов.

Артефакты (каталог `fastapi/`, переопределяется переменной окружения `ARTIFACTS_DIR`):
    saved_model.pkl            — sklearn-пайплайн (modeling.ipynb → joblib.dump);
    preprocessing_params.json  — параметры предобработки, посчитанные при обучении
                                 (медианы, моды, границы клиппинга, возрастные бины,
                                 когорты дат, агрегаты, словарь продуктов, replacer);
    replacer.json              — словарь сворачивания редких категорий;
    personal_als.parquet       — персональные ALS-рекомендации, индекс — `ncodpers`.

Если `preprocessing_params.json` отсутствует (не переобучали после правок
CODE_REVIEW §P1.10), сервис работает на legacy-константах из этого модуля и пишет
об этом предупреждение в лог: такие константы могли разойтись с обучающим запуском.

Переменные окружения: ARTIFACTS_DIR, MODEL_PATH, PARAMS_PATH, REPLACER_PATH,
PERSONAL_RECS_PATH, LOG_LEVEL.
"""

from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import featuretools as ft
import joblib
import numpy as np
import pandas as pd
import woodwork
from fastapi import FastAPI, HTTPException, status
from pydantic import BaseModel, Field

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

# Продукты банка: значения соответствуют `ind_*_ult1` и кодам целевой переменной
# `purchase` в modeling.ipynb (код = позиция в этом списке + 1).
PRODUCTS = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
    'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
    'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1',
]

# Признаки, на которых обучена модель (см. modeling.ipynb, ячейка обучения).
# Держим их явными списками — без позиционной магии `cats[:-24]`.
NUMERIC_FEATURES = [
    'antiguedad', 'renta', 'antiguedad + renta', 'antiguedad / renta', 'renta / antiguedad',
    'antiguedad * renta', 'NATURAL_LOGARITHM(antiguedad)', 'NATURAL_LOGARITHM(renta)',
    'SQUARE_ROOT(antiguedad)', 'SQUARE_ROOT(renta)', 'renta_antiguedad_ratio', 'log_renta',
    'renta_vs_country_mean',
]
CATEGORICAL_FEATURES = [
    'ind_empleado', 'pais_residencia', 'sexo', 'ind_nuevo', 'indrel_1mes', 'tiprel_1mes',
    'indresi', 'conyuemp', 'canal_entrada', 'indfall', 'cod_prov', 'ind_actividad_cliente',
    'segmento', *PRODUCTS, 'total_products', 'age_interval', 'recommended_product_id',
    'mean_renta_by_pais_residencia', 'median_antiguedad_by_pais_residencia',
    'mean_renta_by_segmento', 'median_antiguedad_by_segmento',
]

# Колонки, которые не участвуют в признаках модели (в датасете почти все пропуски).
EXCLUDED_COLUMNS = ('indrel', 'indext')
# Базовые (не производные) колонки, которые обязаны быть в матрице признаков.
BASE_COLUMNS = [
    'ind_empleado', 'pais_residencia', 'sexo', 'ind_nuevo', 'antiguedad', 'indrel_1mes',
    'tiprel_1mes', 'indresi', 'conyuemp', 'canal_entrada', 'indfall', 'cod_prov',
    'ind_actividad_cliente', 'renta', 'segmento', *PRODUCTS, 'total_products', 'age_interval',
    'recommended_product_id',
]

# --- Legacy-константы предобработки -----------------------------------------
# Значения последнего обучающего запуска до появления preprocessing_params.json.
# Пересчитываются в modeling.ipynb и сериализуются в артефакт (CODE_REVIEW §P1.10).
LEGACY_MEDIANS: Dict[str, float] = {'age': 39.0, 'antiguedad': 50.0, 'renta': 101850.0}
LEGACY_MODES: Dict[str, Any] = {
    'ind_empleado': 'N', 'pais_residencia': 'ES', 'sexo': 'V', 'fecha_alta': '2014-07-28',
    'ind_nuevo': 0, 'ult_fec_cli_1t': '2015-12-24', 'indrel_1mes': 1.0, 'tiprel_1mes': 'I',
    'indresi': 'S', 'conyuemp': 'N', 'canal_entrada': 'KHE', 'indfall': 'N', 'tipodom': 1,
    'cod_prov': 28, 'nomprov': 'MADRID', 'ind_actividad_cliente': 0,
    'segmento': '02 - PARTICULARES', **{product: 0 for product in PRODUCTS},
}
LEGACY_CLIP_BOUNDS: Dict[str, List[float]] = {
    'renta': [26449.65, 337117.17],
    'antiguedad': [1.0, 207.0],
}
# Интервалы возраста из обучающего запуска (modeling.ipynb, ячейка 12).
LEGACY_AGE_INTERVALS: List[List[int]] = [
    [2, 23], [24, 28], [29, 36], [37, 41], [42, 45], [46, 49], [50, 54], [55, 63], [64, 164],
]
# Когорты дат из обучающего запуска (могут расходиться с обучением — см. CODE_REVIEW §2.2).
LEGACY_DATE_INTERVALS: Dict[str, List[str]] = {
    'fecha_alta': [
        '2014-08-13 – 2015-02-27', '2012-07-23 – 2012-12-10', '2013-10-18 – 2014-08-13',
        '2012-12-10 – 2013-10-18', '2004-04-23 – 2006-07-11', '2002-02-16 – 2004-04-23',
        '2011-09-01 – 2012-07-23', '2006-07-11 – 2008-09-30', '2008-09-30 – 2011-09-01',
        '2000-05-16 – 2002-02-16', '1995-01-15 – 2000-05-16', '2015-02-27 – 2016-05-31',
    ],
    'ult_fec_cli_1t': [
        '2015-06-30 – 2015-07-09', '2015-07-21 – 2015-08-03', '2015-07-09 – 2015-07-21',
        '2015-08-03 – 2015-09-14', '2015-09-14 – 2015-10-19', '2015-10-19 – 2015-11-18',
        '2015-11-18 – 2015-12-21', '2015-12-21 – 2016-01-11', '2016-01-11 – 2016-02-10',
        '2016-02-10 – 2016-03-16', '2016-03-16 – 2016-04-26', '2016-04-26 – 2016-05-30',
    ],
}


# --- Загрузка артефактов ----------------------------------------------------
def _load_json(path: Path) -> Optional[dict]:
    """Читает JSON-артефакт; при проблемах пишет в лог и возвращает None."""
    if not path.exists():
        return None
    try:
        with path.open(encoding='utf-8') as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error('Не удалось прочитать артефакт %s: %s', path, exc)
        return None


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


def _load_personal_recs() -> Optional[pd.DataFrame]:
    """Персональные ALS-рекомендации с индексом по `ncodpers`."""
    if not PERSONAL_RECS_PATH.exists():
        logger.warning('Персональные рекомендации не найдены: %s', PERSONAL_RECS_PATH)
        return None
    recs = pd.read_parquet(PERSONAL_RECS_PATH)
    if 'ncodpers' in recs.columns:  # старый формат файла — индекс не сохранён
        recs = recs.set_index('ncodpers')
    duplicated = recs.index.duplicated().sum()
    if duplicated:
        logger.warning('В %s дубли индекса ncodpers: %s, оставляю первые',
                       PERSONAL_RECS_PATH, duplicated)
        recs = recs[~recs.index.duplicated(keep='first')]
    return recs


def _update_params_from_json() -> Dict[str, Any]:
    """Подхватывает параметры предобработки из обучающего запуска.

    Артефакт перекрывает legacy-константы; чего в нём нет — берётся из legacy.
    """
    params = _load_json(PARAMS_PATH)
    if not params:
        logger.warning(
            'Параметры предобработки %s не найдены — работаю на legacy-константах. '
            'Они могли разойтись с обучающим запуском: запустите modeling.ipynb, '
            'чтобы артефакт пересобрался (CODE_REVIEW §P1.10).', PARAMS_PATH,
        )
        return {}

    logger.info('Параметры предобработки загружены: %s (сохранены %s)',
                PARAMS_PATH, params.get('created_at', 'дата неизвестна'))
    for key in ('medians', 'modes', 'clip_bounds', 'age_intervals', 'date_cohorts',
                'aggregates', 'replacer', 'label_map', 'feature_columns'):
        if key not in params:
            logger.warning('В %s нет секции %s — использую legacy-значения', PARAMS_PATH, key)
    return params


PARAMS = _update_params_from_json()

MEDIANS: Dict[str, float] = PARAMS.get('medians') or LEGACY_MEDIANS
MODES: Dict[str, Any] = PARAMS.get('modes') or LEGACY_MODES
CLIP_BOUNDS: Dict[str, List[float]] = PARAMS.get('clip_bounds') or LEGACY_CLIP_BOUNDS
AGE_INTERVALS: List[List[int]] = PARAMS.get('age_intervals') or LEGACY_AGE_INTERVALS
DATE_COHORTS: Dict[str, Any] = PARAMS.get('date_cohorts') or {}
DATE_INTERVALS: Dict[str, List[str]] = PARAMS.get('date_intervals') or LEGACY_DATE_INTERVALS
AGGREGATES: Dict[str, Dict[str, float]] = PARAMS.get('aggregates') or {}
REPLACER: Dict[str, Dict[str, str]] = PARAMS.get('replacer') or _load_json(REPLACER_PATH) or {}
# Код класса → название продукта (0 — «покупки не ожидается»).
LABEL_MAP: Dict[int, str] = {int(code): name for code, name in (PARAMS.get('label_map') or {}).items()}

MODEL = _load_model()
PERSONAL_RECS = _load_personal_recs()

CATEGORICAL_COLUMNS = [col for col in CATEGORICAL_FEATURES if col not in PRODUCTS]


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
    version='1.1.0',
)


# --- Предобработка ----------------------------------------------------------
def find_interval(input_date: Any, intervals: List[str]) -> Optional[str]:
    """Legacy-маппинг даты в интервал вида 'YYYY-MM-DD – YYYY-MM-DD'.

    Используется, только если нет `date_cohorts` из обучающего запуска
    (CODE_REVIEW §2.2: границы когорт могли разойтись с обучением).
    """
    if input_date is None or (isinstance(input_date, float) and np.isnan(input_date)):
        return None
    if isinstance(input_date, str):
        parsed = pd.to_datetime(input_date, errors='coerce')
    else:
        parsed = pd.to_datetime(input_date, errors='coerce')
    if pd.isna(parsed):
        return None

    for interval in intervals:
        start_str, end_str = interval.split(' – ')
        start_date = pd.to_datetime(start_str)
        end_date = pd.to_datetime(end_str)
        if start_date <= parsed <= end_date:
            return interval
    return None


def map_date_to_cohort(value: Any, cohort_spec: Dict[str, Any]) -> str:
    """Маппит дату в когорту из `date_cohorts` обучающего запуска.

    Границы хранятся в днях от `base_date`, интервалы полуоткрытые слева
    (`left_days < days <= right_days`) — так же, как их строит `pd.qcut`
    в `create_time_cohorts`. Режим `raw` означает, что при обучении когорты
    не строились и колонка осталась строкой с датой.
    """
    if cohort_spec.get('mode') == 'raw':
        parsed = pd.to_datetime(value, errors='coerce')
        return 'unknown' if pd.isna(parsed) else parsed.strftime('%Y-%m-%d')

    base_date = pd.Timestamp(cohort_spec['base_date']).normalize()
    parsed = pd.to_datetime(value, errors='coerce')
    if pd.isna(parsed):
        # label, который получился при обучении для пропущенной даты (обычно 'nan')
        return cohort_spec.get('missing_label', 'unknown')
    days = int((parsed.normalize() - base_date).days)
    for interval in cohort_spec['intervals']:
        if interval['left_days'] < days <= interval['right_days']:
            return interval['label']
    return 'unknown'


def _coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    for col in ('age', 'antiguedad', 'renta'):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    if 'antiguedad' in df.columns:
        df['antiguedad'] = df['antiguedad'].replace(np.float64(-999999.0), np.nan)
    return df


def _fill_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Пропуски числовых — медианами, категориальных — модами (порядок как при обучении)."""
    for col in ('age', 'antiguedad', 'renta'):
        if col in df.columns:
            df[col] = df[col].fillna(MEDIANS.get(col, LEGACY_MEDIANS[col]))
    for col, value in MODES.items():
        if col in df.columns:
            df[col] = df[col].fillna(value)
    return df


def _add_age_interval(df: pd.DataFrame) -> pd.DataFrame:
    bins = [interval[0] for interval in AGE_INTERVALS] + [AGE_INTERVALS[-1][1]]
    labels = [f'{interval[0]}-{interval[1]}' for interval in AGE_INTERVALS]
    # возраст за пределами обучающих бинов относим к крайнему интервалу,
    # чтобы признак не превращался в NaN (в обучении таких значений не было)
    age = df['age'].clip(lower=bins[0], upper=bins[-1])
    df['age_interval'] = pd.cut(age, bins=bins, labels=labels)
    return df


def _add_date_cohorts(df: pd.DataFrame) -> pd.DataFrame:
    for col in ('fecha_alta', 'ult_fec_cli_1t'):
        if col not in df.columns:
            continue
        if DATE_COHORTS.get(col):
            spec = DATE_COHORTS[col]
            df[col] = df[col].apply(lambda value, spec=spec: map_date_to_cohort(value, spec))
        else:
            df[col] = df[col].apply(lambda value: find_interval(value, DATE_INTERVALS[col]))
    return df


def _fold_rare_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Сворачивает редкие категории в 'other' по словарю из обучения."""
    for col in CATEGORICAL_COLUMNS:
        if col in df.columns and col in REPLACER:
            # replace, а не map: в словаре только редкие значения
            df[col] = df[col].replace(REPLACER[col])
    return df


def _clip_numbers(df: pd.DataFrame) -> pd.DataFrame:
    for col, (lower, upper) in CLIP_BOUNDS.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)
    return df


def _add_personal_recommendation(df: pd.DataFrame) -> pd.DataFrame:
    """Добавляет признак `recommended_product_id` из ALS-модели (по ID клиента)."""
    ncodpers = df['ncodpers'].iloc[0]
    recommendation = None
    if PERSONAL_RECS is not None:
        try:
            recommendation = PERSONAL_RECS['recommended_product_id'].get(ncodpers)
        except (KeyError, TypeError):  # pragma: no cover — защита от неожиданного формата файла
            logger.exception('Ошибка поиска ALS-рекомендации для ncodpers=%s', ncodpers)
    if recommendation is None or pd.isna(recommendation):
        logger.info('Персональная ALS-рекомендация для ncodpers=%s не найдена — 0', ncodpers)
        recommendation = 0
    df['recommended_product_id'] = recommendation
    return df


def _add_total_products(df: pd.DataFrame) -> pd.DataFrame:
    for product in PRODUCTS:
        df[product] = pd.to_numeric(df[product], errors='coerce')
    df['total_products'] = df[PRODUCTS].sum(axis=1)
    return df


def manual_transformations(df: pd.DataFrame) -> pd.DataFrame:
    """Ручные и агрегатные признаки.

    Агрегаты берутся из `preprocessing_params.json` (как при обучении). Если артефакта
    нет, считаются по одной строке запроса — это сдвигает признак (CODE_REVIEW §2.4),
    поэтому такой режим только для legacy-совместимости и логируется предупреждением.
    """
    df['renta_antiguedad_ratio'] = df['renta'] / (df['antiguedad'] + 1)
    df['log_renta'] = np.log1p(df['renta'])

    for col in ('pais_residencia', 'segmento'):
        renta_key = f'mean_renta_by_{col}'
        antiguedad_key = f'median_antiguedad_by_{col}'
        if AGGREGATES.get(renta_key) and AGGREGATES.get(antiguedad_key):
            df[renta_key] = df[col].map(AGGREGATES[renta_key]).astype(float)
            df[antiguedad_key] = df[col].map(AGGREGATES[antiguedad_key]).astype(float)
        else:
            logger.warning('Агрегаты для %s не найдены в параметрах — считаю по строке запроса '
                           '(см. CODE_REVIEW §2.4)', col)
            df[renta_key] = df.groupby(col)['renta'].transform('mean')
            df[antiguedad_key] = df.groupby(col)['antiguedad'].transform('median')

    df['renta_vs_country_mean'] = df['renta'] / df['mean_renta_by_pais_residencia']
    return df


def feature_engineering(df: pd.DataFrame) -> pd.DataFrame:
    """Генерирует производные признаки (арифметика через featuretools + агрегаты)."""
    entity_set = ft.EntitySet(id='bank_data')
    entity_set = entity_set.add_dataframe(
        dataframe_name='main',
        dataframe=df,
        index='unique_id',
        make_index=True,
        logical_types={
            'antiguedad': woodwork.logical_types.Double,
            'renta': woodwork.logical_types.Double,
            **{col: woodwork.logical_types.Categorical for col in df.columns
               if col not in ['antiguedad', 'renta']},
        },
    )

    feature_matrix, _ = ft.dfs(
        entityset=entity_set,
        target_dataframe_name='main',
        trans_primitives=['add_numeric', 'multiply_numeric', 'divide_numeric',
                          'natural_logarithm', 'square_root'],
        agg_primitives=['mean', 'median', 'std', 'max', 'min', 'count', 'num_unique'],
        where_primitives=['count'],
        max_depth=2,
        features_only=False,
        verbose=False,
    )

    df = feature_matrix.drop(columns=['unique_id', 'index'], errors='ignore')
    df = manual_transformations(df)
    # Удаляем дубликаты колонок, которые может породить featuretools
    return df.loc[:, ~df.columns.duplicated()]


def prepare_features(profile: Dict[str, Any]) -> pd.DataFrame:
    """Превращает профиль клиента в матрицу признаков той же формы, что и на обучении."""
    row = pd.DataFrame([profile])
    row = row.drop(columns=list(EXCLUDED_COLUMNS), errors='ignore')

    row = _coerce_numeric(row)
    row = _fill_missing(row)
    row = _add_age_interval(row)
    row = _add_date_cohorts(row)
    row = _fold_rare_categories(row)
    row = _clip_numbers(row)
    row = _add_personal_recommendation(row)
    row = _add_total_products(row)

    missing_base = [col for col in BASE_COLUMNS if col not in row.columns]
    if missing_base:
        raise HTTPException(
            status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
            detail=f'В профиле не хватает обязательных полей: {missing_base}',
        )
    row = row[BASE_COLUMNS]

    row['antiguedad'] = row['antiguedad'].astype(int)
    row = feature_engineering(row)

    for col in NUMERIC_FEATURES:
        if col in row.columns:
            row[col] = pd.to_numeric(row[col], errors='coerce')
    for col in CATEGORICAL_FEATURES:
        if col in row.columns:
            row[col] = row[col].astype('str')

    expected = set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    absent = sorted(expected - set(row.columns))
    if absent:
        logger.error('В матрице признаков нет колонок модели: %s', absent)
    return row


# --- Эндпоинты --------------------------------------------------------------
@app.get('/health', response_model=HealthResponse, summary='Статус сервиса и артефактов')
def health() -> HealthResponse:
    return HealthResponse(
        status='ok' if MODEL is not None else 'degraded',
        model_loaded=MODEL is not None,
        model_path=str(MODEL_PATH),
        params_source='preprocessing_params.json' if PARAMS else 'legacy-константы',
        personal_recs_loaded=PERSONAL_RECS is not None,
    )


@app.post('/predict', response_model=PredictionResponse, summary='Предсказание продукта')
def predict(profile: ClientProfile) -> PredictionResponse:
    """Скорит профиль клиента.

    Обычный `def`, а не `async def`: внутри CPU-bound pandas/featuretools,
    uvicorn сам вынесет вызов в thread pool и не заблокирует event loop
    (CODE_REVIEW §2.3).
    """
    if MODEL is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=f'Модель не загружена ({MODEL_PATH}). Обучение: modeling.ipynb, '
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
