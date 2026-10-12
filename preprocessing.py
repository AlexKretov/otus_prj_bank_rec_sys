"""Общая предобработка для обучения и сервиса.

Модуль — единый источник истины для обучения (`modeling.ipynb`), офлайн-проверки
(`rec_sys.ipynb`) и сервиса (`app1.py`): все они используют одни и те же функции
и константы, поэтому train/serve skew минимизируется по построению.

Импорт модуля лёгкий: чистые функции (бины, когорты, клиппинг, сворачивание
редких категорий, арифметика `feature_engineering`) работают на pandas/numpy
и не тянут тяжёлых зависимостей. `tqdm` — опционален: если его нет,
прогресс-бары молча отключаются.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional

import numpy as np
import pandas as pd

try:  # прогресс-бары нужны только для больших батчей, зависимость опциональна
    from tqdm.auto import tqdm as _tqdm
    _HAS_TQDM = True
except ImportError:  # pragma: no cover — tqdm почти всегда есть в окружении
    _HAS_TQDM = False
    _tqdm = None

logger = logging.getLogger('bank_recommender.preprocessing')

# Батчи меньше этого размера не показывают прогресс-бар: инференс API — одна
# строка на запрос, и любой tqdm в этом случае был бы шумом в логах.
PROGRESS_MIN_ROWS = 1000

# Продукты банка: значения соответствуют `ind_*_ult1`, код целевой переменной
# `purchase` в modeling.ipynb — позиция в этом списке + 1.
PRODUCTS = [
    'ind_ahor_fin_ult1', 'ind_aval_fin_ult1', 'ind_cco_fin_ult1', 'ind_cder_fin_ult1',
    'ind_cno_fin_ult1', 'ind_ctju_fin_ult1', 'ind_ctma_fin_ult1', 'ind_ctop_fin_ult1',
    'ind_ctpp_fin_ult1', 'ind_deco_fin_ult1', 'ind_deme_fin_ult1', 'ind_dela_fin_ult1',
    'ind_ecue_fin_ult1', 'ind_fond_fin_ult1', 'ind_hip_fin_ult1', 'ind_plan_fin_ult1',
    'ind_pres_fin_ult1', 'ind_reca_fin_ult1', 'ind_tjcr_fin_ult1', 'ind_valo_fin_ult1',
    'ind_viv_fin_ult1', 'ind_nomina_ult1', 'ind_nom_pens_ult1', 'ind_recibo_ult1',
]

# Специальные классы целевой переменной (см. modeling.ipynb, «Целевая переменная»).
NO_PURCHASE_CLASS = 0  # в 2016 году клиент ничего не купил
OTHER_CLASS = 99  # первым куплен продукт вне шорт-листа

# Признаки, на которых обучена модель (см. modeling.ipynb, ячейка обучения).
# Держим их явными списками — без позиционной магии вроде `cats[:-24]`.
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

# Категориальные колонки, по которым сворачиваются редкие значения
# (продуктовые флаги 0/1 не сворачиваются — явно, а не срезом списка).
CATEGORICAL_COLUMNS = [col for col in CATEGORICAL_FEATURES if col not in PRODUCTS]

# Колонки, которые не участвуют в признаках модели (в датасете почти все пропуски).
EXCLUDED_COLUMNS = ('indrel', 'indext')
# Базовые (не производные) колонки, которые обязаны быть в матрице признаков.
BASE_COLUMNS = [
    'ind_empleado', 'pais_residencia', 'sexo', 'ind_nuevo', 'antiguedad', 'indrel_1mes',
    'tiprel_1mes', 'indresi', 'conyuemp', 'canal_entrada', 'indfall', 'cod_prov',
    'ind_actividad_cliente', 'renta', 'segmento', *PRODUCTS, 'total_products', 'age_interval',
    'recommended_product_id',
]

# --- Запасные константы предобработки ----------------------------------------
# Используются только если preprocessing_params.json отсутствует или неполон.
# При штатном запуске они пересчитываются в modeling.ipynb и сериализуются.
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
# Когорты дат из сохранённого обучающего запуска для резервного режима.
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


# --- Утилиты логирования и прогресса -----------------------------------------

@contextmanager
def _timed(stage: str, level: int = logging.INFO) -> Iterator[None]:
    """Замеряет и логирует длительность стадии.

    INFO — для стадий обучения (минуты, важны в отчёте), DEBUG — для мелких
    шагов инференса, где INFO был бы шумом (десятки миллисекунд на запрос).
    """
    start = time.perf_counter()
    logger.log(level, '%s — старт', stage)
    try:
        yield
    finally:
        logger.log(level, '%s — готово за %.2f с', stage, time.perf_counter() - start)


def _progress(iterable, *, desc: Optional[str] = None, disable: bool = False, **kwargs):
    """tqdm-обёртка: молчит для мелких батчей и при отсутствии tqdm.

    Инференс API обрабатывает одну строку на запрос — прогресс-бар там был бы
    бесполезен, а `disable=True` его гасит. Порог `PROGRESS_MIN_ROWS`
    применяется вызывающей стороной: `disable=len(df) < PROGRESS_MIN_ROWS`.
    """
    if disable or not _HAS_TQDM:
        return iterable
    return _tqdm(iterable, desc=desc, leave=False, **kwargs)


def _shape(df: pd.DataFrame) -> str:
    """Короткое описание формы фрейма для логов: '1234×57'."""
    return f'{len(df)}×{df.shape[1]}'


@dataclass
class PreprocessingParams:
    """Параметры предобработки из обучающего запуска.

    Сервис читает их из `preprocessing_params.json`; чего в файле нет —
    берётся из запасных констант (с предупреждением в лог). Поле `source`
    говорит, откуда параметры: путь к файлу или `'запасные константы'`.
    """

    medians: Dict[str, float] = field(default_factory=lambda: dict(LEGACY_MEDIANS))
    modes: Dict[str, Any] = field(default_factory=lambda: dict(LEGACY_MODES))
    clip_bounds: Dict[str, List[float]] = field(default_factory=lambda: dict(LEGACY_CLIP_BOUNDS))
    age_intervals: List[List[int]] = field(
        default_factory=lambda: [list(i) for i in LEGACY_AGE_INTERVALS])
    date_cohorts: Dict[str, Any] = field(default_factory=dict)
    date_intervals: Dict[str, List[str]] = field(
        default_factory=lambda: dict(LEGACY_DATE_INTERVALS))
    aggregates: Dict[str, Dict[str, float]] = field(default_factory=dict)
    replacer: Dict[str, Dict[str, str]] = field(default_factory=dict)
    label_map: Dict[int, str] = field(default_factory=dict)
    feature_columns: Dict[str, List[str]] = field(default_factory=dict)
    drift_reference: Optional[Dict[str, Any]] = None
    created_at: Optional[str] = None
    source: str = 'запасные константы'

    @classmethod
    def from_dict(cls, payload: Optional[dict], source: str) -> 'PreprocessingParams':
        """Собирает параметры из словаря; отсутствующие секции — из резервных констант."""
        if not payload:
            logger.warning(
                'Параметры предобработки %s не найдены — работаю на запасных константах. '
                'Они могут отличаться от последнего обучающего запуска: запустите '
                'modeling.ipynb, чтобы артефакт пересобрался.', source,
            )
            return cls()
        for key in ('medians', 'modes', 'clip_bounds', 'age_intervals', 'date_cohorts',
                    'aggregates', 'replacer', 'label_map', 'feature_columns'):
            if key not in payload:
                logger.warning('В %s нет секции %s — использую резервные значения', source, key)
        params = cls(
            medians=payload.get('medians') or dict(LEGACY_MEDIANS),
            modes=payload.get('modes') or dict(LEGACY_MODES),
            clip_bounds=payload.get('clip_bounds') or dict(LEGACY_CLIP_BOUNDS),
            age_intervals=payload.get('age_intervals') or [list(i) for i in LEGACY_AGE_INTERVALS],
            date_cohorts=payload.get('date_cohorts') or {},
            date_intervals=payload.get('date_intervals') or dict(LEGACY_DATE_INTERVALS),
            aggregates=payload.get('aggregates') or {},
            replacer=payload.get('replacer') or {},
            label_map={int(code): name for code, name in (payload.get('label_map') or {}).items()},
            feature_columns=payload.get('feature_columns') or {},
            drift_reference=payload.get('drift_reference'),
            created_at=payload.get('created_at'),
            source=source,
        )
        logger.info(
            'Параметры предобработки собраны из %s: медиан=%d, мод=%d, '
            'clip-колонок=%d, возрастных бинов=%d, когорт=%s, агрегатов=%s, '
            'replacer-колонок=%d, классов=%d',
            source, len(params.medians), len(params.modes), len(params.clip_bounds),
            len(params.age_intervals), list(params.date_cohorts), list(params.aggregates),
            sum(1 for values in params.replacer.values() if values), len(params.label_map),
        )
        return params

    @classmethod
    def from_json(cls, path: Path) -> 'PreprocessingParams':
        """Читает `preprocessing_params.json`; при проблемах — запасные константы."""
        params = load_json(path)
        if params is not None:
            logger.info('Параметры предобработки загружены: %s (сохранены %s)',
                        path, params.get('created_at', 'дата неизвестна'))
        return cls.from_dict(params, source=str(path))


def load_json(path: Path) -> Optional[dict]:
    """Читает JSON-артефакт; при проблемах пишет в лог и возвращает None."""
    if not path.exists():
        logger.debug('JSON-артефакт не найден: %s', path)
        return None
    try:
        with path.open(encoding='utf-8') as file:
            return json.load(file)
    except (OSError, json.JSONDecodeError) as exc:
        logger.error('Не удалось прочитать артефакт %s: %s', path, exc)
        return None


def load_personal_recs(path: Path) -> Optional[pd.DataFrame]:
    """Персональные ALS-рекомендации с индексом по `ncodpers`."""
    if not path.exists():
        logger.warning('Персональные рекомендации не найдены: %s', path)
        return None
    with _timed(f'load_personal_recs: чтение {path}', level=logging.DEBUG):
        recs = pd.read_parquet(path)
    if 'ncodpers' in recs.columns:  # резервный формат файла — индекс не сохранён
        recs = recs.set_index('ncodpers')
    duplicated = int(recs.index.duplicated().sum())
    if duplicated:
        logger.warning('В %s дубли индекса ncodpers: %s, оставляю непустые', path, duplicated)
        # непустые рекомендации — первыми, чтобы keep='first' брал их, а не NaN-строки
        notna_first = np.argsort(~recs['recommended_product_id'].notna().to_numpy(), kind='stable')
        recs = recs.iloc[notna_first]
        recs = recs[~recs.index.duplicated(keep='first')]
    logger.info('Персональные ALS-рекомендации: %s, уникальных клиентов: %s',
                path, len(recs))
    return recs


@lru_cache(maxsize=None)
def _parse_interval_bounds(interval: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    """Границы резервного интервала 'YYYY-MM-DD – YYYY-MM-DD', разобранные один раз.

    Без кэша find_interval парсил по 24 строки дат на каждое значение —
    на инференсе это ~8 мс на запрос только на парсинг.
    """
    start_str, end_str = interval.split(' – ')
    return pd.Timestamp(start_str), pd.Timestamp(end_str)


def find_interval(input_date: Any, intervals: List[str]) -> Optional[str]:
    """Резервный маппинг даты в интервал вида 'YYYY-MM-DD – YYYY-MM-DD'.

    Используется, только если в артефакте нет `date_cohorts` из обучающего запуска.
    """
    if input_date is None or (isinstance(input_date, float) and np.isnan(input_date)):
        return None
    parsed = pd.to_datetime(input_date, errors='coerce')
    if pd.isna(parsed):
        return None

    for interval in intervals:
        start_date, end_date = _parse_interval_bounds(interval)
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


def coerce_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит числовые колонки к числам; `-999999` в стаже — в NaN."""
    with _timed(f'coerce_numeric: {_shape(df)}', level=logging.DEBUG):
        for col in ('age', 'antiguedad', 'renta'):
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
        if 'antiguedad' in df.columns:
            df['antiguedad'] = df['antiguedad'].replace(np.float64(-999999.0), np.nan)
    return df


def fill_missing(df: pd.DataFrame, medians: Dict[str, float],
                 modes: Dict[str, Any]) -> pd.DataFrame:
    """Пропуски числовых — медианами, категориальных — модами (порядок как при обучении).

    Заполняются только колонки, в которых реально есть пропуски: ~40 безусловных
    `fillna` на однострочном фрейме запроса стоили заметной доли латентности
    (каждый вызов pandas — это копирование блока). Значения те же: `fillna(dict)`
    внутри pandas заполняет колонки по одной, как и прежний цикл.
    """
    if df.empty:
        return df
    with_missing = set(df.columns[df.isna().any(axis=0)])
    fills: Dict[str, Any] = {}
    for col in ('age', 'antiguedad', 'renta'):
        if col in df.columns and col in with_missing:
            fills[col] = medians.get(col, LEGACY_MEDIANS[col])
    for col, value in modes.items():
        if col in df.columns and col in with_missing:
            fills[col] = value
    if fills:
        df = df.fillna(fills)
    logger.debug('fill_missing: заполнено колонок %d (медианы: %d, моды: %d)',
                 len(fills),
                 sum(1 for c in fills if c in ('age', 'antiguedad', 'renta')),
                 len(fills) - sum(1 for c in fills if c in ('age', 'antiguedad', 'renta')))
    return df


def add_age_interval(df: pd.DataFrame, age_intervals: List[List[int]]) -> pd.DataFrame:
    """Возрастные корзины из обучающего запуска."""
    bins = [interval[0] for interval in age_intervals] + [age_intervals[-1][1]]
    labels = [f'{interval[0]}-{interval[1]}' for interval in age_intervals]
    # возраст за пределами обучающих бинов относим к крайнему интервалу,
    # чтобы признак не превращался в NaN (в обучении таких значений не было).
    # include_lowest: интервалы pd.cut слева открыты, без него минимальный возраст
    # ровно на границе (после clip) выпадал бы в NaN.
    age = df['age'].clip(lower=bins[0], upper=bins[-1])
    df['age_interval'] = pd.cut(age, bins=bins, labels=labels, include_lowest=True)
    logger.debug('add_age_interval: интервалов %d, границы %s', len(age_intervals),
                 f'{bins[0]}…{bins[-1]}')
    return df


def add_date_cohorts(df: pd.DataFrame, date_cohorts: Dict[str, Any],
                     date_intervals: Dict[str, List[str]]) -> pd.DataFrame:
    """Когорты дат: спецификация из обучения, иначе резервные интервалы."""
    processed: List[str] = []
    for col in ('fecha_alta', 'ult_fec_cli_1t'):
        if col not in df.columns:
            continue
        if date_cohorts.get(col):
            spec = date_cohorts[col]
            df[col] = df[col].apply(lambda value, spec=spec: map_date_to_cohort(value, spec))
            logger.debug('add_date_cohorts: %s → %s (интервалов %d)',
                         col, spec.get('mode'), len(spec.get('intervals', ())))
        else:
            df[col] = df[col].apply(lambda value: find_interval(value, date_intervals[col]))
            logger.debug('add_date_cohorts: %s → резервные интервалы (%d шт.)',
                         col, len(date_intervals[col]))
        processed.append(col)
    if not processed:
        logger.debug('add_date_cohorts: колонок для когорт нет')
    return df


def fold_rare_categories(df: pd.DataFrame, replacer: Dict[str, Dict[str, str]],
                         columns: Optional[List[str]] = None) -> pd.DataFrame:
    """Сворачивает редкие категории в 'other' по словарю из обучения.

    Ключи словаря, сгенерированного при обучении, — строки (на обучении
    колонки приводились к str до подсчёта частот). Поэтому для словарей со
    строковыми ключами сравнение ведём в строковом домене: значения колонки
    приводятся к str, и float `28.0` из запроса матчится с ключом `'28.0'`
    ровно так же, как на обучении. Словари с нестроковыми ключами
    (ручные/тестовые) обрабатываются в исходном числовом домене.

    Пустые словари — тождественное преобразование — пропускаются.
    """
    target_columns = [
        col for col in (CATEGORICAL_COLUMNS if columns is None else columns)
        if replacer.get(col) and col in df.columns
    ]
    if not target_columns:
        logger.debug('fold_rare_categories: нечего сворачивать')
        return df

    disable = len(df) < PROGRESS_MIN_ROWS
    replaced_total = 0
    for col in _progress(target_columns, desc='fold_rare_categories', disable=disable):
        mapping = replacer[col]
        if all(isinstance(key, str) for key in mapping):
            values = df[col].astype(str)
            rare_mask = values.isin(mapping)
        else:  # числовые ключи: сравниваем в исходном домене
            values = df[col]
            rare_mask = values.isin(mapping)
        n_replaced = int(rare_mask.sum())
        replaced_total += n_replaced
        if n_replaced:
            if isinstance(df[col].dtype, pd.CategoricalDtype):
                # присвоение 'other' в category-dtype упало бы: значения нет
                # в категориях — переводим колонку в object
                df[col] = df[col].astype(object)
            df.loc[rare_mask, col] = values[rare_mask].map(mapping)
    logger.debug('fold_rare_categories: колонок %d, значений свёрнуто %d',
                 len(target_columns), replaced_total)
    return df


def clip_numbers(df: pd.DataFrame, clip_bounds: Dict[str, List[float]]) -> pd.DataFrame:
    """Отсекает выбросы по квантильным границам из обучения."""
    for col, (lower, upper) in clip_bounds.items():
        if col in df.columns:
            df[col] = df[col].clip(lower=lower, upper=upper)
    logger.debug('clip_numbers: колонок %d', len(clip_bounds))
    return df


def lookup_personal_recommendation(personal_recs: Optional[pd.DataFrame],
                                  ncodpers: Any) -> tuple[Any, bool]:
    """Персональная ALS-рекомендация по ID клиента; (значение, найден ли клиент)."""
    if personal_recs is not None:
        try:
            recommendation = personal_recs['recommended_product_id'].get(ncodpers)
        except (KeyError, TypeError):  # pragma: no cover — защита от неожиданного формата файла
            logger.exception('Ошибка поиска ALS-рекомендации для ncodpers=%s', ncodpers)
            return 0, False
        if isinstance(recommendation, pd.Series):
            # дубли ncodpers (load_personal_recs их вычищает, но файл могли
            # прочитать напрямую) — берём первую рекомендацию
            recommendation = recommendation.iloc[0] if len(recommendation) else None
        if recommendation is None or pd.isna(recommendation):
            return 0, False
        return recommendation, True
    return 0, False


def add_personal_recommendation(df: pd.DataFrame,
                               personal_recs: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Добавляет признак `recommended_product_id` из ALS-модели (по ID клиента)."""
    ncodpers = df['ncodpers'].iloc[0]
    recommendation, found = lookup_personal_recommendation(personal_recs, ncodpers)
    if not found:
        logger.info('Персональная ALS-рекомендация для ncodpers=%s не найдена — 0', ncodpers)
    else:
        logger.debug('Персональная ALS-рекомендация для ncodpers=%s: %s',
                     ncodpers, recommendation)
    df['recommended_product_id'] = recommendation
    return df


def add_total_products(df: pd.DataFrame, products: Optional[List[str]] = None) -> pd.DataFrame:
    """Число продуктов клиента — сумма флагов владения."""
    columns = PRODUCTS if products is None else products
    if all(pd.api.types.is_numeric_dtype(df[col]) for col in columns):
        # типичный путь API: pydantic приводит флаги к float — to_numeric по ним
        # тождествен, писать назад нечего (экономит ~7 мс на запросе)
        df['total_products'] = df[columns].sum(axis=1)
        return df
    converted = df[columns].apply(pd.to_numeric, errors='coerce')
    df[columns] = converted
    df['total_products'] = converted.sum(axis=1)
    logger.debug('add_total_products: продуктовых колонок %d', len(columns))
    return df


def manual_transformations(df: pd.DataFrame,
                           aggregates: Optional[Dict[str, Dict[str, float]]] = None
                           ) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Ручные и агрегатные признаки.

    На обучении (`aggregates=None`) групповые статистики считаются по
    переданному train-фрейму и возвращаются для сериализации в
    `preprocessing_params.json`. На проде сохранённые агрегаты применяются
    через `map` — иначе статистики считались бы по одной строке запроса.
    Если переданных агрегатов не хватает (резервный режим без артефакта),
    недостающие считаются по входному фрейму — для одной строки это
    вырожденный, но рабочий режим.

    В каждый словарь агрегатов при обучении добавляется ключ '__default__' —
    глобальная статистика по train-фрейму. При apply категория, которой не
    было при обучении, маппится в '__default__', а не в NaN: иначе тестовый
    (или продовый) клиент с невиданной категорией получал бы NaN в числовых
    признаках, на которых RandomForest падает.

    Новые колонки добавляются одним `concat` (порядок ключей = прежний порядок
    присваиваний, чтобы порядок колонок не изменился): поколоночные
    присваивания на однострочном фрейме запроса стоят ~1 мс каждое.
    """
    stored = dict(aggregates) if aggregates else {}
    extras: Dict[str, pd.Series] = {}
    extras['renta_antiguedad_ratio'] = df['renta'] / (df['antiguedad'] + 1)
    extras['log_renta'] = np.log1p(df['renta'])

    mode = 'apply' if aggregates else 'fit'
    logger.debug('manual_transformations[%s]: строк %d', mode, len(df))

    for col in ('pais_residencia', 'segmento'):
        renta_key = f'mean_renta_by_{col}'
        antiguedad_key = f'median_antiguedad_by_{col}'
        if stored.get(renta_key) and stored.get(antiguedad_key):
            renta_default = stored[renta_key].get('__default__')
            if renta_default is None:  # агрегаты от сохранённого запуска без fallback
                renta_default = float(np.mean(list(stored[renta_key].values())))
            antiguedad_default = stored[antiguedad_key].get('__default__')
            if antiguedad_default is None:
                antiguedad_default = float(np.mean(list(stored[antiguedad_key].values())))
            extras[renta_key] = (df[col].map(stored[renta_key])
                                 .astype(float).fillna(renta_default))
            extras[antiguedad_key] = (df[col].map(stored[antiguedad_key])
                                      .astype(float).fillna(antiguedad_default))
        else:
            if aggregates is not None:
                logger.warning('Агрегаты для %s не найдены в параметрах — считаю по входным '
                               'данным', col)
            # astype(float): featuretools отдает категориальные колонки как category,
            # а map по ним сохраняет category-dtype и ломает арифметику ниже
            stored[renta_key] = {
                str(cat): float(val) for cat, val in df.groupby(col)['renta'].mean().items()}
            stored[antiguedad_key] = {
                str(cat): float(val)
                for cat, val in df.groupby(col)['antiguedad'].median().items()}
            # fallback для категорий, которых не было при обучении:
            # глобальная статистика колонки по train-фрейму
            stored[renta_key]['__default__'] = float(df['renta'].mean())
            stored[antiguedad_key]['__default__'] = float(df['antiguedad'].median())
            extras[renta_key] = (df[col].map(stored[renta_key])
                                 .astype(float)
                                 .fillna(stored[renta_key]['__default__']))
            extras[antiguedad_key] = (
                df[col].map(stored[antiguedad_key])
                .astype(float)
                .fillna(stored[antiguedad_key]['__default__']))
            logger.info('manual_transformations[fit]: посчитаны агрегаты по %s '
                        '(%d категорий)', col, len(stored[renta_key]) - 1)

    extras['renta_vs_country_mean'] = df['renta'] / extras['mean_renta_by_pais_residencia']
    df = pd.concat([df, pd.DataFrame(extras, index=df.index)], axis=1)
    return df, stored


def _sanitize_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Защита от inf/NaN в производных признаках.

    Арифметика DFS порождает inf (деление на 0, log(0) = -inf) и NaN
    (log/sqrt от отрицательных значений, 0/0). StandardScaler в
    ColumnTransformer не принимает inf, RandomForest — NaN.
    Заменяем inf на NaN и заполняем числовые признаки нулём:
    детерминированно, без сохранения состояния, поэтому train/serve
    согласованы по построению.

    Возвращает тот же фрейм (in-place по числовым колонкам), чтобы не
    копировать многомиллионный train ещё раз.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    before_inf = int(np.isinf(df[numeric_cols].to_numpy()).sum())
    before_nan = int(df[numeric_cols].isna().to_numpy().sum())
    if before_inf or before_nan:
        df[numeric_cols] = df[numeric_cols].replace([np.inf, -np.inf], np.nan)
        df[numeric_cols] = df[numeric_cols].fillna(0.0)
        logger.info('_sanitize_derived_features: inf=%d, NaN=%d в числовых колонках '
                    'заменены (inf→NaN→0.0)', before_inf, before_nan)
    return df


def feature_engineering(df: pd.DataFrame,
                        aggregates: Optional[Dict[str, Dict[str, float]]] = None,
                        *, cast_categories: bool = True,
                        ) -> tuple[pd.DataFrame, Dict[str, Dict[str, float]]]:
    """Генерирует производные признаки (арифметика + агрегаты).

    Возвращает (матрица признаков, агрегаты): на обучении агрегаты уезжают
    в `preprocessing_params.json`, на проде — приходят оттуда же.

    Арифметические признаки считаются напрямую pandas/numpy: это быстрее и
    надёжнее для единичных запросов сервиса, чем строить featuretools EntitySet
    и запускать DFS на каждый профиль. Значения воспроизводят семантику DFS
    побитово (проверяется тестом на паритет с featuretools):
    те же имена колонок, чистые numpy-операции (log(0) → -inf, log(x<0)/sqrt(x<0)
    → NaN, деление на ноль → inf) и приведение нечисловых колонок к category —
    так делал woodwork, и на этом держится авто-детект числовых/категориальных
    признаков в modeling.ipynb.

    `cast_categories=False` пропускает приведение к category: оно нужно только
    обучению (dtype для авто-детекта в modeling.ipynb), а `prepare_features`
    на инференсе всё равно приводит признаки к str/numeric сразу после.

    Дополнительно в конце вызывается `_sanitize_derived_features`: инф/NaN
    из log/sqrt/деления не должны дойти до `StandardScaler` и
    `RandomForestClassifier` — они падают на таких значениях.
    """
    mode = 'fit' if aggregates is None else 'apply'
    logger.info('feature_engineering[%s]: вход %s, категориальные → category=%s',
                mode, _shape(df), cast_categories)

    # Колонки, которые DFS (woodwork Double) оставлял числовыми; остальные
    # приводились к Categorical — сохраняем, чтобы dtype после этой функции
    # не изменился по сравнению с featuretools-версией.
    numeric_columns = {
        'antiguedad', 'renta',
        'antiguedad + renta', 'antiguedad / renta', 'renta / antiguedad',
        'antiguedad * renta', 'NATURAL_LOGARITHM(antiguedad)',
        'NATURAL_LOGARITHM(renta)', 'SQUARE_ROOT(antiguedad)', 'SQUARE_ROOT(renta)',
    }
    if cast_categories:
        # Прогресс-бар полезен: на train это ~50 колонок × миллионы строк,
        # каждая astype('category') может занимать заметное время.
        disable = len(df) < PROGRESS_MIN_ROWS
        for col in _progress(list(df.columns), desc='feature_engineering: cast → category',
                             disable=disable):
            if col not in numeric_columns:
                df[col] = df[col].astype('category')

    # woodwork Double приводил обе колонки к float64 в выходной матрице — повторяем
    df['antiguedad'] = antiguedad = df['antiguedad'].astype(float)
    df['renta'] = renta = df['renta'].astype(float)
    with np.errstate(all='ignore'):  # те же inf/NaN, что давал DFS, но без RuntimeWarning в лог
        transforms = pd.DataFrame({
            'antiguedad + renta': antiguedad.to_numpy() + renta.to_numpy(),
            'antiguedad / renta': antiguedad.to_numpy() / renta.to_numpy(),
            'renta / antiguedad': renta.to_numpy() / antiguedad.to_numpy(),
            'antiguedad * renta': antiguedad.to_numpy() * renta.to_numpy(),
            'NATURAL_LOGARITHM(antiguedad)': np.log(antiguedad.to_numpy()),
            'NATURAL_LOGARITHM(renta)': np.log(renta.to_numpy()),
            'SQUARE_ROOT(antiguedad)': np.sqrt(antiguedad.to_numpy()),
            'SQUARE_ROOT(renta)': np.sqrt(renta.to_numpy()),
        }, index=df.index)
    # один concat вместо 8 присваиваний (порядок ключей = порядок колонок DFS)
    df = pd.concat([df, transforms], axis=1)
    logger.debug('feature_engineering[%s]: добавлено арифметических признаков: %d',
                 mode, len(transforms.columns))

    df = df.drop(columns=['unique_id', 'index'], errors='ignore')
    # DFS с make_index всегда возвращал свежий RangeIndex 0..n-1 — повторяем,
    # чтобы результат не зависел от индекса входного фрейма
    df = df.reset_index(drop=True)
    df, aggregates_out = manual_transformations(df, aggregates)
    # inf/NaN из log(0), 0/0, sqrt(neg) ломают StandardScaler в ColumnTransformer
    # и RandomForestClassifier — заменяем на конечные значения здесь.
    df = _sanitize_derived_features(df)
    # Удаляем дубликаты колонок после объединения производных признаков
    df = df.loc[:, ~df.columns.duplicated()]
    logger.info('feature_engineering[%s]: выход %s, колонок модели (numeric=%d, cat=%d)',
                mode, _shape(df), len(NUMERIC_FEATURES), len(CATEGORICAL_FEATURES))
    return df, aggregates_out


# --- Обучение параметров предобработки (только на train) ----------------------
# Производные/служебные колонки не участвуют в сворачивании редких категорий
# и расчёте мод: они создаются отдельными шагами после базовой очистки.
_DERIVED_COLUMNS = frozenset({'age_interval', 'total_products', 'recommended_product_id'})

# Числовые колонки с медианами и квантильным клиппингом.
_MEDIAN_COLUMNS = ('age', 'antiguedad', 'renta')

# Идентификатор и дата среза: не категории и не медианные числовые.
_MODES_EXCLUDED = frozenset({'age', 'antiguedad', 'fecha_dato', 'ncodpers', 'renta'})


def calculate_age_intervals(df: pd.DataFrame, n_intervals: int = 3) -> list:
    """Возрастные корзины по кумулятивному распределению продуктов.

    Перенесено из modeling.ipynb без изменений: границы подбираются так,
    чтобы в каждый интервал попадала ~1/n_intervals суммарных продуктов.
    Ожидает колонки `age` и `total_products`.
    """
    age_groups = df.groupby('age', observed=True)['total_products'].sum().reset_index()
    age_groups = age_groups.sort_values('age')

    cum_products = np.cumsum(age_groups['total_products'].values)
    total = cum_products[-1]
    step = total / n_intervals

    targets = np.arange(1, n_intervals) * step
    idxs = np.searchsorted(cum_products, targets, side='right')
    upper_ages = age_groups['age'].values[np.minimum(idxs, len(age_groups) - 1)]

    intervals = []
    lower = age_groups['age'].iloc[0]
    for upper in upper_ages:
        intervals.append((lower, upper))
        lower = upper + 1
    intervals.append((lower, age_groups['age'].iloc[-1]))
    logger.info('calculate_age_intervals: n_intervals=%d, границы: %s',
                n_intervals, intervals)
    return intervals


def create_time_cohorts(series: pd.Series, max_cohorts: int = 30) -> tuple:
    """Возвращает (метки когорт, спецификация когорт для сервиса).

    Спецификация описывает границы когорт в днях от base_date, чтобы сервис
    применял ровно те же интервалы, что и обучение: интервал i покрывает
    (left_days, right_days], где left_days — правый день предыдущего бина,
    right_days — максимальный день бина.
    """
    series_dt = pd.to_datetime(series)
    base_date = series_dt.min().normalize()

    # Вычисляем дни относительно базовой даты
    days = (series_dt.dt.normalize() - base_date).dt.days

    # Определяем оптимальное количество когорт
    unique_days = days.unique()
    n_cohorts = min(max_cohorts, len(unique_days))

    # Создаем интервалы через qcut (быстрее чем apply)
    try:
        qbins = pd.qcut(days, q=n_cohorts, precision=0, duplicates='drop')
    except ValueError:
        logger.warning('create_time_cohorts: qcut не удался (n_cohorts=%d) — '
                       'режим raw', n_cohorts)
        return series_dt.dt.strftime('%Y-%m-%d').astype('str'), {'mode': 'raw'}

    # Генерируем строковые метки для всех интервалов сразу
    interval_labels = [
        f"{(base_date + pd.Timedelta(days=int(iv.left))).strftime('%Y-%m-%d')}"
        f" – {(base_date + pd.Timedelta(days=int(iv.right))).strftime('%Y-%m-%d')}"
        for iv in qbins.cat.categories
    ]

    # Маппинг категорий на строки через векторные операции
    label_map = dict(zip(qbins.cat.categories, interval_labels))
    cohort_labels = qbins.cat.rename_categories(label_map).astype('str')

    # Границы для сервиса берем по фактическому распределению дней по бинам:
    # так спецификация воспроизводит обучающие метки один в один
    codes = qbins.cat.codes.to_numpy()
    days_np = days.to_numpy()
    rights = [int(days_np[codes == i].max()) for i in range(len(interval_labels))]
    lefts = [int(days_np[codes == 0].min()) - 1] + rights[:-1]
    missing_label = str(cohort_labels[days.isna()].iloc[0]) if days.isna().any() else 'unknown'
    spec = {
        'mode': 'intervals',
        'base_date': base_date.strftime('%Y-%m-%d'),
        'closed': 'right',
        'missing_label': missing_label,
        'intervals': [
            {'left_days': left, 'right_days': right, 'label': label}
            for left, right, label in zip(lefts, rights, interval_labels)
        ],
    }
    logger.info('create_time_cohorts: base_date=%s, когорт=%d, missing_label=%s',
                spec['base_date'], len(spec['intervals']), missing_label)
    return cohort_labels, spec


def count_cohort_mismatches(raw_dates: pd.Series, cohorts: pd.Series,
                            spec: Dict[str, Any]) -> int:
    """Сколько обучающих меток не воспроизводит сервисная спецификация когорт.

    Векторная проверка из modeling.ipynb: интервалы отсортированы и не
    пересекаются, поэтому сводится к одному pd.cut вместо построчного map.
    """
    base_date = pd.Timestamp(spec['base_date']).normalize()
    parsed = pd.to_datetime(raw_dates, errors='coerce')
    days = (parsed.dt.normalize() - base_date).dt.days

    bins = [spec['intervals'][0]['left_days']] + [iv['right_days'] for iv in spec['intervals']]
    labels = [iv['label'] for iv in spec['intervals']]

    restored = pd.cut(days, bins=bins, labels=labels, right=True).astype(object)
    restored[days.isna()] = spec.get('missing_label', 'unknown')
    return int((restored.astype(str).to_numpy() != cohorts.astype(str).to_numpy()).sum())


def compute_drift_reference(df: pd.DataFrame, columns=_MEDIAN_COLUMNS,
                            n_bins: int = 10) -> Dict[str, Any]:
    """Эталонные гистограммы входных числовых признаков для дрейф-мониторинга.

    Децили распределения train: сервис (`drift.DriftMonitor`) считает PSI
    скользящего окна запросов против этих бинов. Пропуски и сентинель стажа
    -999999 игнорируются (ожидается фрейм после coerce_numeric).
    """
    reference: Dict[str, Any] = {}
    disable = len(df) < PROGRESS_MIN_ROWS
    for col in _progress(list(columns), desc='compute_drift_reference', disable=disable):
        if col not in df.columns:
            continue
        values = pd.to_numeric(df[col], errors='coerce')
        values = values.replace(np.float64(-999999.0), np.nan).dropna()
        if values.nunique() < 2:
            logger.debug('compute_drift_reference: %s пропущен (уникальных < 2)', col)
            continue
        edges = np.unique(
            values.quantile(np.linspace(0.0, 1.0, n_bins + 1)).to_numpy(dtype=float))
        if len(edges) < 3:  # гистограмма из 1-2 бинов неинформативна
            logger.debug('compute_drift_reference: %s пропущен (бинов < 3)', col)
            continue
        edges[0], edges[-1] = -np.inf, np.inf
        counts, _ = np.histogram(values.to_numpy(dtype=float), bins=edges)
        reference[col] = {
            'edges': [float(edge) for edge in edges],
            'shares': (counts / counts.sum()).tolist(),
        }
        logger.debug('compute_drift_reference: %s — бинов %d, значений %d',
                     col, len(edges) - 1, int(counts.sum()))
    logger.info('compute_drift_reference: колонок в эталоне: %d', len(reference))
    return reference


def fit_preprocessing_params(fit_df: pd.DataFrame, *, rare_threshold: float = 0.01,
                             n_age_intervals: int = 9, max_cohorts: int = 12,
                             clip_quantiles: tuple = (0.01, 0.96),
                             drift_columns=_MEDIAN_COLUMNS,
                             drift_bins: int = 10) -> PreprocessingParams:
    """Обучает статистики предобработки на train-части (без утечки из теста).

    Медианы, моды, квантили клиппинга, наборы редких категорий, когорты дат
    и возрастные бины считаются только на train. Test и продовые профили
    трансформируются `apply_preprocessing` с теми же параметрами — так сохраняется
    train/serve-согласованность.

    Колонку таргета (`purchase`) функция удаляет, если она есть во фрейме:
    таргет не должен влиять на статистики. Агрегаты (`mean_renta_by_*` и др.)
    здесь не считаются — они обучаются в `feature_engineering` на train-части.
    """
    logger.info('fit_preprocessing_params: старт, train %s '
                '(rare_threshold=%.3f, n_age_intervals=%d, max_cohorts=%d, '
                'clip_quantiles=%s, drift_bins=%d)',
                _shape(fit_df), rare_threshold, n_age_intervals, max_cohorts,
                clip_quantiles, drift_bins)
    overall_start = time.perf_counter()

    with _timed('fit_preprocessing_params: coerce_numeric'):
        df = coerce_numeric(fit_df.copy())
    df = df.drop(columns=['purchase'], errors='ignore')

    with _timed('fit_preprocessing_params: медианы и моды'):
        medians = {col: float(df[col].median()) for col in _MEDIAN_COLUMNS if col in df.columns}
        modes_columns = [col for col in df.columns
                         if col not in _MODES_EXCLUDED and col not in _DERIVED_COLUMNS]
        modes = {}
        disable = len(df) < PROGRESS_MIN_ROWS
        for col in _progress(modes_columns, desc='fit: modes', disable=disable):
            mode = df[col].mode(dropna=True)
            if not mode.empty:
                modes[col] = mode.iloc[0]
    logger.info('fit_preprocessing_params: медианы=%s, мод посчитано=%d',
                medians, len(modes))

    # Порядок повторяет обучающий и продовый пайплайн: fillna → бины
    # возраста → когорты дат → сворачивание редких категорий → клиппинг.
    with _timed('fit_preprocessing_params: fill_missing'):
        work = fill_missing(df, medians, modes)

    products_present = [product for product in PRODUCTS if product in work.columns]
    if not products_present or 'age' not in work.columns:
        raise ValueError('для возрастных интервалов нужны колонки age и продуктовые флаги')

    with _timed('fit_preprocessing_params: age_intervals'):
        work = add_total_products(work, products_present)
        age_intervals = calculate_age_intervals(work, n_age_intervals)
        work = add_age_interval(work, age_intervals)

    with _timed('fit_preprocessing_params: date_cohorts'):
        date_cohorts = {}
        for col in ('fecha_alta', 'ult_fec_cli_1t'):
            if col not in work.columns:
                continue
            raw_dates = work[col].copy()  # моды уже подставлены — как при обучении
            cohorts, spec = create_time_cohorts(work[col], max_cohorts=max_cohorts)
            work[col] = cohorts.to_numpy()
            date_cohorts[col] = spec
            if spec.get('mode') == 'intervals':
                mismatches = count_cohort_mismatches(raw_dates, cohorts, spec)
                if mismatches:
                    logger.warning('Спецификация когорт %s не воспроизводит %s значений '
                                   'обучения', col, mismatches)

    with _timed('fit_preprocessing_params: replacer (редкие категории)'):
        # Редкие категории: как при обучении — частоты считаются в строковом
        # домене (astype(str)), продуктовые флаги не сворачиваются.
        replacer_columns = [col for col in modes_columns if col not in PRODUCTS]
        replacer = {}
        disable = len(work) < PROGRESS_MIN_ROWS
        for col in _progress(replacer_columns, desc='fit: rare→other', disable=disable):
            as_str = work[col].astype(str)
            counts = as_str.value_counts(normalize=True)
            rare_values = counts[counts < rare_threshold].index
            replacer[col] = {str(value): 'other' for value in rare_values}
        n_replacer = sum(1 for values in replacer.values() if values)
    logger.info('fit_preprocessing_params: колонок с редкими категориями: %d из %d',
                n_replacer, len(replacer_columns))

    with _timed('fit_preprocessing_params: clip_bounds'):
        clip_bounds = {}
        lower_q, upper_q = clip_quantiles
        for col in ('renta', 'antiguedad'):
            if col in work.columns:
                clip_bounds[col] = [float(work[col].quantile(lower_q)),
                                    float(work[col].quantile(upper_q))]
    logger.info('fit_preprocessing_params: clip_bounds=%s', clip_bounds)

    with _timed('fit_preprocessing_params: drift_reference'):
        drift_reference = compute_drift_reference(df, drift_columns, drift_bins)

    logger.info('fit_preprocessing_params: готово за %.2f с',
                time.perf_counter() - overall_start)

    return PreprocessingParams(
        medians=medians,
        modes=modes,
        clip_bounds=clip_bounds,
        age_intervals=[[int(lower), int(upper)] for lower, upper in age_intervals],
        date_cohorts=date_cohorts,
        replacer=replacer,
        drift_reference=drift_reference,
        source='fit(train)',
    )


def apply_preprocessing(df: pd.DataFrame, params: PreprocessingParams) -> pd.DataFrame:
    """Трансформирует батч профилей параметрами обучающего запуска.

    Тот же кодовый путь и порядок шагов, что в `prepare_features` на проде
    (кроме ALS-рекомендации: здесь колонка `recommended_product_id` ожидается
    уже подмёргнутой, а на проде её добавляет `add_personal_recommendation`).
    """
    logger.info('apply_preprocessing: вход %s, источник параметров: %s',
                _shape(df), params.source)
    overall_start = time.perf_counter()
    df = coerce_numeric(df)
    df = fill_missing(df, params.medians, params.modes)
    if 'age' in df.columns:
        df = add_age_interval(df, params.age_intervals)
    df = add_date_cohorts(df, params.date_cohorts, params.date_intervals)
    df = fold_rare_categories(df, params.replacer)
    df = clip_numbers(df, params.clip_bounds)
    if all(product in df.columns for product in PRODUCTS):
        df = add_total_products(df)
    logger.info('apply_preprocessing: выход %s за %.2f с',
                _shape(df), time.perf_counter() - overall_start)
    return df


def temporal_split(dates: pd.Series, test_size: float = 0.3) -> tuple:
    """Временное разбиение по датам: train — значения ≤ cutoff, test — позже.

    Возвращает (cutoff, train_mask, test_mask). modeling.ipynb разбивает по дате
    привлечения клиента (`fecha_alta`): train — старые клиенты, test — новые.
    По дате среза (`fecha_dato` последнего наблюдения) разбивать нельзя: посадив
    в train клиентов с ранней последней датой, мы получили бы почти одних
    ушедших из банка клиентов с таргетом 0 (проверено экспериментально).
    NaT уезжает в train. В отличие от случайного сплита здесь нет
    «подглядывания в будущее», и профили одного клиента не смешиваются.
    """
    normalized = pd.to_datetime(pd.Series(dates)).dt.normalize()
    unique_dates = np.sort(normalized.dropna().unique())
    if len(unique_dates) < 2:
        raise ValueError('временное разбиение невозможно: '
                         'нужно минимум 2 различные даты среза')
    cutoff = pd.Timestamp(normalized.quantile(1 - test_size)).normalize()
    test_mask = normalized > cutoff
    if not test_mask.any() or test_mask.all():
        # Распределение дат вырождено (квантиль попал на край): берём
        # ближайшую реальную дату, чтобы обе части были непусты
        index = int(round((1 - test_size) * (len(unique_dates) - 1)))
        index = min(max(index, 0), len(unique_dates) - 2)
        cutoff = pd.Timestamp(unique_dates[index])
        test_mask = normalized > cutoff
    logger.info('temporal_split: строк %d, cutoff=%s, train=%d, test=%d',
                len(normalized), cutoff.date(), int((~test_mask).sum()), int(test_mask.sum()))
    return cutoff, ~test_mask, test_mask


def prepare_features(profile: Dict[str, Any], params: PreprocessingParams,
                     personal_recs: Optional[pd.DataFrame] = None) -> pd.DataFrame:
    """Превращает профиль клиента в матрицу признаков той же формы, что и на обучении.

    Порядок шагов повторяет обучение: сначала fillna медиан/мод, и только
    потом — возрастные бины и когорты дат. Базовые
    трансформации общие для прода и обучения — `apply_preprocessing`.
    Бросает ValueError, если в профиле нет обязательных колонок.
    """
    logger.debug('prepare_features: старт (params.source=%s)', params.source)
    row = pd.DataFrame([profile])
    row = row.drop(columns=list(EXCLUDED_COLUMNS), errors='ignore')

    # Сырые колонки, без которых признаки не построить: базовые минус создаваемые
    # (`total_products`, `age_interval`, `recommended_product_id`) плюс сырьё для них.
    required_inputs = [col for col in BASE_COLUMNS
                       if col not in ('total_products', 'age_interval', 'recommended_product_id')]
    required_inputs += ['ncodpers', 'age', 'fecha_alta', 'ult_fec_cli_1t']
    missing_inputs = [col for col in required_inputs if col not in row.columns]
    if missing_inputs:
        raise ValueError(f'В профиле не хватает обязательных полей: {missing_inputs}')

    row = apply_preprocessing(row, params)
    row = add_personal_recommendation(row, personal_recs)
    row = row[BASE_COLUMNS]

    row['antiguedad'] = row['antiguedad'].astype(int)
    # cast_categories=False: на инференсе dtypes задаёт цикл ниже, а ~45
    # приведений к category только добавляли латентности
    row, _ = feature_engineering(row, params.aggregates or None, cast_categories=False)

    # Списки числовых/категориальных признаков — из артефакта этого же запуска
    # (автодетект в modeling.ipynb зависит от данных: число уникальных значений
    # числовой колонки может упасть ниже 25, и она станет категориальной).
    # Списки берутся из артефакта этого же запуска; если их нет, используем
    # константы модуля как резерв.
    feature_columns = params.feature_columns or {}
    numeric_features = feature_columns.get('numeric') or NUMERIC_FEATURES
    categorical_features = feature_columns.get('categorical') or CATEGORICAL_FEATURES

    # Приведение признаков к итоговым типам — батчем, а не по колонке
    # (каждое поколоночное присваивание в pandas копирует блоки данных).
    # Числовые, уже приведённые к float на предыдущих шагах, не трогаем —
    # to_numeric по ним тождественен.
    numeric_needs_cast = [
        col for col in numeric_features
        if col in row.columns and not pd.api.types.is_numeric_dtype(row[col])
    ]
    if numeric_needs_cast:
        row[numeric_needs_cast] = row[numeric_needs_cast].apply(pd.to_numeric, errors='coerce')
    categorical_present = [col for col in categorical_features if col in row.columns]
    if categorical_present:
        row[categorical_present] = row[categorical_present].astype('str')

    expected = set(numeric_features) | set(categorical_features)
    absent = sorted(expected - set(row.columns))
    if absent:
        logger.error('В матрице признаков нет колонок модели: %s', absent)
    logger.debug('prepare_features: готово, форма %s', _shape(row))
    return row