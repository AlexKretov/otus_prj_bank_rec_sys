"""Unit-тесты общего модуля предобработки (CODE_REVIEW §P2.16).

Чистые функции тестируются без тяжёлых зависимостей; сквозной
`prepare_features` требует featuretools/woodwork (есть в requirements.txt).
"""

from importlib.util import find_spec

import numpy as np
import pandas as pd
import pytest

import preprocessing
from preprocessing import (
    BASE_COLUMNS,
    CATEGORICAL_FEATURES,
    LEGACY_AGE_INTERVALS,
    LEGACY_CLIP_BOUNDS,
    LEGACY_DATE_INTERVALS,
    LEGACY_MEDIANS,
    LEGACY_MODES,
    NUMERIC_FEATURES,
    PRODUCTS,
    PreprocessingParams,
)
from tests.conftest import make_profile

pytestmark = pytest.mark.filterwarnings('ignore::FutureWarning')


def test_find_interval_matches_legacy_bounds():
    intervals = LEGACY_DATE_INTERVALS['fecha_alta']
    assert preprocessing.find_interval('2010-05-17', intervals) is not None
    assert preprocessing.find_interval('1900-01-01', intervals) is None  # вне всех интервалов
    assert preprocessing.find_interval(None, intervals) is None
    assert preprocessing.find_interval(float('nan'), intervals) is None
    assert preprocessing.find_interval('not-a-date', intervals) is None


def test_map_date_to_cohort_interval_semantics():
    """Полуоткрытые интервалы (left, right], как у pd.qcut при обучении."""
    spec = {
        'mode': 'intervals',
        'base_date': '2020-01-01',
        'missing_label': 'nan',
        'intervals': [
            {'left_days': -1, 'right_days': 10, 'label': 'first'},
            {'left_days': 10, 'right_days': 20, 'label': 'second'},
        ],
    }
    assert preprocessing.map_date_to_cohort('2020-01-01', spec) == 'first'  # день 0
    assert preprocessing.map_date_to_cohort('2020-01-11', spec) == 'first'  # день 10: правый край
    assert preprocessing.map_date_to_cohort('2020-01-12', spec) == 'second'  # день 11
    assert preprocessing.map_date_to_cohort('2021-06-01', spec) == 'unknown'  # вне интервалов
    assert preprocessing.map_date_to_cohort(None, spec) == 'nan'  # пропуск → метка обучения


def test_map_date_to_cohort_raw_mode():
    spec = {'mode': 'raw'}
    assert preprocessing.map_date_to_cohort('2020-03-05', spec) == '2020-03-05'
    assert preprocessing.map_date_to_cohort(None, spec) == 'unknown'


def test_add_age_interval_clips_out_of_range():
    """Возраст вне обучающих бинов — в крайний интервал, а не в NaN."""
    df = pd.DataFrame({'age': [1.0, 45.0, 200.0]})
    out = preprocessing.add_age_interval(df, LEGACY_AGE_INTERVALS)
    assert out['age_interval'].isna().sum() == 0
    labels = [f'{lo}-{hi}' for lo, hi in LEGACY_AGE_INTERVALS]
    assert list(out['age_interval'].astype(str)) == [labels[0], '42-45', labels[-1]]


def test_fold_rare_categories_keeps_frequent_values():
    """Регрессия CODE_REVIEW §1.4: частотные значения не превращаются в NaN."""
    df = pd.DataFrame({'pais_residencia': ['ES', 'ES', 'XX'], 'sexo': ['V', 'H', 'V']})
    replacer = {'pais_residencia': {'XX': 'other'}}
    out = preprocessing.fold_rare_categories(df, replacer)
    assert list(out['pais_residencia']) == ['ES', 'ES', 'other']
    assert list(out['sexo']) == ['V', 'H', 'V']  # колонки без словаря не трогаем


def test_coerce_numeric_handles_antiguedad_sentinel():
    df = pd.DataFrame({'age': ['bad', 30], 'antiguedad': [-999999.0, 10], 'renta': [100, 200]})
    out = preprocessing.coerce_numeric(df)
    assert pd.isna(out['age'].iloc[0])
    assert pd.isna(out['antiguedad'].iloc[0])
    assert out['antiguedad'].iloc[1] == 10


def test_fill_missing_uses_medians_and_modes():
    df = pd.DataFrame({'age': [np.nan], 'antiguedad': [5.0], 'renta': [np.nan],
                       'sexo': [None], 'segmento': ['X']})
    out = preprocessing.fill_missing(df, LEGACY_MEDIANS, LEGACY_MODES)
    assert out['age'].iloc[0] == LEGACY_MEDIANS['age']
    assert out['renta'].iloc[0] == LEGACY_MEDIANS['renta']
    assert out['sexo'].iloc[0] == LEGACY_MODES['sexo']
    assert out['segmento'].iloc[0] == 'X'


def test_fill_missing_before_bins_leaves_no_nans():
    """Регрессия CODE_REVIEW §2.1: fillna — до расчёта бинов."""
    df = pd.DataFrame({'age': [np.nan], 'antiguedad': [np.nan], 'renta': [np.nan]})
    df = preprocessing.fill_missing(df, LEGACY_MEDIANS, LEGACY_MODES)
    df = preprocessing.add_age_interval(df, LEGACY_AGE_INTERVALS)
    assert df['age_interval'].isna().sum() == 0


def test_clip_numbers_applies_training_bounds():
    df = pd.DataFrame({'renta': [0.0, 1e9], 'antiguedad': [-5.0, 1e6]})
    out = preprocessing.clip_numbers(df, LEGACY_CLIP_BOUNDS)
    assert out['renta'].tolist() == LEGACY_CLIP_BOUNDS['renta']
    assert out['antiguedad'].tolist() == LEGACY_CLIP_BOUNDS['antiguedad']


def test_lookup_personal_recommendation_hit_miss_nan():
    recs = pd.DataFrame(
        {'recommended_product_id': ['ind_tjcr_fin_ult1', None]},
        index=pd.Index([1001, 1002], name='ncodpers'),
    )
    assert preprocessing.lookup_personal_recommendation(recs, 1001) == ('ind_tjcr_fin_ult1', True)
    assert preprocessing.lookup_personal_recommendation(recs, 1002) == (0, False)  # NaN → 0
    assert preprocessing.lookup_personal_recommendation(recs, 999999) == (0, False)  # мимо → 0
    assert preprocessing.lookup_personal_recommendation(None, 1001) == (0, False)  # файла нет → 0


def test_lookup_personal_recommendation_survives_duplicates():
    """Дубли ncodpers без дедупа: .get возвращает Series — берём первую."""
    recs = pd.DataFrame(
        {'recommended_product_id': ['ind_tjcr_fin_ult1', 'ind_cco_fin_ult1']},
        index=pd.Index([1001, 1001], name='ncodpers'),
    )
    value, found = preprocessing.lookup_personal_recommendation(recs, 1001)
    assert found and value == 'ind_tjcr_fin_ult1'


def test_load_personal_recs_prefers_non_null_and_dedupes(tmp_path):
    """При дедупе оставляем непустую рекомендацию, а не первую строку."""
    recs = pd.DataFrame(
        {'ncodpers': [7, 7, 8], 'recommended_product_id': [None, 'ind_cco_fin_ult1', None]})
    path = tmp_path / 'personal_als.parquet'
    recs.to_parquet(path)
    loaded = preprocessing.load_personal_recs(path)
    assert loaded.index.name == 'ncodpers'
    assert len(loaded) == 2
    assert loaded.loc[7, 'recommended_product_id'] == 'ind_cco_fin_ult1'


def test_load_personal_recs_missing_file_returns_none(tmp_path):
    assert preprocessing.load_personal_recs(tmp_path / 'absent.parquet') is None


def test_manual_transformations_train_serve_consistency():
    """Агрегаты, посчитанные на фрейме, применяются к одной строке один в один."""
    frame = pd.DataFrame({
        'pais_residencia': ['ES', 'ES', 'FR'],
        'segmento': ['A', 'B', 'A'],
        'renta': [100.0, 200.0, 300.0],
        'antiguedad': [10.0, 20.0, 30.0],
    })
    trained, aggregates = preprocessing.manual_transformations(frame.copy(), None)
    assert set(aggregates) == {'mean_renta_by_pais_residencia', 'mean_renta_by_segmento',
                               'median_antiguedad_by_pais_residencia',
                               'median_antiguedad_by_segmento'}
    assert aggregates['mean_renta_by_pais_residencia'] == {'ES': 150.0, 'FR': 300.0}

    single = frame.iloc[[0]].copy()
    served, _ = preprocessing.manual_transformations(single, aggregates)
    for col in aggregates:
        assert served[col].iloc[0] == trained[col].iloc[0]
    assert served['renta_vs_country_mean'].iloc[0] == pytest.approx(100.0 / 150.0)


def test_add_total_products_sums_flags():
    df = pd.DataFrame({product: [0, 1] for product in PRODUCTS})
    df['ind_cco_fin_ult1'] = [1, 1]
    out = preprocessing.add_total_products(df)
    assert out['total_products'].tolist() == [1, 24]


def test_preprocessing_params_legacy_fallback(tmp_path):
    params = PreprocessingParams.from_json(tmp_path / 'absent.json')
    assert params.source == 'legacy-константы'
    assert params.medians == LEGACY_MEDIANS
    assert params.aggregates == {}


def test_preprocessing_params_partial_file_fills_legacy(tmp_path):
    """Неполный файл: недостающие секции добираются из legacy."""
    payload = {'created_at': '2026-01-01',
               'medians': {'age': 30.0, 'antiguedad': 1.0, 'renta': 5.0},
               'label_map': {'0': 'no_purchase'}}
    path = tmp_path / 'preprocessing_params.json'
    path.write_text(__import__('json').dumps(payload), encoding='utf-8')
    params = PreprocessingParams.from_json(path)
    assert params.medians['age'] == 30.0
    assert params.modes == LEGACY_MODES
    assert params.label_map == {0: 'no_purchase'}


def test_prepare_features_missing_fields_raises(legacy_params, tiny_personal_recs):
    with pytest.raises(ValueError, match='не хватает обязательных полей'):
        preprocessing.prepare_features({'ncodpers': 1}, legacy_params, tiny_personal_recs)


@pytest.mark.skipif(find_spec('featuretools') is None, reason='нужны featuretools/woodwork')
def test_prepare_features_golden(legacy_params, tiny_personal_recs):
    """Golden-test: фиксированный вход → ожидаемая матрица признаков."""
    pytest.importorskip('featuretools')
    row = preprocessing.prepare_features(make_profile(), legacy_params, tiny_personal_recs)

    assert row.shape == (1, len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
    assert set(row.columns) == set(NUMERIC_FEATURES) | set(CATEGORICAL_FEATURES)
    first = row.iloc[0]
    # детерминированные признаки — точные значения
    assert first['total_products'] == '1'
    assert first['age_interval'] == '42-45'
    assert first['recommended_product_id'] == 'ind_tjcr_fin_ult1'  # хит по ncodpers=1001
    assert first['renta'] == pytest.approx(120000.0)
    assert first['antiguedad'] == pytest.approx(66.0)
    assert first['log_renta'] == pytest.approx(np.log1p(120000.0))
    # legacy-режим без агрегатов: среднее по одной строке — само значение
    assert first['renta_vs_country_mean'] == pytest.approx(1.0)
    assert first['mean_renta_by_pais_residencia'] == '120000.0'
    # когорты дат считаются по ходу пайплайна, но модель их не использует —
    # в матрицу признаков они не входят (как и при обучении)
    assert 'fecha_alta' not in row.columns
    assert 'ult_fec_cli_1t' not in row.columns
    # детерминизм: повторный вызов даёт тот же вектор
    again = preprocessing.prepare_features(make_profile(), legacy_params, tiny_personal_recs)
    pd.testing.assert_frame_equal(row, again)


@pytest.mark.skipif(find_spec('featuretools') is None, reason='нужны featuretools/woodwork')
def test_prepare_features_unknown_client_zero_rec(legacy_params, tiny_personal_recs):
    pytest.importorskip('featuretools')
    row = preprocessing.prepare_features(make_profile(ncodpers=999999), legacy_params,
                                         tiny_personal_recs)
    assert row['recommended_product_id'].iloc[0] == '0'


def test_base_columns_cover_model_features():
    """BASE_COLUMNS + производные покрывают входы модели (защита от опечаток)."""
    derived = {'age_interval', 'recommended_product_id', 'total_products',
               'mean_renta_by_pais_residencia', 'median_antiguedad_by_pais_residencia',
               'mean_renta_by_segmento', 'median_antiguedad_by_segmento',
               'renta_antiguedad_ratio', 'log_renta', 'renta_vs_country_mean',
               'antiguedad + renta', 'antiguedad / renta', 'renta / antiguedad',
               'antiguedad * renta', 'NATURAL_LOGARITHM(antiguedad)',
               'NATURAL_LOGARITHM(renta)', 'SQUARE_ROOT(antiguedad)', 'SQUARE_ROOT(renta)'}
    assert set(BASE_COLUMNS) | derived >= set(CATEGORICAL_FEATURES) | set(NUMERIC_FEATURES)
