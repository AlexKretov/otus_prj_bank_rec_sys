"""Unit-тесты общего модуля предобработки (CODE_REVIEW §P2.16).

Чистые функции и сквозной `prepare_features` тестируются без тяжёлых
зависимостей: арифметика признаков считается pandas/numpy. Тест на паритет
с featuretools (эталонная семантика старой DFS-версии) скипается, если
featuretools не установлен.
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
    feature_engineering,
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
    assert aggregates['mean_renta_by_pais_residencia'] == {
        'ES': 150.0, 'FR': 300.0, '__default__': 200.0}

    single = frame.iloc[[0]].copy()
    served, _ = preprocessing.manual_transformations(single, aggregates)
    for col in set(aggregates) - {'renta_vs_country_mean'}:
        assert served[col].iloc[0] == trained[col].iloc[0]
    assert served['renta_vs_country_mean'].iloc[0] == pytest.approx(100.0 / 150.0)


def test_manual_transformations_unseen_category_uses_default():
    """Категория, которой не было при обучении, → '__default__', а не NaN.

    При временном разбиении и на проде встречаются значения, отсутствовавшие
    в train; NaN в числовых признаках ронял бы RandomForest.
    """
    trained, aggregates = preprocessing.manual_transformations(pd.DataFrame({
        'pais_residencia': ['ES', 'FR'], 'segmento': ['A', 'B'],
        'renta': [100.0, 300.0], 'antiguedad': [10.0, 30.0],
    }), None)
    served, _ = preprocessing.manual_transformations(pd.DataFrame({
        'pais_residencia': ['XX'], 'segmento': ['ZZ'],
        'renta': [60.0], 'antiguedad': [5.0],
    }), aggregates)
    first = served.iloc[0]
    assert first['mean_renta_by_pais_residencia'] == pytest.approx(200.0)  # mean по train
    assert first['median_antiguedad_by_segmento'] == pytest.approx(20.0)  # median по train
    assert not served[['mean_renta_by_pais_residencia', 'renta_vs_country_mean']].isna().any().any()
    # агрегаты старого формата (без '__default__') — fallback через среднее значений
    legacy_aggregates = {key: {cat: val for cat, val in values.items() if cat != '__default__'}
                         for key, values in aggregates.items()}
    served_legacy, _ = preprocessing.manual_transformations(pd.DataFrame({
        'pais_residencia': ['XX'], 'segmento': ['ZZ'],
        'renta': [60.0], 'antiguedad': [5.0],
    }), legacy_aggregates)
    assert not served_legacy[['mean_renta_by_pais_residencia',
                              'renta_vs_country_mean']].isna().any().any()


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


def test_prepare_features_golden(legacy_params, tiny_personal_recs):
    """Golden-test: фиксированный вход → ожидаемая матрица признаков."""
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


def test_prepare_features_unknown_client_zero_rec(legacy_params, tiny_personal_recs):
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


def test_feature_engineering_thread_safe():
    """Параллельные вызовы не падают (регрессия на featuretools-гонку).

    Старая версия строила featuretools EntitySet + DFS на каждый вызов;
    при ~50 конкурентных запросах каждый четвёртый падал с
    `KeyError: 'DataFrame main does not exist in bank_data'` → HTTP 500
    → SLO success_rate ~70% в test.ipynb.
    """
    from concurrent.futures import ThreadPoolExecutor

    profile = make_profile()

    def one(_):
        try:
            row = preprocessing.prepare_features(dict(profile), PreprocessingParams(), None)
            return None if 'antiguedad + renta' in row.columns else 'нет признаков'
        except Exception as exc:  # noqa: BLE001 — любая ошибка = провал регрессии
            return f'{type(exc).__name__}: {exc}'

    with ThreadPoolExecutor(max_workers=32) as executor:
        errors = [result for result in executor.map(one, range(64)) if result]
    assert not errors, errors[:3]


@pytest.mark.skipif(find_spec('featuretools') is None, reason='нужен featuretools (эталон)')
def test_feature_engineering_matches_featuretools():
    """Паритет с DFS: те же колонки, dtypes и значения, что давал featuretools.

    Эталон — дословно старая реализация (EntitySet + ft.dfs) до замены на
    pandas/numpy. Граничные случаи включают нули и отрицательные значения
    (log(0) → -inf, log/sqrt отрицательных → NaN, деление на ноль → inf).
    """
    import warnings

    pytest.importorskip('woodwork')
    import featuretools as ft
    import woodwork

    def reference_dfs(df):
        entity_set = ft.EntitySet(id='bank_data')
        entity_set = entity_set.add_dataframe(
            dataframe_name='main', dataframe=df, index='unique_id', make_index=True,
            logical_types={
                'antiguedad': woodwork.logical_types.Double,
                'renta': woodwork.logical_types.Double,
                **{col: woodwork.logical_types.Categorical for col in df.columns
                   if col not in ['antiguedad', 'renta']},
            })
        feature_matrix, _ = ft.dfs(
            entityset=entity_set, target_dataframe_name='main',
            trans_primitives=['add_numeric', 'multiply_numeric', 'divide_numeric',
                              'natural_logarithm', 'square_root'],
            agg_primitives=['mean', 'median', 'std', 'max', 'min', 'count', 'num_unique'],
            where_primitives=['count'], max_depth=2, features_only=False, verbose=False)
        matrix = feature_matrix.drop(columns=['unique_id', 'index'], errors='ignore')
        matrix, _ = preprocessing.manual_transformations(matrix, None)
        return matrix.loc[:, ~matrix.columns.duplicated()]

    rng = np.random.default_rng(0)
    size = 20
    frame = pd.DataFrame({
        'ind_empleado': rng.choice(['N', 'A'], size),
        'pais_residencia': rng.choice(['ES', 'FR'], size),
        'sexo': rng.choice(['H', 'V'], size),
        'ind_nuevo': rng.choice([0.0, 1.0], size),
        'antiguedad': rng.choice([-5, 0, 1, 25, 66, 200], size),
        'indrel_1mes': rng.choice(['1.0', 'P'], size),
        'tiprel_1mes': rng.choice(['A', 'I'], size),
        'indresi': ['S'] * size,
        'conyuemp': ['N'] * size,
        'canal_entrada': rng.choice(['KHE', 'KAT'], size),
        'indfall': ['N'] * size,
        'cod_prov': rng.choice([28.0, 8.0], size),
        'ind_actividad_cliente': rng.choice([0.0, 1.0], size),
        'renta': rng.choice([0.0, -3.5, 50000.0, 120000.0, 250000.0], size),
        'segmento': rng.choice(['02 - PARTICULARES', '03 - UNIVERSITARIO'], size),
        **{product: rng.choice([0, 1], size) for product in PRODUCTS},
        'total_products': rng.integers(0, 10, size),
        'age_interval': rng.choice(['24-28', '37-41'], size),
        'recommended_product_id': rng.choice([0, 'ind_tjcr_fin_ult1'], size),
    })[BASE_COLUMNS]

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        expected = reference_dfs(frame.copy())
        actual, _ = feature_engineering(frame.copy())

    assert list(actual.columns) == list(expected.columns)
    # check_names=False: имя 'unique_id' у индекса — внутренний артефакт DFS
    pd.testing.assert_frame_equal(actual, expected, check_names=False)


def test_fold_rare_categories_matches_training_semantics():
    """Строковые ключи сравниваются в строковом домене — как на обучении.

    На обучении колонки приводились к str до подсчёта частот, поэтому ключи
    словаря — строки, и float-значение 28.0 из запроса должно сворачиваться
    по ключу '28.0'. Старая семантика (равенство поколоночному `replace`)
    сравнивала в исходном домене и float не сворачивала — train/serve skew.
    Числовые ключи по-прежнему работают в исходном домене.
    """
    replacer = {
        'cod_prov': {'28.0': 'other'},              # строковый ключ, float-колонка
        'indrel_1mes': {3.0: 'other'},              # числовой ключ — исходный домен
        'sexo': {'H': 'other'},                     # обычный случай
        'segmento': {},                             # пустой словарь — no-op
    }
    frame = pd.DataFrame({
        'cod_prov': [28.0, 8.0, np.nan],
        'indrel_1mes': [3.0, '1.0', 4.0],
        'sexo': ['H', 'V', None],
        'segmento': ['02 - PARTICULARES', '03 - UNIVERSITARIO', None],
    })
    actual = preprocessing.fold_rare_categories(frame.copy(), replacer)
    assert list(actual['cod_prov'].fillna('#')) == ['other', 8.0, '#']
    assert list(actual['indrel_1mes'].fillna('#')) == ['other', '1.0', 4.0]
    assert list(actual['sexo'].fillna('#')) == ['other', 'V', '#']
    assert list(actual['segmento'].fillna('#')) == [
        '02 - PARTICULARES', '03 - UNIVERSITARIO', '#']


# --- P3: статистики только на train, временное разбиение ---------------------


def make_raw_clients(n=60, seed=42):
    """Синтетический raw-фрейм профилей (как срез train_ver2.csv) для fit/apply."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        'ind_empleado': rng.choice(['N', 'A', 'S', np.nan], n, p=[0.7, 0.1, 0.1, 0.1]),
        'pais_residencia': rng.choice(['ES', 'FR', 'MX', 'ZZ'], n, p=[0.8, 0.1, 0.05, 0.05]),
        'sexo': rng.choice(['H', 'V'], n),
        'age': rng.normal(45, 15, n).round(0).astype(int),
        'fecha_alta': rng.choice(['2010-05-17', '2012-01-01', '2014-07-28', None], n),
        'ind_nuevo': rng.choice([0.0, 1.0], n, p=[0.9, 0.1]),
        'antiguedad': np.where(rng.random(n) < 0.05, -999999.0,
                               rng.integers(0, 200, n).astype(float)),
        'ult_fec_cli_1t': rng.choice(['2015-06-30', '2015-12-24', None], n),
        'indrel_1mes': rng.choice(['1.0', 'P', 3.0], n),
        'tiprel_1mes': rng.choice(['A', 'I'], n),
        'indresi': ['S'] * n,
        'conyuemp': ['N'] * n,
        'canal_entrada': rng.choice(['KHE', 'KAT', 'K00'], n, p=[0.8, 0.1, 0.1]),
        'indfall': ['N'] * n,
        'cod_prov': rng.choice([28.0, 8.0, 47.0], n, p=[0.8, 0.1, 0.1]),
        'ind_actividad_cliente': rng.choice([0.0, 1.0], n),
        'renta': np.round(rng.uniform(10000.0, 300000.0, n), 2),
        'segmento': rng.choice(['01 - TOP', '02 - PARTICULARES', '03 - UNIVERSITARIO'], n),
    })
    for product in PRODUCTS:
        frame[product] = rng.choice([0, 1], n, p=[0.7, 0.3])
    return frame


def test_fit_preprocessing_params_computed_on_fit_frame():
    """Статистики считаются из переданного (train) фрейма и воспроизводимы."""
    frame = make_raw_clients()
    params = preprocessing.fit_preprocessing_params(frame)
    # медианы — ровно по train-фрейму (antiguedad уже без сентинели -999999)
    cleaned = preprocessing.coerce_numeric(frame.copy())
    assert params.medians['age'] == pytest.approx(float(cleaned['age'].median()))
    assert params.medians['renta'] == pytest.approx(float(cleaned['renta'].median()))
    assert params.medians['antiguedad'] == pytest.approx(
        float(cleaned['antiguedad'].median()))
    # клиппинг — квантили 1%/96% по train
    work = preprocessing.fill_missing(cleaned, params.medians, params.modes)
    assert params.clip_bounds['renta'] == pytest.approx(
        [float(work['renta'].quantile(0.01)), float(work['renta'].quantile(0.96))])
    # возрастные бины покрывают train-возраста
    assert params.age_intervals[0][0] <= work['age'].min()
    assert params.age_intervals[-1][1] >= work['age'].max()
    # когорты дат прошли самопроверку: спецификация воспроизводит обучающие метки
    assert set(params.date_cohorts) == {'fecha_alta', 'ult_fec_cli_1t'}
    # эталон дрейфа посчитан по трём числовым признакам
    assert set(params.drift_reference) == {'age', 'antiguedad', 'renta'}
    for reference in params.drift_reference.values():
        assert len(reference['edges']) == len(reference['shares']) + 1
        assert sum(reference['shares']) == pytest.approx(1.0)


def test_fit_apply_train_serve_parity(tiny_personal_recs):
    """Строка применённой батч-предобработки == prepare_features того же профиля.

    Главный train/serve-инвариант P2 теперь работает и в новом режиме
    «fit на train → apply на test»: признаки одного и того же клиента
    совпадают между офлайном и продом один в один.
    """
    fit_frame = make_raw_clients()
    params = preprocessing.fit_preprocessing_params(fit_frame)
    # агрегаты обучения (как в modeling.ipynb: feature_engineering на train)
    batch = preprocessing.apply_preprocessing(fit_frame.copy(), params)
    batch['recommended_product_id'] = '0'  # на обучении — подмёрж ALS-таблицы
    batch = batch[BASE_COLUMNS]
    batch['antiguedad'] = batch['antiguedad'].astype(int)
    features, aggregates = preprocessing.feature_engineering(batch)

    # прод-эквивалент: те же параметры (с агрегатами train), построчно
    params = preprocessing.PreprocessingParams(
        medians=params.medians, modes=params.modes, clip_bounds=params.clip_bounds,
        age_intervals=params.age_intervals, date_cohorts=params.date_cohorts,
        replacer=params.replacer, aggregates=aggregates, source='test')

    for i in (0, 7, len(fit_frame) - 1):
        profile = fit_frame.iloc[i].to_dict()
        profile['ncodpers'] = 424242  # клиента нет в ALS-таблице → rec 0, как в batch
        row = preprocessing.prepare_features(profile, params, tiny_personal_recs)
        expected = features.iloc[[i]].copy()
        assert set(row.columns) == set(expected.columns)
        for col in row.columns:
            actual_value = row.iloc[0][col]
            expected_value = expected.iloc[0][col]
            if col in NUMERIC_FEATURES:
                actual_float = float(actual_value)
                expected_float = float(expected_value)
                both_nan = np.isnan(actual_float) and np.isnan(expected_float)
                assert both_nan or actual_float == pytest.approx(expected_float, abs=1e-6), col
            else:
                assert str(actual_value) == str(expected_value), col


def test_temporal_split_orders_by_snapshot_date():
    dates = pd.Series(pd.to_datetime(
        ['2015-01-28'] * 10 + ['2015-05-28'] * 10 + ['2015-12-28'] * 10))
    cutoff, train_mask, test_mask = preprocessing.temporal_split(dates, test_size=0.3)
    assert train_mask.sum() and test_mask.sum()
    assert dates[train_mask].max() <= cutoff < dates[test_mask].min()
    # границы не пересекаются: поздние срезы целиком в test
    assert not (train_mask & test_mask).any()
    assert (train_mask | test_mask).all()


def test_temporal_split_two_dates_both_sides_non_empty():
    dates = pd.Series(pd.to_datetime(['2015-01-28'] * 7 + ['2015-02-28'] * 3))
    _, train_mask, test_mask = preprocessing.temporal_split(dates, test_size=0.3)
    assert train_mask.any() and test_mask.any()


def test_temporal_split_single_date_raises():
    dates = pd.Series(pd.to_datetime(['2015-01-28'] * 10))
    with pytest.raises(ValueError, match='2 различные даты'):
        preprocessing.temporal_split(dates)


def test_compute_drift_reference_deciles():
    frame = pd.DataFrame({'age': np.arange(100, dtype=float),
                          'antiguedad': np.arange(100, dtype=float),
                          'renta': np.concatenate([[np.nan], np.arange(1, 100, dtype=float)])})
    reference = preprocessing.compute_drift_reference(frame, n_bins=10)
    assert set(reference) == {'age', 'antiguedad', 'renta'}
    for info in reference.values():
        assert len(info['edges']) == 10 + 1
        assert info['edges'][0] == float('-inf') and info['edges'][-1] == float('inf')
        assert sum(info['shares']) == pytest.approx(1.0)
    # на 100 равномерно различных значениях децили — ровно десятые доли
    assert reference['age']['shares'] == pytest.approx([0.1] * 10, abs=1e-9)
    assert reference['antiguedad']['shares'] == pytest.approx([0.1] * 10, abs=1e-9)
