"""Общие фикстуры pytest-набора.

Тесты hermetic: синтетические профили и игрушечные артефакты собираются
в фикстурах, `data/train_ver2.csv` не нужен. Проверки настоящих артефактов
репозитория — в `test_artifacts.py` (скипаются, если файлов нет).
"""

import logging
import warnings

import numpy as np
import pandas as pd
import pytest

from preprocessing import PRODUCTS, PreprocessingParams, prepare_features

logging.getLogger('bank_recommender').setLevel(logging.ERROR)


def make_profile(**overrides):
    """Реалистичный профиль клиента: все сырые поля + продуктовые флаги."""
    profile = {
        'fecha_dato': '2015-12-28',
        'ncodpers': 1001,
        'ind_empleado': 'N',
        'pais_residencia': 'ES',
        'sexo': 'H',
        'age': 45.0,
        'fecha_alta': '2010-05-17',
        'ind_nuevo': 0.0,
        'antiguedad': 66.0,
        'indrel': 1.0,
        'ult_fec_cli_1t': '2015-06-30',
        'indrel_1mes': '1.0',
        'tiprel_1mes': 'I',
        'indresi': 'S',
        'indext': 'N',
        'conyuemp': 'N',
        'canal_entrada': 'KHE',
        'indfall': 'N',
        'tipodom': 1.0,
        'cod_prov': 28.0,
        'nomprov': 'MADRID',
        'ind_actividad_cliente': 1.0,
        'renta': 120000.0,
        'segmento': '02 - PARTICULARES',
    }
    for product in PRODUCTS:
        profile[product] = 0
    profile['ind_cco_fin_ult1'] = 1
    profile.update(overrides)
    return profile


@pytest.fixture()
def sample_profile():
    return make_profile()


@pytest.fixture()
def legacy_params():
    """Параметры без файла: запасные константы (детерминированы)."""
    return PreprocessingParams()


@pytest.fixture()
def tiny_personal_recs():
    """Игрушечные ALS-рекомендации: хит, NaN и отсутствие клиента."""
    return pd.DataFrame(
        {'recommended_product_id': ['ind_tjcr_fin_ult1', None]},
        index=pd.Index([1001, 1002], name='ncodpers'),
    )


@pytest.fixture(scope='session')
def tiny_model_bundle():
    """Игрушечный sklearn-пайплайн на признаках `prepare_features`.

    Готовит признаки для 24 синтетических профилей и обучает маленький
    RandomForest с классами 0/3 — достаточно для сквозных тестов API
    без настоящей `saved_model.pkl`.
    """
    pytest.importorskip('sklearn')
    from sklearn.compose import ColumnTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import OneHotEncoder, StandardScaler

    from preprocessing import CATEGORICAL_FEATURES, NUMERIC_FEATURES

    params = PreprocessingParams()
    recs = pd.DataFrame(
        {'recommended_product_id': ['ind_tjcr_fin_ult1']},
        index=pd.Index([1001], name='ncodpers'),
    )
    segments = ['02 - PARTICULARES', '03 - UNIVERSITARIO']
    frames = []
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        for i in range(24):
            profile = make_profile(
                ncodpers=2000 + i,
                age=20.0 + i,
                renta=50000.0 + i * 5000,
                pais_residencia='ES' if i % 2 else 'FR',
                segmento=segments[i % 2],
                ind_cco_fin_ult1=i % 2,
                ind_tjcr_fin_ult1=(i + 1) % 2,
            )
            frames.append(prepare_features(profile, params, recs))
    X = pd.concat(frames, ignore_index=True)
    y = pd.Series([0, 3] * 12)
    model = Pipeline([
        ('preprocessor', ColumnTransformer([
            ('num', StandardScaler(), NUMERIC_FEATURES),
            ('cat', OneHotEncoder(handle_unknown='ignore'), CATEGORICAL_FEATURES),
        ])),
        ('classifier', RandomForestClassifier(n_estimators=5, random_state=0)),
    ])
    model.fit(X, y)
    return {'model': model, 'label_map': {0: 'no_purchase', 3: 'ind_cco_fin_ult1'}}


def make_raw_panel(n_clients: int = 150, n_snapshots: int = 5, seed: int = 7) -> pd.DataFrame:
    """Синтетическая панель в схеме `train_ver2.csv` для сквозных проверок.

    Каждая строка — клиент в месячном снимке. Клиенты появляются не в первом
    снимке (есть cold start для ALS), часть уходит, продукты покупаются
    с вероятностью, зависящей от клиента: данных достаточно, чтобы прогнать
    весь путь «события → таргет → ALS-признак → предобработка → модель → метрики»
    без настоящего датасета.
    """
    rng = np.random.default_rng(seed)
    snapshots = pd.date_range('2015-02-28', periods=n_snapshots, freq='ME')
    records = []
    for offset in range(n_clients):
        client_id = 10_000 + offset
        join_index = int(rng.integers(0, max(n_snapshots - 1, 1)))
        propensity = float(rng.uniform(0.05, 0.7))
        age = float(int(rng.normal(45, 15)))
        renta = float(np.round(rng.uniform(20_000.0, 250_000.0), 2))
        antiguedad = float(int(rng.integers(0, 200)))
        fecha_alta = (snapshots[join_index] - pd.Timedelta(days=int(antiguedad) * 30)
                      ).strftime('%Y-%m-%d')
        product_state = {product: 0 for product in PRODUCTS}
        product_state['ind_cco_fin_ult1'] = 1
        for snapshot_index in range(join_index, n_snapshots):
            if snapshot_index > join_index and rng.random() < 0.05:
                break
            if rng.random() < propensity:
                available = [product for product, value in product_state.items() if value == 0]
                if available:
                    product_state[str(rng.choice(available))] = 1
            records.append({
                'ncodpers': client_id,
                'fecha_dato': snapshots[snapshot_index],
                'fecha_alta': fecha_alta,
                'ind_empleado': str(rng.choice(['N', 'A', 'S'])),
                'pais_residencia': str(rng.choice(['ES', 'FR'], p=[0.9, 0.1])),
                'sexo': str(rng.choice(['H', 'V'])),
                'age': age,
                'ind_nuevo': int(snapshot_index == join_index),
                'antiguedad': antiguedad + 30 * (snapshot_index - join_index),
                'indrel': 1.0,
                # Дата последнего выхода из банка: заполнена у части клиентов,
                # иначе когорты дат в fit_preprocessing_params вырождаются.
                'ult_fec_cli_1t': (snapshots[max(snapshot_index - 1, 0)]
                                   .strftime('%Y-%m-%d') if rng.random() < 0.3 else None),
                'indrel_1mes': '1.0',
                'tiprel_1mes': str(rng.choice(['A', 'I'])),
                'indresi': 'S',
                'indext': 'N',
                'conyuemp': 'N',
                'canal_entrada': str(rng.choice(['KHE', 'KAT'])),
                'indfall': 'N',
                'tipodom': 1.0,
                'cod_prov': float(rng.choice([28.0, 8.0])),
                'nomprov': str(rng.choice(['MADRID', 'BARCELONA'])),
                'ind_actividad_cliente': int(rng.random() < 0.5),
                'renta': renta,
                'segmento': str(rng.choice(['02 - PARTICULARES', '03 - UNIVERSITARIO'])),
                **product_state,
            })
    return pd.DataFrame(records)
