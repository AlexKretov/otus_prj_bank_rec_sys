"""Общие фикстуры pytest-набора (CODE_REVIEW §P2.16).

Тесты hermetic: синтетические профили и игрушечные артефакты собираются
в фикстурах, `data/train_ver2.csv` не нужен. Проверки настоящих артефактов
репозитория — в `test_artifacts.py` (скипаются, если файлов нет).
"""

import logging
import warnings

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
    """Параметры без файла: legacy-константы (детерминированы)."""
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
    pytest.importorskip('featuretools')
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
