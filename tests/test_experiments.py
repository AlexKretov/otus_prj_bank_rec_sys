"""Тесты протокола экспериментов: плечи моделей, честность сравнения, абляции.

Проверяем, что Random Forest и CatBoost собираются из одного кода, обучаются на одной
матрице признаков, оцениваются одной функцией и что абляция ALS-признака меняет
только состав признаков, а не способ оценки.
"""

import numpy as np
import pandas as pd
import pytest

import experiments
from recsys_metrics import OTHER_CLASS

CLASSES = [0, 3, 5, 99]
CANDIDATES = [3, 5]


def make_synthetic_features(n: int = 200, seed: int = 0):
    """Матрица признаков в форме модели: числа + категориальные строки."""
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        'antiguedad': rng.integers(0, 200, n).astype(float),
        'renta': rng.uniform(20_000, 250_000, n).round(2),
        'recommended_product_id': rng.choice(['0', 'ind_cco_fin_ult1', 'ind_cno_fin_ult1'], n),
        'segmento': rng.choice(['02 - PARTICULARES', '03 - UNIVERSITARIO'], n),
        'ind_empleado': rng.choice(['N', 'A'], n),
    })
    # Класс зависит от признаков: модель должна выучить что-то осмысленное.
    latent = frame['renta'] + 300 * frame['antiguedad']
    target = np.where(frame['recommended_product_id'] == 'ind_cco_fin_ult1', 3, 0)
    target = np.where(latent > np.quantile(latent, 0.8), 5, target)
    target = np.where(rng.random(n) < 0.05, OTHER_CLASS, target)
    frame['purchase'] = target
    return frame


NUMERIC = ['antiguedad', 'renta']
CATEGORICAL = ['recommended_product_id', 'segmento', 'ind_empleado']


def test_macro_roc_auc_rewards_correct_ranking():
    y_true = [0, 0, 3, 3]
    perfect = np.array([[0.9, 0.05, 0.05, 0.0], [0.8, 0.1, 0.1, 0.0],
                        [0.05, 0.9, 0.05, 0.0], [0.1, 0.8, 0.1, 0.0]])
    random_like = np.full((4, 4), 0.25)
    assert experiments.macro_roc_auc(y_true, perfect, CLASSES) == pytest.approx(1.0)
    assert experiments.macro_roc_auc(y_true, random_like, CLASSES) == pytest.approx(0.5)


def test_macro_roc_auc_ignores_absent_classes_and_renormalises():
    y_true = [0, 0, 3]
    proba = np.array([[1.0, 0.0, 0.0, 0.0], [0.5, 0.5, 0.0, 0.0], [0.0, 0.0, 0.0, 3.0]])
    score = experiments.macro_roc_auc(y_true, proba, CLASSES)
    assert 0.0 <= score <= 1.0  # не падаем на классах, которых нет в y_true


@pytest.mark.parametrize('model_kind', experiments.MODEL_KINDS)
def test_pipelines_fit_and_evaluate(model_kind):
    if model_kind == 'catboost':
        pytest.importorskip('catboost')
    frame = make_synthetic_features()
    X, y = frame.drop(columns=['purchase']), frame['purchase']

    params = {'classifier__n_estimators': 5, 'classifier__max_depth': 5} \
        if model_kind == 'random_forest' else {'classifier__iterations': 20,
                                               'classifier__depth': 3}
    pipeline = experiments.build_pipeline(model_kind, NUMERIC, CATEGORICAL, params=params)
    pipeline.fit(X, y)

    result = experiments.evaluate(pipeline, X, y, candidate_classes=CANDIDATES)
    assert set(result.metrics) >= {'roc_auc_ovr_macro', 'accuracy', 'f1_macro', 'pr_auc_macro'}
    assert {'shortlist_purchasers', 'purchasers_any', 'all_rows'} <= set(result.top_k['scope'])
    assert {'precision_at_k', 'recall_at_k', 'ndcg_at_k'} <= set(result.top_k)
    summary = result.summary('model')
    assert summary['variant'] == 'model'
    assert 'shortlist_purchasers_ndcg_at_k' in summary


@pytest.mark.parametrize('model_kind', experiments.MODEL_KINDS)
def test_study_builds_model_with_best_params(model_kind):
    if model_kind == 'catboost':
        pytest.importorskip('catboost')
    frame = make_synthetic_features(n=150, seed=1)
    X, y = frame.drop(columns=['purchase']), frame['purchase']

    study = experiments.run_study(model_kind, X, y, NUMERIC, CATEGORICAL,
                                  n_trials=1, timeout=300, n_splits=2)
    assert study.model_kind == model_kind
    assert study.best_params
    assert study.n_trials == 1
    assert 'classifier__' in next(iter(study.best_params))

    model = experiments.build_model(study, NUMERIC, CATEGORICAL)
    model.fit(X, y)
    result = experiments.evaluate(model, X, y, candidate_classes=CANDIDATES)
    assert np.isfinite(result.metrics['roc_auc_ovr_macro'])


def test_catboost_preprocessor_passes_categories_without_one_hot():
    pytest.importorskip('catboost')
    frame = make_synthetic_features(n=50)
    preprocessor = experiments.build_preprocessor('catboost', NUMERIC, CATEGORICAL)
    transformed = preprocessor.fit_transform(frame[NUMERIC + CATEGORICAL])
    assert isinstance(transformed, pd.DataFrame), 'CatBoost ожидает колонки по именам'
    assert list(transformed.columns) == NUMERIC + CATEGORICAL
    assert transformed['segmento'].dtype == object


def test_random_forest_preprocessor_scales_and_one_hots():
    frame = make_synthetic_features(n=50)
    preprocessor = experiments.build_preprocessor('random_forest', NUMERIC, CATEGORICAL)
    transformed = preprocessor.fit_transform(frame[NUMERIC + CATEGORICAL])
    names = preprocessor.get_feature_names_out()
    assert transformed.shape[0] == len(frame)
    assert any(name.startswith('cat__segmento') for name in names)
    numeric_block = np.asarray(transformed[:, :len(NUMERIC)])
    if hasattr(numeric_block, 'todense'):
        numeric_block = numeric_block.todense()
    assert np.allclose(np.asarray(numeric_block).std(axis=0), 1.0, atol=0.15)


def test_unknown_model_kind_raises():
    with pytest.raises(ValueError):
        experiments.build_pipeline('xgboost', NUMERIC, CATEGORICAL)
    with pytest.raises(ValueError):
        experiments.suggest_params(None, 'xgboost')


def test_drop_feature_removes_column_and_guards_typos():
    features = ['a', 'b', experiments.ALS_FEATURE]
    assert experiments.drop_feature(features, experiments.ALS_FEATURE) == ['a', 'b']
    with pytest.raises(ValueError):
        experiments.drop_feature(features, 'unknown_column')


def test_ablation_and_comparison_tables_shape():
    frame = make_synthetic_features(n=160, seed=2)
    X, y = frame.drop(columns=['purchase']), frame['purchase']
    baseline_params = {'classifier__n_estimators': 5, 'classifier__max_depth': 5}

    variants = {}
    for name, features in (('with_als', NUMERIC + CATEGORICAL),
                           ('without_als', experiments.drop_feature(NUMERIC + CATEGORICAL,
                                                                    experiments.ALS_FEATURE))):
        categorical = [column for column in features if column in CATEGORICAL]
        numeric = [column for column in features if column in NUMERIC]
        model = experiments.build_pipeline('random_forest', numeric, categorical,
                                           params=baseline_params)
        model.fit(X[features], y)
        variants[name] = experiments.evaluate(model, X[features], y,
                                              candidate_classes=CANDIDATES)

    table = experiments.ablation_rows(variants)
    assert list(table['variant']) == ['with_als', 'without_als']
    assert {'pr_auc_macro', 'shortlist_purchasers_precision_at_k'} <= set(table)

    study = experiments.StudyResult(model_kind='random_forest', best_params=baseline_params,
                                    best_value=0.7, cv_std=0.01, n_trials=3, elapsed_sec=1.0)
    comparison = experiments.comparison_table([study], {'random_forest': variants['with_als']})
    assert len(comparison) == 1
    assert {'model', 'cv_roc_auc_macro', 'f1_macro', 'best_params'} <= set(comparison)
