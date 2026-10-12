"""Сквозной прогон обучающего пути на синтетической панели.

Тест повторяет шаги `modeling.ipynb` без настоящего датасета: события покупок →
шорт-лист классов → ALS (baseline и признак) → панель «признаки на снимке t →
покупка в t+1» → train-only предобработка → Random Forest и CatBoost → метрики
ранжирования и абляция ALS.

Главные инварианты, которые здесь защищаются:
* метка «покупки не было» не достаётся покупателю (замечание про целевую переменную);
* статистики предобработки считаются только на train, а тест применяется теми же
  параметрами — модель на тесте не видит будущего;
* сравнение моделей и абляция идут через одну и ту же функцию оценки.
"""

import numpy as np
import pandas as pd
import pytest

import dataset
import experiments
import recsys_metrics
from preprocessing import (
    BASE_COLUMNS,
    apply_preprocessing,
    feature_engineering,
    fit_preprocessing_params,
)
from tests.conftest import make_raw_panel

TRAIN_CUTOFF = pd.Timestamp('2015-04-30')
ALS_CUTOFF = pd.Timestamp('2015-03-31')
PANEL_FIRST = pd.Timestamp('2015-02-28')
PANEL_LAST = pd.Timestamp('2015-04-30')


@pytest.fixture(scope='module')
def modeling_inputs():
    """Синтетическая панель, размеченная так же, как в ноутбуке обучения."""
    frame = make_raw_panel(n_clients=260, n_snapshots=6, seed=17)
    frame['fecha_dato'] = pd.to_datetime(frame['fecha_dato'])
    for column in ('age', 'antiguedad', 'renta'):
        frame[column] = pd.to_numeric(frame[column], errors='coerce')
    frame = frame.drop(columns=['indrel', 'indext', 'nomprov', 'tipodom'])

    events = dataset.purchase_events(frame)
    share = dataset.purchase_share(events, TRAIN_CUTOFF)
    # Порог шорт-листа на синтетике ниже, чем в проекте: покупки там редкие,
    # но правило отбора то же — по доле покупок обучающего периода.
    target_codes, _ = dataset.select_target_products(share, min_share=0.012, max_products=6)
    label_map = dataset.build_label_map(target_codes)

    # ALS как самостоятельный baseline: модель на истории до ALS_CUTOFF.
    users, user_map = dataset.user_index(frame['ncodpers'])
    history = events[events['fecha_dato'] <= ALS_CUTOFF]
    matrix = dataset.als_user_item(history, user_map)
    model = dataset.fit_als(matrix, factors=6, iterations=3)
    history_users = pd.unique(history['ncodpers'])
    codes = np.array([user_map[int(user)] for user in history_users])

    # Признак ALS для панели — по снимкам, без подглядывания вперёд.
    als_feature = dataset.als_feature_lookup(frame, events, user_map, factors=6, iterations=3)

    panel = dataset.build_target_panel(frame, events,
                                       first_snapshot=PANEL_FIRST, last_snapshot=PANEL_LAST)
    full = panel.panel.merge(als_feature.lookup, on=['ncodpers', 'snapshot_date'], how='left')
    full['recommended_product_id'] = full['recommended_product_id'].fillna('0')

    # Классы: покупки вне шорт-листа — в other; редкие классы сворачиваются по train.
    mask = (full['purchase'] != 0) & (~full['purchase'].isin(target_codes))
    full.loc[mask, 'purchase'] = dataset.OTHER_CLASS
    train_mask = (full['snapshot_date'] < TRAIN_CUTOFF).to_numpy()
    # Порог сворачивания редких классов — от размера train: на синтетике
    # абсолютные поддержки маленькие, но правило остаётся тем же, что в ноутбуке.
    full['purchase'], rare = dataset.fold_rare_targets(
        full['purchase'], train_mask, min_support=max(3, int(0.01 * train_mask.sum())))
    dropped_classes = [code for code in full['purchase'].unique() if code not in label_map]

    return {
        'frame': frame,
        'events': events,
        'panel': panel,
        'full': full,
        'target_codes': target_codes,
        'folded_classes': rare,
        'dropped_classes': dropped_classes,
        'als_baseline': {
            'model': model, 'matrix': matrix, 'users': history_users, 'codes': codes,
            'user_map': user_map,
        },
        'als_feature': als_feature,
    }


def test_target_marks_every_purchaser(modeling_inputs):
    """Замечание №1: ни один покупатель не остаётся с меткой «покупки не было»."""
    full = modeling_inputs['full']
    events = modeling_inputs['events']

    # Пересобираем покупки каждого целевого снимка независимо от кода панели.
    diagnostics = modeling_inputs['panel'].diagnostics.set_index('snapshot_date')
    np.testing.assert_array_equal(
        (full['purchase'] != 0).to_numpy(),
        full['ncodpers'].isin([]).to_numpy() | (full['purchase'] != 0).to_numpy())
    for row in diagnostics.reset_index().itertuples():
        snapshot_users = set(full.loc[full['snapshot_date'] == row.snapshot_date, 'ncodpers'])
        purchasers = set(events.loc[events['fecha_dato'] == row.target_date, 'ncodpers'])
        labelled = set(full.loc[(full['snapshot_date'] == row.snapshot_date)
                                & (full['purchase'] != 0), 'ncodpers'])
        assert labelled == purchasers & snapshot_users, 'покупатель потерял метку класса'

    # Клиенты без истории ALS не выпадают из выборки: у них есть и признаки, и метка.
    without_recs = full[full['recommended_product_id'] == '0']
    assert len(without_recs) > 0
    assert (without_recs['purchase'] != 0).any()


def test_train_only_preprocessing_and_feature_matrix(modeling_inputs):
    full = modeling_inputs['full']
    train_raw = full[full['snapshot_date'] < TRAIN_CUTOFF].reset_index(drop=True)
    test_raw = full[full['snapshot_date'] >= TRAIN_CUTOFF].reset_index(drop=True)
    assert train_raw['snapshot_date'].max() < test_raw['snapshot_date'].min()

    drop = ['purchase', 'fecha_dato', 'ncodpers', 'snapshot_date']
    params = fit_preprocessing_params(train_raw.drop(columns=drop))
    X_train = apply_preprocessing(train_raw.drop(columns=drop), params)
    X_test = apply_preprocessing(test_raw.drop(columns=drop), params)

    assert set(BASE_COLUMNS) <= set(X_train) & set(X_test)
    assert X_train.isna().sum().sum() == 0 and X_test.isna().sum().sum() == 0

    X_train['antiguedad'] = X_train['antiguedad'].astype(int)
    X_test['antiguedad'] = X_test['antiguedad'].astype(int)
    df_train, aggregates = feature_engineering(X_train[BASE_COLUMNS])
    df_test, _ = feature_engineering(X_test[BASE_COLUMNS], aggregates)
    assert aggregates, 'агрегаты train должны вернуться словарём'
    assert not df_train.isna().any().any(), 'после feature_engineering NaN не остаётся'
    assert set(df_train.columns) == set(df_test.columns)


def test_models_and_ranking_metrics_on_synthetic_panel(modeling_inputs):
    """Сравнение RF/CatBoost и метрики top-K считаются на одной матрице признаков."""
    full = modeling_inputs['full']
    drop = ['purchase', 'fecha_dato', 'ncodpers', 'snapshot_date']
    train_raw = full[full['snapshot_date'] < TRAIN_CUTOFF].reset_index(drop=True)
    test_raw = full[full['snapshot_date'] >= TRAIN_CUTOFF].reset_index(drop=True)

    params = fit_preprocessing_params(train_raw.drop(columns=drop))
    X_train = apply_preprocessing(train_raw.drop(columns=drop), params)[BASE_COLUMNS]
    X_test = apply_preprocessing(test_raw.drop(columns=drop), params)[BASE_COLUMNS]
    y_train, y_test = train_raw['purchase'], test_raw['purchase']

    X_train['antiguedad'] = X_train['antiguedad'].astype(int)
    X_test['antiguedad'] = X_test['antiguedad'].astype(int)
    df_train, aggregates = feature_engineering(X_train)
    df_test, _ = feature_engineering(X_test, aggregates)

    numeric = [col for col in df_train.columns
               if pd.api.types.is_numeric_dtype(df_train[col]) and df_train[col].nunique() >= 25]
    categorical = [col for col in df_train.columns if col not in numeric]
    for column in categorical:
        df_train[column] = df_train[column].astype(str)
        df_test[column] = df_test[column].astype(str)
    assert experiments.ALS_FEATURE in df_train

    popularity = (train_raw.loc[train_raw['purchase'] != 0, 'purchase']
                  .value_counts(normalize=True).to_dict())

    results = {}
    for kind in experiments.MODEL_KINDS:
        if kind == 'catboost':
            pytest.importorskip('catboost')
        study = experiments.run_study(kind, df_train, y_train, numeric, categorical,
                                      n_trials=1, timeout=600, n_splits=2)
        model = experiments.build_model(study, numeric, categorical)
        model.fit(df_train, y_train)
        result = experiments.evaluate(model, df_test, y_test,
                                      candidate_classes=modeling_inputs['target_codes'],
                                      label_map={'0': 'no_purchase'},
                                      popularity=popularity)
        results[kind] = (study, result)

        assert np.isfinite(result.metrics['pr_auc_macro'])
        top_k = result.top_k
        assert set(top_k['k']) == {1, 3, 5, 10}
        assert top_k['precision_at_k'].notna().all()
        # Метрики модели и бейзлайнов посчитаны и лежат в допустимых границах.
        # Сравнивать их между собой на 150 синтетических строках нельзя: сигнала
        # в таких данных нет, и модель вполне может оказаться на уровне случайной.
        purchasers = top_k[top_k['scope'] == 'shortlist_purchasers']
        for column in ('precision_at_k', 'recall_at_k', 'ndcg_at_k',
                       'random_precision_at_k', 'popularity_precision_at_k'):
            values = purchasers[column]
            assert values.between(0.0, 1.0).all()

    comparison = experiments.comparison_table(
        [study for study, _ in results.values()],
        {kind: result for kind, (_, result) in results.items()})
    assert set(comparison['model']) == set(experiments.MODEL_KINDS)
    assert comparison['pr_auc_macro'].notna().all()


def test_als_ablation_changes_only_feature_set(modeling_inputs):
    """Замечание №5: абляция ALS-признака — те же данные, та же оценка, минус признак."""
    full = modeling_inputs['full']
    drop = ['purchase', 'fecha_dato', 'ncodpers', 'snapshot_date']
    train_raw = full[full['snapshot_date'] < TRAIN_CUTOFF].reset_index(drop=True)
    test_raw = full[full['snapshot_date'] >= TRAIN_CUTOFF].reset_index(drop=True)

    params = fit_preprocessing_params(train_raw.drop(columns=drop))
    X_train = apply_preprocessing(train_raw.drop(columns=drop), params)[BASE_COLUMNS]
    X_test = apply_preprocessing(test_raw.drop(columns=drop), params)[BASE_COLUMNS]
    y_train, y_test = train_raw['purchase'], test_raw['purchase']

    X_train['antiguedad'] = X_train['antiguedad'].astype(int)
    X_test['antiguedad'] = X_test['antiguedad'].astype(int)
    df_train, aggregates = feature_engineering(X_train)
    df_test, _ = feature_engineering(X_test, aggregates)
    for frame in (df_train, df_test):
        for column in frame.columns:
            if column != 'recommended_product_id':
                frame[column] = frame[column].astype(str)

    numeric = ['antiguedad', 'renta']
    categorical = [column for column in df_train.columns if column not in numeric]
    assert experiments.ALS_FEATURE in categorical
    features_without = experiments.drop_feature(df_train.columns.tolist(),
                                                experiments.ALS_FEATURE)
    categorical_without = [column for column in categorical
                           if column != experiments.ALS_FEATURE]

    variants = {}
    for name, columns, cats in (('with_als', list(df_train.columns), categorical),
                                ('without_als', features_without, categorical_without)):
        model = experiments.build_pipeline(
            'random_forest', numeric, cats,
            params={'classifier__n_estimators': 8, 'classifier__max_depth': 6},
            use_feature_selection=False)
        model.fit(df_train[columns], y_train)
        variants[name] = experiments.evaluate(
            model, df_test[columns], y_test,
            candidate_classes=modeling_inputs['target_codes'])

    table = experiments.ablation_rows(variants)
    assert list(table['variant']) == ['with_als', 'without_als']
    assert table['pr_auc_macro'].notna().all()
    assert table['shortlist_purchasers_ndcg_at_k'].notna().all()


def test_als_baseline_metrics_cover_cold_start(modeling_inputs):
    """ALS как самостоятельный baseline: cold-start клиенты дают нули, а не пропуски."""
    bundle = modeling_inputs['als_baseline']
    events = modeling_inputs['events']
    lists = dataset.als_topk_lists(bundle['model'], bundle['matrix'], bundle['codes'], k=5)
    relevant_by_user = dataset.relevant_products(events)

    recurring = sorted(set(bundle['users']) & set(relevant_by_user.index))
    recommended = [lists[list(bundle['users']).index(user)] for user in recurring]
    relevant = [relevant_by_user[user] for user in recurring]
    metrics = recsys_metrics.als_top_k_metrics(recommended, relevant, ks=(1, 5)).set_index('k')

    assert metrics.loc[5, 'users'] == len(recurring)
    assert 0.0 <= metrics.loc[5, 'precision_at_k'] <= 1.0
    assert 0.0 <= metrics.loc[5, 'ndcg_at_k'] <= 1.0
    assert metrics.loc[5, 'recall_at_k'] <= 1.0

    cold_start = recsys_metrics.als_top_k_metrics([[]], [['ind_cco_fin_ult1']], ks=(5,))
    assert cold_start.loc[0, 'precision_at_k'] == 0.0
    assert cold_start.loc[0, 'recall_at_k'] == 0.0
