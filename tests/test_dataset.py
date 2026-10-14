"""Тесты построения датасета: события покупок, таргет панели и ALS-признак.

Главная проверка здесь — регрессионная для замечания про целевую переменную:
метку «покупки не было» нельзя получить клиенту, который в целевом снимке
покупку совершил, независимо от того, есть ли у него история для ALS.
"""

import numpy as np
import pandas as pd
import pytest

import dataset
from preprocessing import OTHER_CLASS, PRODUCTS
from tests.conftest import make_raw_panel

CUTOFF = pd.Timestamp('2015-04-30')


def _tiny_panel() -> pd.DataFrame:
    """Ручная панель: покупка в третьем снимке, «отвал» клиента, cold start."""
    rows = []
    base = {
        'ind_empleado': 'N', 'pais_residencia': 'ES', 'sexo': 'H', 'age': 40.0,
        'ind_nuevo': 0, 'antiguedad': 100.0, 'indrel_1mes': '1.0', 'tiprel_1mes': 'I',
        'indresi': 'S', 'conyuemp': 'N', 'canal_entrada': 'KHE', 'indfall': 'N',
        'cod_prov': 28.0, 'ind_actividad_cliente': 1.0, 'renta': 100000.0,
        'segmento': '02 - PARTICULARES', 'ult_fec_cli_1t': None,
        **{product: 0 for product in PRODUCTS},
    }
    snapshots = pd.to_datetime(['2015-01-28', '2015-02-28', '2015-03-28'])
    # Клиент 1: держит продукт 3, в феврале покупает продукт 5.
    # Клиент 2: появляется только в феврале (cold start — истории для ALS нет)
    #           и покупает продукт 3 в марте: это и есть «покупка без истории».
    # Клиент 3: уходит после января.
    states = {
        1: [(0, {'ind_cco_fin_ult1': 1}), (1, {'ind_cco_fin_ult1': 1, 'ind_cno_fin_ult1': 1}),
            (2, {'ind_cco_fin_ult1': 1, 'ind_cno_fin_ult1': 1})],
        2: [(1, {'ind_ctop_fin_ult1': 1}),
            (2, {'ind_ctop_fin_ult1': 1, 'ind_cco_fin_ult1': 1})],
        3: [(0, {'ind_cco_fin_ult1': 1})],
    }
    for client, snapshots_state in states.items():
        for snapshot_index, products in snapshots_state:
            row = dict(base)
            row.update(products)
            row.update({'ncodpers': client, 'fecha_dato': snapshots[snapshot_index],
                        'fecha_alta': '2014-01-01'})
            rows.append(row)
    return pd.DataFrame(rows)


def test_purchase_events_marks_only_new_products():
    frame = _tiny_panel()
    events = dataset.purchase_events(frame)
    pairs = set(zip(events['ncodpers'], events['fecha_dato'].dt.strftime('%Y-%m-%d'),
                    events['purchase']))
    # Продукт 3 в первом снимке клиента считается покупкой (конвенция датасета),
    # продукт 5 — покупка февраля, продукт 8 — покупка клиента без истории.
    assert (1, '2015-01-28', 3) in pairs
    assert (1, '2015-02-28', 5) in pairs
    assert (2, '2015-03-28', 3) in pairs
    # Повторяющееся владение продуктом покупкой не считается.
    assert (1, '2015-03-28', 3) not in pairs
    assert (3, '2015-02-28', 3) not in pairs


def test_panel_target_labels_every_purchaser():
    """Замечание №1: покупка клиента из следующего снимка обязана стать меткой класса.

    Клиент 2 появился только в феврале (у него нет истории для ALS) — его покупка
    всё равно должна попасть в таргет, а не превратиться в «покупки не было».
    """
    frame = _tiny_panel()
    events = dataset.purchase_events(frame)
    panel = dataset.build_target_panel(frame, events,
                                       first_snapshot='2015-01-28',
                                       last_snapshot='2015-02-28')

    assert set(panel.panel['snapshot_date']) == {pd.Timestamp('2015-01-28'),
                                                 pd.Timestamp('2015-02-28')}
    january = panel.panel[panel.panel['snapshot_date'] == pd.Timestamp('2015-01-28')]
    february = panel.panel[panel.panel['snapshot_date'] == pd.Timestamp('2015-02-28')]

    # Январский снимок: клиент 1 покупает продукт 5, клиент 3 уходит — метки нет.
    assert january.set_index('ncodpers')['purchase'].to_dict() == {1: 5, 3: 0}
    # Февральский снимок: продукт 3 купил cold-start клиент 2 — он в панели и с меткой
    # покупки, а не с нулём (регрессия из замечания №1).
    assert february.set_index('ncodpers')['purchase'].to_dict() == {1: 0, 2: 3}


def test_panel_diagnostics_report_unpredictable_purchasers():
    """Покупатель, которого нет в предыдущем снимке, виден в диагностике, но не в панели."""
    frame = _tiny_panel()
    events = dataset.purchase_events(frame)
    panel = dataset.build_target_panel(frame, events,
                                       first_snapshot='2015-01-28',
                                       last_snapshot='2015-01-28')
    diagnostics = panel.diagnostics.iloc[0]
    # В феврале покупали клиенты 1 и 2; клиента 2 в январе нет.
    assert diagnostics['purchasers_in_target'] == 2
    assert diagnostics['purchasers_in_snapshot'] == 1
    assert diagnostics['purchasers_missing_in_snapshot'] == 1
    assert panel.summary['unpredictable_purchaser_share'] == pytest.approx(0.5)


def test_panel_labels_match_events_for_every_snapshot():
    """Инвариант: метка класса есть ⟺ клиент в снимке и купил в следующем снимке."""
    frame = make_raw_panel(n_clients=80, n_snapshots=4, seed=11)
    frame['fecha_dato'] = pd.to_datetime(frame['fecha_dato'])
    events = dataset.purchase_events(frame)
    panel = dataset.build_target_panel(frame, events)

    first_purchase = (events.sort_values(['ncodpers', 'purchase'])
                      .drop_duplicates(['ncodpers', 'fecha_dato'], keep='first')
                      .set_index(['ncodpers', 'fecha_dato'])['purchase'])
    for row in panel.panel[panel.panel['purchase'] != 0].itertuples():
        target_date = panel.diagnostics.set_index('snapshot_date').loc[
            row.snapshot_date, 'target_date']
        assert first_purchase[(row.ncodpers, target_date)] == row.purchase

    # Ни один покупатель целевого снимка, присутствующий в снимке, не получил 0.
    for row in panel.diagnostics.itertuples():
        snapshot_users = set(panel.panel.loc[panel.panel['snapshot_date'] == row.snapshot_date,
                                             'ncodpers'])
        purchasers = set(events.loc[events['fecha_dato'] == row.target_date, 'ncodpers'])
        labelled = set(panel.panel.loc[(panel.panel['snapshot_date'] == row.snapshot_date)
                                       & (panel.panel['purchase'] != 0), 'ncodpers'])
        assert labelled == purchasers & snapshot_users


def test_shortlist_and_label_map_are_built_from_history():
    events = pd.DataFrame({
        'fecha_dato': pd.to_datetime(['2015-01-28'] * 100),
        'purchase': [3] * 70 + [5] * 20 + [7] * 5 + [24] * 5,
    })
    share = dataset.purchase_share(events, CUTOFF)
    codes, dropped = dataset.select_target_products(share, min_share=0.2, max_products=1)
    assert codes == [3]
    assert dropped == [5]

    label_map = dataset.build_label_map([3, 5])
    assert label_map[0] == 'no_purchase'
    assert label_map[3] == PRODUCTS[2]
    assert label_map[OTHER_CLASS] == 'other'


def test_fold_rare_targets_counts_support_on_train_only():
    """Редкий класс сворачивается по статистике train: тест не влияет на состав классов."""
    labels = pd.Series([3] * 50 + [5] * 5 + [7] * 40)
    train_mask = np.array([True] * 55 + [False] * 40)
    folded, rare = dataset.fold_rare_targets(labels, train_mask, min_support=10)
    assert rare == [5]
    assert folded.iloc[50] == OTHER_CLASS
    assert (folded.iloc[55:] == 7).all()


def test_als_feature_lookup_uses_only_past_history():
    """Признак ALS на снимке строится по истории строго до снимка и не предлагает своё."""
    frame = make_raw_panel(n_clients=120, n_snapshots=4, seed=5)
    frame['fecha_dato'] = pd.to_datetime(frame['fecha_dato'])
    events = dataset.purchase_events(frame)
    users, user_map = dataset.user_index(frame['ncodpers'])

    feature = dataset.als_feature_lookup(frame, events, user_map, factors=6, iterations=3,
                                         refit_every=1)
    lookup = feature.lookup
    first_snapshot = frame['fecha_dato'].min()
    later_snapshots = set(frame['fecha_dato'].unique()) - {first_snapshot}
    # На первом снимке истории ещё нет: модель не строится, строк для него нет.
    assert first_snapshot not in feature.models
    assert set(lookup['snapshot_date']) == later_snapshots
    assert len(feature.models) == len(later_snapshots)
    assert lookup['recommended_product_id'].notna().all()

    merged = lookup.merge(
        frame[['ncodpers', 'fecha_dato', *PRODUCTS]].rename(
            columns={'fecha_dato': 'snapshot_date'}),
        on=['ncodpers', 'snapshot_date'], how='left')
    owned_recommended = [getattr(row, row.recommended_product_id)
                         for row in merged.itertuples()]
    assert sum(owned_recommended) == 0, 'ALS предложил продукт, который уже есть у клиента'

    latest = feature.latest_lookup()
    assert latest.index.name == 'ncodpers'
    assert latest.index.is_unique
    assert latest.shape[0] == frame.loc[frame['fecha_dato'] == frame['fecha_dato'].max(),
                                        'ncodpers'].nunique()


def test_als_topk_lists_decode_product_names():
    frame = make_raw_panel(n_clients=60, n_snapshots=3, seed=9)
    frame['fecha_dato'] = pd.to_datetime(frame['fecha_dato'])
    events = dataset.purchase_events(frame)
    users, user_map = dataset.user_index(frame['ncodpers'])
    history = events[events['fecha_dato'] < frame['fecha_dato'].max()]
    matrix = dataset.als_user_item(history, user_map)
    model = dataset.fit_als(matrix, factors=6, iterations=3)
    codes = np.array([user_map[int(user)] for user in pd.unique(history['ncodpers'])])

    lists = dataset.als_topk_lists(model, matrix, codes[:5], k=3)
    assert len(lists) == 5
    for names in lists:
        assert len(names) == 3
        assert all(name in PRODUCTS for name in names)


def test_product_name_rejects_service_codes():
    assert dataset.product_name(3) == PRODUCTS[2]
    assert dataset.product_name(0) is None
    assert dataset.product_name(OTHER_CLASS) is None
