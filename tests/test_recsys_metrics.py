"""Тесты метрик ранжирования `recsys_metrics`.

Проверяем формулы на игрушечных примерах, где правильный ответ считается в уме:
идеальный список, попадание на второе место, разбавление метрик клиентами без
покупок, аналитический random-бейзлайн и нулевые метрики cold-start для ALS.
"""

import numpy as np
import pandas as pd
import pytest

import recsys_metrics as metrics

CLASSES = [0, 3, 5, 8, 99]
CANDIDATES = [3, 5, 8]


def _scores(rows):
    """Матрица вероятностей: строки задаются словарём {класс: вероятность}."""
    return np.array([[row.get(cls, 0.0) for cls in CLASSES] for row in rows])


def test_perfect_ranking_gives_unit_metrics():
    y_true = [3, 5, 8]
    scores = _scores([
        {3: 0.9, 5: 0.1},
        {5: 0.9, 3: 0.1},
        {8: 0.9, 3: 0.1},
    ])
    report = metrics.top_k_report(y_true, scores, CLASSES, CANDIDATES, ks=(1, 3))
    scope = report[report['scope'] == 'shortlist_purchasers'].set_index('k')
    assert scope.loc[1, 'precision_at_k'] == pytest.approx(1.0)
    assert scope.loc[1, 'recall_at_k'] == pytest.approx(1.0)
    assert scope.loc[1, 'ndcg_at_k'] == pytest.approx(1.0)
    assert scope.loc[3, 'ndcg_at_k'] == pytest.approx(1.0)


def test_ndcg_penalises_second_position():
    """Купленный продукт на втором месте: precision@2 = 0.5, NDCG@2 = 1/log2(3)."""
    y_true = [5]
    scores = _scores([{3: 0.7, 5: 0.3, 8: 0.0}])
    report = metrics.top_k_report(y_true, scores, CLASSES, CANDIDATES, ks=(1, 2))
    scope = report[report['scope'] == 'shortlist_purchasers'].set_index('k')
    assert scope.loc[1, 'precision_at_k'] == pytest.approx(0.0)
    assert scope.loc[1, 'ndcg_at_k'] == pytest.approx(0.0)
    assert scope.loc[2, 'precision_at_k'] == pytest.approx(0.5)
    assert scope.loc[2, 'recall_at_k'] == pytest.approx(1.0)
    assert scope.loc[2, 'ndcg_at_k'] == pytest.approx(1.0 / np.log2(3))


def test_no_purchase_rows_dilute_all_rows_scope_only():
    """Клиент без покупки портит метрики скоупа all_rows и не попадает в purchasers-скоупы."""
    y_true = [5, 0]
    scores = _scores([{5: 0.9}, {5: 0.9}])
    report = metrics.top_k_report(y_true, scores, CLASSES, CANDIDATES, ks=(1,))
    by_scope = report.set_index('scope')

    assert by_scope.loc['shortlist_purchasers', 'users'] == 1
    assert by_scope.loc['shortlist_purchasers', 'precision_at_k'] == pytest.approx(1.0)
    assert by_scope.loc['purchasers_any', 'users'] == 1
    assert by_scope.loc['all_rows', 'users'] == 2
    assert by_scope.loc['all_rows', 'precision_at_k'] == pytest.approx(0.5)
    assert by_scope.loc['all_rows', 'recall_at_k'] == pytest.approx(1.0)


def test_purchase_outside_shortlist_is_miss_for_purchasers_scope():
    """Покупка редкого продукта (`other`) в скоупе покупателей считается промахом."""
    y_true = [99, 3]
    scores = _scores([{3: 0.9}, {3: 0.9}])
    report = metrics.top_k_report(y_true, scores, CLASSES, CANDIDATES, ks=(1,))
    by_scope = report.set_index('scope')

    assert by_scope.loc['purchasers_any', 'users'] == 2
    assert by_scope.loc['purchasers_any', 'precision_at_k'] == pytest.approx(0.5)
    assert by_scope.loc['purchasers_any', 'recall_at_k'] == pytest.approx(1.0)
    assert by_scope.loc['shortlist_purchasers', 'users'] == 1
    # В скоупе покупателей шорт-листа доля покупок по построению равна 1,
    # а доля покупок шорт-листа по всем строкам — 0.5.
    assert by_scope.loc['shortlist_purchasers', 'purchase_share'] == pytest.approx(1.0)
    assert by_scope.loc['all_rows', 'purchase_share'] == pytest.approx(0.5)


def test_ties_are_broken_by_candidate_order():
    """Ничьи разрешаются детерминированно: раньше в шорт-листе — выше в списке."""
    scores = _scores([{3: 0.5, 5: 0.5}, {3: 0.5, 5: 0.5, 8: 0.5}])
    top = metrics.top_k_products(scores, CLASSES, CANDIDATES, k=3)
    assert top.tolist() == [[3, 5, 8], [3, 5, 8]]


def test_random_baseline_matches_analytic_expectation():
    """Random-бейзлайн: E[precision@K] = purchase_share / M, E[recall@K] = K / M."""
    y_true = [3, 5, 0, 0]
    report = metrics.random_baseline(y_true, CANDIDATES, ks=(1, 3))
    scope = report[report['scope'] == 'all_rows'].set_index('k')
    purchase_share = 0.5
    assert scope.loc[1, 'random_precision_at_k'] == pytest.approx(purchase_share / 3)
    assert scope.loc[3, 'random_precision_at_k'] == pytest.approx(purchase_share / 3)
    assert scope.loc[3, 'random_recall_at_k'] == pytest.approx(1.0)
    assert scope.loc[1, 'random_hit_rate_at_k'] == pytest.approx(purchase_share / 3)


def test_popularity_baseline_uses_fixed_order():
    """Популярностный бейзлайн ранжирует кандидатов один раз для всех клиентов."""
    popularity = {5: 0.6, 3: 0.3, 8: 0.1}
    order = metrics.popularity_ranking(popularity, CANDIDATES)
    assert order == [5, 3, 8]

    y_true = [5, 8, 0]
    scores = _scores([{5: 0.9}, {5: 0.9}, {5: 0.9}])
    report = metrics.top_k_report(y_true, scores, CLASSES, CANDIDATES, ks=(1, 2),
                                  popularity=popularity)
    scope = report[report['scope'] == 'all_rows'].set_index('k')
    # Популярностный список — [5, 3, 8]. В топ-1 попадает только первый клиент.
    assert scope.loc[1, 'popularity_precision_at_k'] == pytest.approx(1 / 3)
    # В топ-2 попадает [5, 3]: первый клиент угадан, второй (купил 8) — нет.
    assert scope.loc[2, 'popularity_precision_at_k'] == pytest.approx((1 / 2) / 3)
    # Класс 8 стоит третьим, в топ-2 он не попадает даже у купившего его клиента.
    assert scope.loc[2, 'popularity_recall_at_k'] == pytest.approx(1 / 2)


def test_top_k_report_has_labels_and_all_baselines():
    y_true = [3, 0]
    scores = _scores([{3: 0.9}, {3: 0.9}])
    report = metrics.top_k_report(y_true, scores, CLASSES, CANDIDATES, ks=(1,),
                                  popularity={3: 1.0, 5: 0.0, 8: 0.0})
    assert {'random_precision_at_k', 'popularity_precision_at_k', 'scope_label'} <= set(report)
    assert report['scope_label'].notna().all()
    # Слияние с бейзлайнами не должно плодить колонки-дубли с суффиксами _x/_y:
    # иначе отчёт и артефакт теряют колонку users.
    assert 'users' in report.columns
    assert not [column for column in report.columns
                if column.endswith(('_x', '_y'))]


def test_als_metrics_recall_is_normalised_by_relevant_size():
    """Для ALS релевантных продуктов может быть несколько — recall делится на их число."""
    recommended = [[3, 5], [8, 3], []]
    relevant = [[3], [8, 5], []]
    report = metrics.als_top_k_metrics(recommended, relevant, ks=(1, 2)).set_index('k')

    # precision@1: оба первых места (3 и 8) релевантны, третий клиент — cold start.
    assert report.loc[1, 'precision_at_k'] == pytest.approx(2 / 3)
    assert report.loc[2, 'precision_at_k'] == pytest.approx((1 / 2 + 1 / 2 + 0) / 3)
    # recall@2: первый клиент нашёл 1 из 1, второй — 1 из 2, у третьего нет корзины.
    assert report.loc[2, 'recall_at_k'] == pytest.approx((1.0 + 0.5) / 2)
    # hit_rate@2 = 2 из 3 клиентов.
    assert report.loc[2, 'hit_rate_at_k'] == pytest.approx(2 / 3)
    # средний размер корзины: (1 + 2 + 0) / 3.
    assert report.loc[2, 'relevant_per_user'] == pytest.approx(1.0)


def test_als_metrics_cold_start_rows_are_zeros_not_nans():
    report = metrics.als_top_k_metrics([[], []], [[3], []], ks=(1,))
    assert report.loc[0, 'precision_at_k'] == 0.0
    assert report.loc[0, 'hit_rate_at_k'] == 0.0
    assert report.loc[0, 'purchasers'] == 1


def test_als_metrics_length_mismatch_raises():
    with pytest.raises(ValueError):
        metrics.als_top_k_metrics([[3]], [[3], [5]])


def test_unknown_scope_raises():
    scores = _scores([{3: 0.9}])
    with pytest.raises(ValueError):
        metrics.top_k_report([3], scores, CLASSES, CANDIDATES, ks=(1,), scopes=('unknown',))


def test_ranking_stats_matches_report():
    """`ranking_stats` — то же «сырьё», на котором считаются метрики отчёта."""
    y_true = [5, 0, 99]
    scores = _scores([{5: 0.9}, {3: 0.9}, {3: 0.9}])
    stats = metrics.ranking_stats(y_true, scores, CLASSES, CANDIDATES)
    assert isinstance(stats, pd.DataFrame)
    assert stats['rank'].tolist() == [1, 0, 0]
    assert stats['relevant'].tolist() == [True, False, False]
    assert stats['purchased'].tolist() == [True, False, True]


def test_missing_candidates_are_reported_not_silently_dropped():
    """Класс шорт-листа, которого нет среди классов модели, виден вызывающему коду."""
    y_true = [3, 0]
    scores = _scores([{3: 0.9}, {3: 0.9}])
    with pytest.warns(RuntimeWarning, match='отсутствуют среди классов модели'):
        report = metrics.top_k_report(y_true, scores, CLASSES, [3, 5, 12], ks=(1,))
    assert report.attrs['candidates_missing'] == [12]
    assert report.attrs['candidates_used'] == [3, 5]
    # Random-бейзлайн считался по доступным кандидатам (M = 2), а не по трём.
    scope = report[report['scope'] == 'all_rows'].set_index('k')
    assert scope.loc[1, 'random_precision_at_k'] == pytest.approx(0.5 / 2)


def test_all_candidates_missing_raises():
    with pytest.raises(ValueError):
        metrics.ranking_stats([3], _scores([{3: 0.9}]), [0, 3], [99])
