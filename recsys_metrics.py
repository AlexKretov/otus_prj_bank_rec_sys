"""Метрики ранжирования рекомендательной системы банковских продуктов.

Модуль держит всю математику оценки качества персональных списков продуктов:
он импортируется ноутбуком обучения (`modeling.ipynb`) и тестами, поэтому каждая
формула описана ровно один раз и не расползается по ячейкам.

Термины
-------
* **Кандидаты** — продукты, которые система может показать клиенту: шорт-лист
  целевых продуктов. Служебные классы ``0`` (покупки не было) и ``99`` (``other``)
  в список рекомендаций не попадают: «ничего не покупать» и «редкий продукт вне
  шорт-листа» нельзя предложить клиенту.
* **Релевантность** — продукт из шорт-листа, который клиент фактически купил
  в целевом периоде. В задаче «следующая покупка» релевантный объект один, поэтому
  ``recall@K`` в однорелевантной постановке совпадает с ``hit_rate@K``
  (попадание, нормированное на размер релевантного множества, равный единице).

Метрики
-------
* ``precision@K = hit@K / K`` — сколько из K показанных продуктов угадано;
* ``recall@K = hit@K / |relevant|`` — доля найденных релевантных продуктов;
* ``ndcg@K = 1 / log2(rank + 1)`` — попадание с учётом позиции продукта в списке
  (для промаха — 0, для попадания на первое место — 1);
* ``hit_rate@K`` — доля строк, где купленный продукт попал в топ-K.

Скоупы
------
Одни и те же метрики считаются в трёх скоупах, потому что они отвечают на разные
вопросы:

* ``shortlist_purchasers`` — клиенты, купившие продукт из шорт-листа. Основной
  скоуп рекомендаций: именно этих клиентов система обязана обслужить точным списком;
* ``purchasers_any`` — все клиенты, купившие хоть что-то. Покупка вне шорт-листа
  считается промахом: систему можно упрекнуть, что она не предложила редкий продукт,
  но выдать его списком она не может по построению;
* ``all_rows`` — все строки теста, включая клиентов без покупки. Показывает, какую
  долю всего потока клиентов система обслуживает попаданием в топ-K.

Бейзлайны
---------
* ``random`` — ожидаемые метрики равномерно случайного списка из M кандидатов
  (считаются аналитически: ``E[hit@K] = K / M`` для релевантной строки);
* ``popularity`` — фиксированный топ-K самых популярных продуктов обучающего
  периода. Для персональных рекомендаций это слабый, но обязательный бассейн
  сравнения: если модель не лучше «показать всем одно и то же», она бесполезна.
"""

from __future__ import annotations

import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

NO_PURCHASE_CLASS = 0
OTHER_CLASS = 99
DEFAULT_KS: Tuple[int, ...] = (1, 3, 5, 10)
DEFAULT_SCOPES: Tuple[str, ...] = ('shortlist_purchasers', 'purchasers_any', 'all_rows')
SCOPE_LABELS = {
    'shortlist_purchasers': 'клиенты с покупкой из шорт-листа',
    'purchasers_any': 'все покупатели (покупка вне шорт-листа — промах)',
    'all_rows': 'все строки теста, включая клиентов без покупки',
}


def _int_array(values: Iterable) -> np.ndarray:
    """Классы и индексы приводим к int64: коды продуктов в проекте целочисленные."""
    return np.asarray(list(values), dtype=np.int64)


def available_candidates(classes: Sequence[int],
                         candidate_classes: Sequence[int]) -> Tuple[np.ndarray, List[int]]:
    """Кандидаты, которые модель вообще может предложить, и потерянные при этом.

    Класс из шорт-листа может отсутствовать среди классов модели (например, в train
    не было ни одной такой покупки). Молча выкинуть его нельзя: метрики станут
    несравнимыми между вариантами, поэтому потерянные кандидаты возвращаются
    вызывающему коду — он обязан их показать.
    """
    lookup = {int(cls): index for index, cls in enumerate(_int_array(classes))}
    available = [int(cls) for cls in candidate_classes if int(cls) in lookup]
    missing = [int(cls) for cls in candidate_classes if int(cls) not in lookup]
    return np.asarray(available, dtype=np.int64), missing


def candidate_columns(classes: Sequence[int], candidate_classes: Sequence[int]) -> np.ndarray:
    """Позиции колонок матрицы вероятностей, отвечающих продуктам-кандидатам."""
    available, _ = available_candidates(classes, candidate_classes)
    if available.size == 0:
        raise ValueError('ни один из кандидатов не входит в классы модели')
    lookup = {int(cls): index for index, cls in enumerate(_int_array(classes))}
    return np.array([lookup[int(cls)] for cls in available], dtype=np.int64)


def _ranks(scores: np.ndarray, classes: Sequence[int],
           candidate_classes: Sequence[int]) -> Tuple[np.ndarray, np.ndarray]:
    """Позиция (1..M) каждого кандидата в персональном списке каждой строки.

    Список — сортировка по убыванию вероятности; ничьи разрешаются стабильно
    (порядок кандидатов), поэтому результат детерминирован и совпадает
    с ``top_k_products``.
    """
    available, _ = available_candidates(classes, candidate_classes)
    columns = candidate_columns(classes, candidate_classes)
    sub_scores = np.asarray(scores, dtype=np.float64)[:, columns]
    order = np.argsort(-sub_scores, axis=1, kind='stable')
    ranks = np.empty_like(order)
    positions = np.tile(np.arange(1, sub_scores.shape[1] + 1), (sub_scores.shape[0], 1))
    np.put_along_axis(ranks, order, positions, axis=1)
    return ranks, available


def _relevant_positions(y_true: np.ndarray, candidate_classes: np.ndarray) -> np.ndarray:
    """Индекс релевантного кандидата в каждой строке (-1, если покупки в шорт-листе нет)."""
    lookup = {int(cls): index for index, cls in enumerate(candidate_classes)}
    return np.array([lookup.get(int(value), -1) for value in y_true], dtype=np.int64)


def _scope_masks(purchased: np.ndarray, relevant: np.ndarray,
                 scopes: Sequence[str]) -> Dict[str, np.ndarray]:
    """Маски строк для каждого скоупа оценки."""
    masks = {
        'shortlist_purchasers': relevant,
        'purchasers_any': purchased,
        'all_rows': np.ones(len(purchased), dtype=bool),
    }
    unknown = [scope for scope in scopes if scope not in masks]
    if unknown:
        raise ValueError(f'неизвестный скоуп: {unknown}; ожидается один из {tuple(masks)}')
    return {scope: masks[scope] for scope in scopes}


def _discounted_gain(ranks: np.ndarray, hits: np.ndarray) -> np.ndarray:
    """1 / log2(rank + 1) для попаданий и 0 для промахов (rank = 0 — нет попадания)."""
    gain = np.zeros(len(ranks), dtype=np.float64)
    np.divide(1.0, np.log2(ranks + 1.0), out=gain, where=hits)
    return gain


def top_k_products(scores: np.ndarray, classes: Sequence[int],
                   candidate_classes: Sequence[int], k: int) -> np.ndarray:
    """Топ-k кодов продуктов для каждой строки (по убыванию вероятности)."""
    available, _ = available_candidates(classes, candidate_classes)
    columns = candidate_columns(classes, candidate_classes)
    sub_scores = np.asarray(scores, dtype=np.float64)[:, columns]
    order = np.argsort(-sub_scores, axis=1, kind='stable')[:, :k]
    return available[order]


def ranking_stats(y_true: Sequence[int], scores: np.ndarray, classes: Sequence[int],
                  candidate_classes: Sequence[int]) -> pd.DataFrame:
    """Построчные «сырьё» метрик: позиция купленного продукта и попадание.

    Возвращает DataFrame с колонками ``rank`` (позиция релевантного продукта,
    ``0`` если покупка вне шорт-листа или её не было) и ``relevant`` (покупка
    продукта из шорт-листа). Вынесено отдельно, чтобы абляции и бенчмарки
    использовали одну и ту же арифметику.
    """
    y_true_arr = _int_array(y_true)
    ranks, candidate_arr = _ranks(scores, classes, candidate_classes)
    relevant = _relevant_positions(y_true_arr, candidate_arr)
    positions = np.clip(relevant, 0, None)
    row_rank = np.where(relevant >= 0,
                        ranks[np.arange(len(y_true_arr)), positions], 0)
    return pd.DataFrame({
        'rank': row_rank.astype(np.int64),
        'relevant': relevant >= 0,
        'purchased': y_true_arr != NO_PURCHASE_CLASS,
    })


def _metrics_table(stats: pd.DataFrame, ks: Sequence[int],
                   scopes: Sequence[str]) -> pd.DataFrame:
    """Метрики по скоупам и k по готовой таблице построчных попаданий."""
    rows: List[Dict[str, object]] = []
    masks = _scope_masks(stats['purchased'].to_numpy(), stats['relevant'].to_numpy(), scopes)

    for scope, mask in masks.items():
        scoped = stats[mask]
        relevance = scoped['relevant'].to_numpy()
        ranks = scoped['rank'].to_numpy()
        for k in ks:
            hits = (ranks > 0) & (ranks <= k)
            hits_float = hits.astype(np.float64)
            # recall@K нормируется на размер релевантного множества; в задаче
            # «следующая покупка» релевантный продукт один, поэтому recall@K = hit@K.
            has_relevant = relevance
            rows.append({
                'scope': scope,
                'k': int(k),
                'users': int(len(scoped)),
                'purchasers': int(has_relevant.sum()),
                'purchase_share': float(has_relevant.mean()) if len(scoped) else float('nan'),
                'precision_at_k': float(hits_float.mean()) / k if len(scoped) else float('nan'),
                'recall_at_k': (float(hits_float[has_relevant].mean())
                                if has_relevant.any() else float('nan')),
                'hit_rate_at_k': float(hits_float.mean()) if len(scoped) else float('nan'),
                'ndcg_at_k': float(_discounted_gain(ranks, hits).mean())
                if len(scoped) else float('nan'),
            })
    return pd.DataFrame(rows)


def random_baseline(y_true: Sequence[int], candidate_classes: Sequence[int],
                    ks: Sequence[int] = DEFAULT_KS,
                    scopes: Sequence[str] = DEFAULT_SCOPES) -> pd.DataFrame:
    """Аналитические метрики равномерно случайного списка из M кандидатов."""
    y_true_arr = _int_array(y_true)
    candidate_arr = _int_array(candidate_classes)
    relevance = _relevant_positions(y_true_arr, candidate_arr) >= 0
    purchased = y_true_arr != NO_PURCHASE_CLASS
    n_candidates = len(candidate_arr)
    rows = []
    for scope, mask in _scope_masks(purchased, relevance, scopes).items():
        relevant_share = float(relevance[mask].mean()) if mask.any() else float('nan')
        for k in ks:
            expected_dcg = float(np.sum(1.0 / np.log2(np.arange(1, k + 1) + 1)) / n_candidates)
            rows.append({
                'scope': scope,
                'k': int(k),
                'random_precision_at_k': relevant_share / n_candidates,
                'random_recall_at_k': float(k) / n_candidates,
                'random_hit_rate_at_k': relevant_share * float(k) / n_candidates,
                'random_ndcg_at_k': relevant_share * expected_dcg,
            })
    # Только scope/k и метрики бейзлайна: служебные колонки модели (users и др.)
    # в отчёт не тащим, иначе merge породит дубли имён с суффиксами _x/_y.
    return pd.DataFrame(rows)


def popularity_ranking(popularity: Dict[int, float],
                       candidate_classes: Sequence[int]) -> List[int]:
    """Фиксированный список кандидатов: от самого популярного к самому редкому."""
    candidate_arr = _int_array(candidate_classes)
    weights = np.array([float(popularity.get(int(cls), 0.0)) for cls in candidate_arr])
    order = np.argsort(-weights, kind='stable')
    return [int(cls) for cls in candidate_arr[order]]


def popularity_baseline(y_true: Sequence[int], candidate_classes: Sequence[int],
                        popularity: Dict[int, float], ks: Sequence[int] = DEFAULT_KS,
                        scopes: Sequence[str] = DEFAULT_SCOPES) -> pd.DataFrame:
    """Метрики бейзлайна «показать всем топ-K популярных продуктов train-периода»."""
    y_true_arr = _int_array(y_true)
    candidate_arr = _int_array(candidate_classes)
    order = popularity_ranking(popularity, candidate_arr)
    rank_of = {int(cls): index + 1 for index, cls in enumerate(order)}
    ranks = np.array([rank_of.get(int(value), 0) for value in y_true_arr], dtype=np.int64)
    stats = pd.DataFrame({
        'rank': ranks,
        'relevant': ranks > 0,
        'purchased': y_true_arr != NO_PURCHASE_CLASS,
    })
    table = _metrics_table(stats, ks, scopes)
    table = table.rename(columns={
        'precision_at_k': 'popularity_precision_at_k',
        'recall_at_k': 'popularity_recall_at_k',
        'hit_rate_at_k': 'popularity_hit_rate_at_k',
        'ndcg_at_k': 'popularity_ndcg_at_k',
    })
    return table.drop(columns=['users', 'purchasers', 'purchase_share'])


def top_k_report(y_true: Sequence[int], scores: np.ndarray, classes: Sequence[int],
                 candidate_classes: Sequence[int], ks: Sequence[int] = DEFAULT_KS,
                 scopes: Sequence[str] = DEFAULT_SCOPES,
                 popularity: Optional[Dict[int, float]] = None) -> pd.DataFrame:
    """Итоговая таблица top-K метрик модели и бейзлайнов (артефакт ``topk_metrics.csv``)."""
    available, missing = available_candidates(classes, candidate_classes)
    if missing:
        warnings.warn('продукты шорт-листа отсутствуют среди классов модели и исключены '
                      f'из рекомендаций: {missing}', RuntimeWarning, stacklevel=2)

    stats = ranking_stats(y_true, scores, classes, candidate_classes)
    table = _metrics_table(stats, ks, scopes)
    # Бейзлайны считаются по тому же набору кандидатов, который доступен модели, —
    # иначе сравнение с random/popularity было бы несопоставимым.
    table = table.merge(random_baseline(y_true, available, ks, scopes),
                        on=['scope', 'k'], how='left')
    if popularity:
        table = table.merge(popularity_baseline(y_true, available, popularity, ks, scopes),
                            on=['scope', 'k'], how='left')
    table['scope_label'] = table['scope'].map(SCOPE_LABELS).fillna(table['scope'])
    table.attrs['candidates_used'] = available.tolist()
    table.attrs['candidates_missing'] = missing
    return table


def als_top_k_metrics(recommended: Sequence[Sequence[int]], relevant: Sequence[Sequence[int]],
                      ks: Sequence[int] = DEFAULT_KS) -> pd.DataFrame:
    """Метрики готовых списков рекомендаций (ALS как самостоятельная система).

    ``recommended`` — список списков рекомендованных продуктов на клиента,
    ``relevant`` — список множеств реально купленных продуктов на клиента (индекс
    в обоих списках — один и тот же клиент). Пустой список рекомендаций (cold start,
    у клиента нет истории) даёт нулевые метрики, а не пропуск: клиент всё равно
    входит в скоуп оценки.
    """
    if len(recommended) != len(relevant):
        raise ValueError('рекомендации и релевантные продукты должны быть одной длины')

    n_users = len(recommended)
    positions: List[int] = []
    sizes: List[int] = []
    for recs, rel in zip(recommended, relevant):
        # Сравниваем элементы как есть: ALS-рекомендации в проекте приходят
        # в именах продуктов, а не в кодах классов.
        rel_set = set(rel)
        sizes.append(len(rel_set))
        rank = 0
        for position, item in enumerate(recs, start=1):
            if item in rel_set:
                rank = position
                break
        positions.append(rank)
    positions_arr = np.array(positions, dtype=np.int64)
    sizes_arr = np.array(sizes, dtype=np.int64)
    has_relevant = sizes_arr > 0

    rows = []
    for k in ks:
        hits = (positions_arr > 0) & (positions_arr <= k)
        hits_float = hits.astype(np.float64)
        rows.append({
            'k': int(k),
            'users': n_users,
            'purchasers': int(has_relevant.sum()),
            'purchase_share': float(has_relevant.mean()) if n_users else float('nan'),
            'precision_at_k': float(hits_float.mean()) / k if n_users else float('nan'),
            'recall_at_k': (float((hits_float[has_relevant] / sizes_arr[has_relevant]).mean())
                            if has_relevant.any() else float('nan')),
            'hit_rate_at_k': float(hits_float.mean()) if n_users else float('nan'),
            'ndcg_at_k': float(_discounted_gain(positions_arr, hits).mean())
            if n_users else float('nan'),
            'relevant_per_user': float(sizes_arr.mean()) if n_users else float('nan'),
        })
    return pd.DataFrame(rows)


def top_k_lists(recommended: Sequence[Sequence[int]],
                relevant: Sequence[Sequence[int]], k: int) -> pd.DataFrame:
    """Построчная расшифровка попаданий для ручной проверки примеров."""
    rows = []
    for index, (recs, rel) in enumerate(zip(recommended, relevant)):
        rel_set = set(rel)
        top = [int(item) for item in list(recs)[:k]]
        rows.append({
            'row': index,
            'top_k': top,
            'relevant': sorted(rel_set),
            'hit': bool(set(top) & rel_set),
        })
    return pd.DataFrame(rows)
