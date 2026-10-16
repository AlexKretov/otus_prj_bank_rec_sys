"""Построение обучающего датасета: события покупок, целевая переменная, ALS-признак.

Модуль отвечает за всё, что стоит между сырым `train_ver2.csv` и матрицей признаков:
он используется ноутбуком `modeling.ipynb` и тестами, поэтому правила построения
таргета зафиксированы в одном месте и не зависят от того, какие клиенты попали
в ALS-историю.

Ключевые правила
----------------
1. **Целевая переменная строится по всем событиям покупок.** Событие — момент,
   когда продукт появился у клиента впервые (``0 → 1`` между снимками). Таргет
   панели — покупка клиента в *следующем* снимке. Метка «покупки не было» ставится
   только тем клиентам, у которых покупки в целевом снимке действительно нет;
   наличие или отсутствие истории для ALS на таргет не влияет.
2. **Панель — все клиенты снимка.** Строка датасета — клиент в снимке `t`,
   его признаки — состояние на `t`, таргет — покупка в `t+1`. Клиенты без истории
   в ALS получают признак-заглушку `0`, но остаются в выборке и получают метку.
3. **Шорт-лист продуктов** строится по покупкам обучающего периода (без подглядывания
   в целевой период) и фиксирует пространство классов; покупки вне шорт-листа
   сворачиваются в класс `other`.

Диагностика (`TargetPanel.summary`) считает, сколько покупателей целевого снимка
вообще отсутствуют в предыдущем снимке (новые клиенты и вернувшиеся) — их покупку
нельзя предсказать по панели, и это число нужно видеть при чтении метрик.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import scipy.sparse

from preprocessing import NO_PURCHASE_CLASS, OTHER_CLASS, PRODUCTS

ALS_FEATURE_NAME = 'recommended_product_id'


def product_name(code: int) -> Optional[str]:
    """Код продукта (позиция в ``PRODUCTS`` + 1) → имя продукта."""
    code = int(code)
    if 1 <= code <= len(PRODUCTS):
        return PRODUCTS[code - 1]
    return None


def purchase_events(df: pd.DataFrame, products: Sequence[str] = PRODUCTS,
                    user_column: str = 'ncodpers',
                    date_column: str = 'fecha_dato') -> pd.DataFrame:
    """События покупок: продукт появился у клиента впервые (``0 → 1``).

    Возвращает таблицу ``ncodpers | fecha_dato | purchase`` по **всем** клиентам
    и продуктам: фильтрация по истории ALS здесь не делается, иначе часть
    покупателей потеряла бы метку.
    """
    sorted_df = df.sort_values([user_column, date_column])
    previous = sorted_df.groupby(user_column)[list(products)].shift(1).fillna(0)
    current = sorted_df[list(products)].fillna(0).to_numpy(dtype='int8')
    previous_arr = previous.to_numpy(dtype='int8')
    new_purchase = (current == 1) & (previous_arr == 0)

    rows, cols = np.nonzero(new_purchase)
    return pd.DataFrame({
        user_column: sorted_df[user_column].to_numpy()[rows],
        date_column: sorted_df[date_column].to_numpy()[rows],
        'purchase': (cols + 1).astype(int),  # код продукта = позиция в PRODUCTS + 1
    })


def purchase_share(events: pd.DataFrame, until: pd.Timestamp) -> pd.Series:
    """Доля покупок по продуктам за период обучения (строго до ``until``)."""
    history = events[events['fecha_dato'] <= pd.Timestamp(until)]
    return history['purchase'].value_counts(normalize=True)


def select_target_products(share: pd.Series, *, min_share: float = 0.01,
                           max_products: int = 12) -> Tuple[List[int], List[int]]:
    """Шорт-лист классов: продукты с долей покупок не ниже порога.

    Возвращает (коды шорт-листа, коды, отброшенные ограничением ``max_products``).
    Порог и ограничение считаются по обучающему периоду: состав классов фиксируется
    до того, как модель увидит целевой период.
    """
    candidates = share[share >= min_share]
    codes = sorted(int(code) for code in candidates.index[:max_products])
    dropped = [int(code) for code in candidates.index[max_products:]]
    return codes, dropped


def build_label_map(target_codes: Iterable[int],
                    products: Sequence[str] = PRODUCTS) -> Dict[int, str]:
    """Расшифровка классов модели: ``0`` — покупки нет, ``99`` — продукт вне шорт-листа."""
    label_map: Dict[int, str] = {NO_PURCHASE_CLASS: 'no_purchase'}
    label_map.update({int(code): products[int(code) - 1] for code in target_codes})
    label_map[OTHER_CLASS] = 'other'
    return label_map


def fold_rare_targets(labels: pd.Series, reference_mask: np.ndarray, *,
                      min_support: int, other_class: int = OTHER_CLASS
                      ) -> Tuple[pd.Series, List[int]]:
    """Сворачивает редкие классы в ``other`` по статистике обучающей части.

    Считаем support только на train (`reference_mask`): иначе состав классов
    подсказывался бы тестовой выборкой.
    """
    labels = labels.copy()
    counts = labels[reference_mask].value_counts()
    protected = [NO_PURCHASE_CLASS, other_class]
    rare_mask = (counts < min_support) & (~counts.index.isin(protected))
    rare = [int(code) for code in counts[rare_mask].index]
    if rare:
        labels = labels.where(~labels.isin(rare), other_class)
    return labels, sorted(rare)


@dataclass
class TargetPanel:
    """Панель «признаки на снимке `t` → покупка в снимке `t+1`»."""

    panel: pd.DataFrame
    diagnostics: pd.DataFrame

    @property
    def summary(self) -> Dict[str, float]:
        return {
            'snapshots': int(len(self.diagnostics)),
            'rows': int(len(self.panel)),
            'clients': int(self.panel['ncodpers'].nunique()),
            'purchasers': int((self.panel['purchase'] != NO_PURCHASE_CLASS).sum()),
            'purchase_share': float((self.panel['purchase'] != NO_PURCHASE_CLASS).mean()),
            # Покупатели целевого снимка, которых нет в предыдущем снимке:
            # их покупку панель предсказать не может (новые/вернувшиеся клиенты).
            'unpredictable_purchasers': int(
                self.diagnostics['purchasers_missing_in_snapshot'].sum()),
            'unpredictable_purchaser_share': float(
                self.diagnostics['purchasers_missing_in_snapshot'].sum()
                / max(self.diagnostics['purchasers_in_target'].sum(), 1)),
        }


def build_target_panel(df: pd.DataFrame, events: pd.DataFrame, *,
                       first_snapshot: Optional[pd.Timestamp] = None,
                       last_snapshot: Optional[pd.Timestamp] = None,
                       date_column: str = 'fecha_dato',
                       user_column: str = 'ncodpers') -> TargetPanel:
    """Собирает панель по всем снимкам окна: таргет — покупка в следующем снимке.

    Для каждого покупателя целевого снимка берётся минимальный код продукта
    (детерминированное правило при нескольких покупках в одном месяце). Клиенты,
    которых нет среди покупателей целевого снимка, получают метку «покупки не было» —
    но только если они действительно есть в снимке `t`: покупки вернувшихся клиентов
    посчитать нельзя, и они видны в диагностике.
    """
    dates = sorted(pd.to_datetime(df[date_column].unique()))
    # last_snapshot — последний снимок-строка панели; таргет для него берётся из
    # следующего снимка датасета, даже если тот выходит за границу окна.
    next_date = {current: following for current, following in zip(dates[:-1], dates[1:])}
    row_dates = [date for date in dates
                 if (first_snapshot is None or date >= pd.Timestamp(first_snapshot))
                 and (last_snapshot is None or date <= pd.Timestamp(last_snapshot))]
    pairs = [(date, next_date[date]) for date in row_dates if date in next_date]
    if not pairs:
        raise ValueError('не нашлось ни одной пары снимков для построения таргета')

    frames: List[pd.DataFrame] = []
    diagnostics: List[Dict[str, object]] = []
    for snapshot, target_date in pairs:
        snapshot_df = df[df[date_column] == snapshot].copy()
        if snapshot_df.empty:
            continue
        target_events = events[events[date_column] == target_date]
        # Первая покупка месяца: минимальный код продукта.
        first_purchase = (target_events.sort_values([user_column, 'purchase'])
                          .drop_duplicates(user_column, keep='first')
                          .set_index(user_column)['purchase'])
        snapshot_df['purchase'] = (snapshot_df[user_column].map(first_purchase)
                                   .fillna(NO_PURCHASE_CLASS).astype(int))
        snapshot_df['snapshot_date'] = snapshot
        frames.append(snapshot_df)

        snapshot_users = set(snapshot_df[user_column])
        target_users = set(first_purchase.index)
        diagnostics.append({
            'snapshot_date': snapshot,
            'target_date': target_date,
            'rows': int(len(snapshot_df)),
            'clients': int(snapshot_df[user_column].nunique()),
            'purchasers_in_target': int(len(target_users)),
            'purchasers_in_snapshot': int((snapshot_df['purchase'] != NO_PURCHASE_CLASS).sum()),
            'purchasers_missing_in_snapshot': int(len(target_users - snapshot_users)),
            'purchase_share': float((snapshot_df['purchase'] != NO_PURCHASE_CLASS).mean()),
        })

    if not frames:
        raise ValueError('панель пуста: проверьте окно снимков')

    panel = pd.concat(frames, ignore_index=True)
    return TargetPanel(panel=panel, diagnostics=pd.DataFrame(diagnostics))


def user_index(ids: Iterable[int]) -> Tuple[np.ndarray, Dict[int, int]]:
    """Уникальные клиенты → коды строк ALS-матрицы и обратный словарь."""
    unique = pd.unique(np.asarray(list(ids)))
    mapping = {int(user): index for index, user in enumerate(unique)}
    return np.asarray(unique, dtype=np.int64), mapping


def als_user_item(events: pd.DataFrame, user_map: Dict[int, int], *,
                  products_n: int = len(PRODUCTS),
                  user_column: str = 'ncodpers') -> scipy.sparse.csr_matrix:
    """Разреженная матрица «клиент × продукт» для ALS.

    Колонка = код продукта (1..24, колонка 0 пустая), строка = код клиента из
    ``user_map``: тот же индекс передаётся в ``ALS.recommend``, поэтому клиент
    и его персональная рекомендация сопоставлены по построению.
    """
    rows = np.array([user_map[int(user)] for user in events[user_column]], dtype=np.int64)
    cols = events['purchase'].to_numpy(dtype=np.int64)
    shape = (len(user_map), products_n + 1)
    return scipy.sparse.csr_matrix((np.ones(len(events), dtype=np.int8), (rows, cols)),
                                   shape=shape, dtype=np.int8)


def fit_als(user_item: scipy.sparse.csr_matrix, *, factors: int = 50, iterations: int = 50,
            regularization: float = 0.05, random_state: int = 0):
    """Обучение ALS на истории покупок (импорт implicit — только здесь)."""
    from implicit.als import AlternatingLeastSquares

    model = AlternatingLeastSquares(factors=factors, iterations=iterations,
                                    regularization=regularization,
                                    random_state=random_state)
    model.fit(user_item)
    return model


def top1_from_factors(model, user_codes: np.ndarray,
                      owned: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Top-1 ALS-рекомендация для клиентов кодами: векторная реализация.

    Повторяет семантику ``implicit.ALS.recommend(..., filter_already_liked_items=True)``
    (уже имеющиеся продукты исключаются, служебная колонка 0 не участвует),
    но считается одним матричным умножением на всех клиентов снимка, а не циклом
    по пользователям: на масштабе миллиона клиентов это разница в минутах и секундах.
    """
    user_factors = np.asarray(model.user_factors, dtype=np.float32)[user_codes]
    item_factors = np.asarray(model.item_factors, dtype=np.float32)
    scores = user_factors @ item_factors.T
    if owned is not None:
        scores = np.where(owned > 0, -np.inf, scores)
    scores[:, 0] = -np.inf  # колонка 0 — служебная (кодов продуктов с нулём не бывает)
    best = np.argmax(scores, axis=1).astype(np.int64)
    return best, scores[np.arange(len(best)), best].astype(np.float64)


def owned_matrix(snapshot_df: pd.DataFrame,
                 products: Sequence[str] = PRODUCTS) -> np.ndarray:
    """Матрица «клиент × продукт» на снимке: 1, если продукт уже есть у клиента.

    Колонки выровнены с кодами продуктов (колонка 0 — служебная, всегда 0),
    поэтому матрица совместима с кодами из ``top1_from_factors``.
    """
    flags = snapshot_df[list(products)].fillna(0).to_numpy(dtype='int8')
    zeros = np.zeros((len(flags), 1), dtype='int8')
    return np.concatenate([zeros, flags], axis=1)


@dataclass
class AlsFeature:
    """Признак персональной ALS-рекомендации по строкам панели."""

    lookup: pd.DataFrame  # ncodpers | snapshot_date | recommended_product_id | recommended_score
    models: Dict[pd.Timestamp, object]  # снимок → обученная на его истории ALS-модель
    fit_seconds: float

    def latest_lookup(self) -> pd.DataFrame:
        """Рекомендации по последнему снимку — артефакт для сервиса (lookup по ncodpers)."""
        if self.lookup.empty:
            return self.lookup
        last = self.lookup['snapshot_date'].max()
        return (self.lookup[self.lookup['snapshot_date'] == last]
                .set_index('ncodpers')[['recommended_product_id']])


def als_feature_lookup(df: pd.DataFrame, events: pd.DataFrame, user_map: Dict[int, int], *,
                       snapshots: Optional[Sequence[pd.Timestamp]] = None,
                       products: Sequence[str] = PRODUCTS, factors: int = 50,
                       iterations: int = 50, regularization: float = 0.05,
                       random_state: int = 0, refit_every: int = 1,
                       date_column: str = 'fecha_dato',
                       user_column: str = 'ncodpers') -> AlsFeature:
    """Персональная рекомендация ALS для каждой строки панели — без заглядывания вперёд.

    Для снимка `t` модель учится на событиях **строго до** `t`, а рекомендация берётся
    для состояния клиента на `t`. Это то, что доступно в момент прогноза, поэтому
    признак не подглядывает в будущее ни на train-, ни на test-строках.

    Переобучение модели на каждый снимок дорого, поэтому ``refit_every`` позволяет
    обновлять её реже (например, раз в квартал): между обновлениями используется
    последняя модель, обученная на истории не позже текущего снимка, — устаревшая
    история не нарушает запрет на подглядывание вперёд.
    """
    available = sorted(pd.to_datetime(df[date_column].unique()))
    if snapshots is not None:
        wanted = {pd.Timestamp(value) for value in snapshots}
        available = [date for date in available if date in wanted]

    rows: List[pd.DataFrame] = []
    models: Dict[pd.Timestamp, object] = {}
    model = None
    started = time.perf_counter()
    for position, snapshot in enumerate(available):
        history = events[events[date_column] < snapshot]
        if history.empty:
            continue
        if model is None or refit_every <= 1 or position % refit_every == 0:
            matrix = als_user_item(history, user_map, products_n=len(products),
                                   user_column=user_column)
            model = fit_als(matrix, factors=factors, iterations=iterations,
                            regularization=regularization, random_state=random_state)
        models[snapshot] = model

        snapshot_df = df[df[date_column] == snapshot]
        users = snapshot_df[user_column].to_numpy()
        codes = np.array([user_map[int(user)] for user in users], dtype=np.int64)
        item_codes, scores = top1_from_factors(model, codes, owned_matrix(snapshot_df, products))
        rows.append(pd.DataFrame({
            user_column: users.astype(np.int64),
            'snapshot_date': snapshot,
            ALS_FEATURE_NAME: [product_name(code) for code in item_codes],
            'recommended_score': scores,
        }))

    lookup = pd.concat(rows, ignore_index=True) if rows else pd.DataFrame(
        columns=[user_column, 'snapshot_date', ALS_FEATURE_NAME, 'recommended_score'])
    return AlsFeature(lookup=lookup, models=models, fit_seconds=time.perf_counter() - started)


def als_topk_lists(model, user_item: scipy.sparse.csr_matrix, user_codes: np.ndarray,
                   k: int, *, batch_size: int = 2048) -> List[List[str]]:
    """Top-k персональных рекомендаций ALS в именах продуктов (для метрик ранжирования)."""
    lists: List[List[str]] = []
    for start in range(0, len(user_codes), batch_size):
        batch = user_codes[start:start + batch_size]
        batch_ids, _ = model.recommend(batch, user_item[batch],
                                       filter_already_liked_items=True, N=k)
        for row in batch_ids:
            lists.append([name for name in (product_name(code) for code in row)
                          if name is not None])
    return lists


def relevant_products(events: pd.DataFrame,
                      products: Sequence[str] = PRODUCTS) -> pd.Series:
    """Купленные продукты по клиентам: ``ncodpers → множество имён продуктов``."""
    return events.groupby('ncodpers')['purchase'].apply(
        lambda codes: {products[int(code) - 1] for code in codes})
