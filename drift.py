"""Мониторинг дрейфа входных признаков микросервиса (PSI).

Сервис копит сырые числовые значения входных запросов в скользящем окне
и при скрейпе `/metrics` (и по запросу `GET /drift`) считает PSI окна против
эталонных гистограмм обучающего запуска. Эталон (`drift_reference`) пишется
в `fastapi/preprocessing_params.json` ноутбуком modeling.ipynb
(`preprocessing.compute_drift_reference`).

PSI (population stability index): sum((ref - obs) * ln(ref / obs)) по бинам
эталона. Практические пороги: < 0.1 — сдвига нет, 0.1–0.25 — умеренный,
> 0.25 — значительный (на таком пороге стоит алерт
`BankRecommenderFeatureDrift` в fastapi/prometheus/alerts.yml).

Импорт модуля лёгкий: только math, threading, collections, numpy.
"""

from __future__ import annotations

import logging
import math
import threading
from collections import deque
from typing import Any, Dict, Optional

import numpy as np

logger = logging.getLogger('bank_recommender.drift')

# Пороги PSI: см. docstring модуля.
PSI_MODERATE = 0.1
PSI_SIGNIFICANT = 0.25


def psi_score(reference_shares: np.ndarray, observed_shares: np.ndarray,
              eps: float = 1e-3) -> float:
    """PSI двух распределений по одинаковым бинам (доли floored в eps)."""
    reference = np.clip(np.asarray(reference_shares, dtype=float), eps, None)
    observed = np.clip(np.asarray(observed_shares, dtype=float), eps, None)
    reference = reference / reference.sum()
    observed = observed / observed.sum()
    return float(np.sum((reference - observed) * np.log(reference / observed)))


class DriftMonitor:
    """Скользящее окно наблюдений + PSI против эталона обучения.

    Потокобезопасен: `observe` вызывается из пула потоков uvicorn.
    Если эталона нет (`drift_reference` отсутствует или равен null), монитор
    пассивен: `observe` — no-op, отчёт — status 'no_reference'.
    """

    def __init__(self, reference: Optional[Dict[str, Any]] = None,
                 window_size: int = 2000, min_samples: int = 200):
        self.reference = reference or {}
        self.window_size = int(window_size)
        self.min_samples = int(min_samples)
        self._lock = threading.Lock()
        self._buckets = {
            feature: deque(maxlen=self.window_size) for feature in self.reference
        }
        self.observed_total = 0

    @property
    def enabled(self) -> bool:
        return bool(self.reference)

    def observe(self, **features: Any) -> None:
        """Добавляет наблюдение (вызывается на каждый запрос /predict).

        Значения могут быть None (поле не передали) или нечисловыми —
        такие пропускаются: дрейф считается только по реально присланным
        значениям, подставленные медианы обучения распределение портят.
        """
        if not self.enabled:
            return
        updates: Dict[str, float] = {}
        for feature in self._buckets:
            value = features.get(feature)
            if value is None:
                continue
            try:
                numeric = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(numeric):
                updates[feature] = numeric
        if not updates:
            return
        with self._lock:
            for feature, numeric in updates.items():
                self._buckets[feature].append(numeric)
            self.observed_total += 1

    def psi(self, feature: str) -> Optional[float]:
        """PSI окна по признаку; None, если наблюдений меньше min_samples."""
        bucket = self._buckets.get(feature)
        if bucket is None:
            return None
        with self._lock:
            values = np.asarray(bucket, dtype=float)
        if values.size < self.min_samples:
            return None
        edges = np.asarray(self.reference[feature]['edges'], dtype=float)
        counts, _ = np.histogram(values, bins=edges)
        return psi_score(np.asarray(self.reference[feature]['shares'], dtype=float),
                         counts / counts.sum())

    def report(self) -> Dict[str, Any]:
        """Отчёт по всем отслеживаемым признакам (для GET /drift)."""
        if not self.enabled:
            return {'status': 'no_reference',
                    'detail': 'в preprocessing_params.json нет непустой секции drift_reference — '
                              'переобучите модель актуальным modeling.ipynb'}
        features = {}
        for feature in self.reference:
            value = self.psi(feature)
            with self._lock:
                samples = len(self._buckets[feature])
            if value is None:
                status = 'collecting'
            elif value >= PSI_SIGNIFICANT:
                status = 'significant'
            elif value >= PSI_MODERATE:
                status = 'moderate'
            else:
                status = 'ok'
            features[feature] = {'psi': value, 'samples': samples, 'status': status}
        overall = 'collecting' if any(info['status'] == 'collecting'
                                      for info in features.values()) else 'ok'
        if any(info['status'] == 'significant' for info in features.values()):
            overall = 'significant'
        elif any(info['status'] == 'moderate' for info in features.values()):
            overall = 'moderate'
        return {
            'status': overall,
            'window_size': self.window_size,
            'min_samples': self.min_samples,
            'observed_total': self.observed_total,
            'thresholds': {'moderate': PSI_MODERATE, 'significant': PSI_SIGNIFICANT},
            'features': features,
        }
