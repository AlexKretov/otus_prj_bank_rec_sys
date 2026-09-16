"""Тесты дрейф-мониторинга (drift.py) и эндпоинта GET /drift."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app1
from drift import PSI_MODERATE, PSI_SIGNIFICANT, DriftMonitor, psi_score
from tests.conftest import make_profile


def make_reference(values, n_bins=5, edges=None):
    """Эталонная гистограмма по списку значений (формат drift_reference)."""
    values = np.asarray(values, dtype=float)
    edges = np.linspace(values.min(), values.max(), n_bins + 1) if edges is None else edges
    edges = edges.astype(float)
    edges[0], edges[-1] = -np.inf, np.inf
    counts, _ = np.histogram(values, bins=edges)
    return {'edges': edges.tolist(), 'shares': (counts / counts.sum()).tolist()}


def test_psi_score_identical_is_zero():
    assert psi_score([0.5, 0.5], [0.5, 0.5]) == pytest.approx(0.0)


def test_psi_score_grows_with_shift():
    small = psi_score([0.9, 0.1], [0.8, 0.2])
    large = psi_score([0.9, 0.1], [0.1, 0.9])
    assert 0.0 < small < large
    assert large > PSI_SIGNIFICANT  # полный разворот распределения — значительный сдвиг


def test_monitor_disabled_without_reference():
    monitor = DriftMonitor(None)
    assert not monitor.enabled
    monitor.observe(age=30.0)  # no-op, не падает
    assert monitor.report()['status'] == 'no_reference'


def test_monitor_needs_min_samples_before_psi():
    monitor = DriftMonitor({'age': make_reference(np.arange(100, dtype=float))},
                           window_size=1000, min_samples=10)
    monitor.observe(age=5.0)
    assert monitor.psi('age') is None
    info = monitor.report()
    assert info['status'] == 'collecting'
    assert info['features']['age']['samples'] == 1


def test_monitor_psi_zero_on_same_distribution():
    rng = np.random.default_rng(0)
    reference_values = rng.normal(45.0, 10.0, 5000)
    monitor = DriftMonitor({'age': make_reference(reference_values)},
                           window_size=3000, min_samples=100)
    for value in rng.normal(45.0, 10.0, 2000):  # то же распределение
        monitor.observe(age=value)
    assert monitor.psi('age') == pytest.approx(0.0, abs=0.05)
    assert monitor.report()['features']['age']['status'] == 'ok'


def test_monitor_detects_shifted_distribution():
    rng = np.random.default_rng(1)
    reference_values = rng.normal(45.0, 10.0, 5000)
    monitor = DriftMonitor({'age': make_reference(reference_values)},
                           window_size=2000, min_samples=100)
    for value in rng.normal(75.0, 10.0, 2000):  # среднее переехало
        monitor.observe(age=value)
    psi = monitor.psi('age')
    assert psi is not None and psi > PSI_SIGNIFICANT
    info = monitor.report()
    assert info['status'] == 'significant'
    assert info['features']['age']['status'] == 'significant'


def test_monitor_status_matches_thresholds():
    rng = np.random.default_rng(2)
    reference_values = rng.normal(45.0, 10.0, 5000)
    monitor = DriftMonitor({'age': make_reference(reference_values)},
                           window_size=2000, min_samples=100)
    for value in rng.normal(48.0, 10.0, 2000):  # небольшой сдвиг среднего
        monitor.observe(age=value)
    psi = monitor.psi('age')
    status = monitor.report()['features']['age']['status']
    expected = ('significant' if psi >= PSI_SIGNIFICANT
                else 'moderate' if psi >= PSI_MODERATE else 'ok')
    assert status == expected


def test_monitor_skips_none_and_non_numeric():
    monitor = DriftMonitor({'renta': make_reference(np.arange(100, dtype=float))},
                           min_samples=3)
    monitor.observe(renta=None)
    monitor.observe(renta='not-a-number')
    monitor.observe(renta=float('nan'))
    monitor.observe(renta=float('inf'))
    assert monitor.observed_total == 0
    assert monitor.psi('renta') is None
    monitor.observe(renta=50.0)
    assert monitor.observed_total == 1


def test_monitor_thread_safety_and_window_cap():
    from concurrent.futures import ThreadPoolExecutor

    monitor = DriftMonitor({'age': make_reference(np.arange(100, dtype=float))},
                           window_size=500, min_samples=100)

    with ThreadPoolExecutor(max_workers=16) as executor:
        list(executor.map(lambda i: monitor.observe(age=float(i % 150)), range(2000)))
    assert monitor.observed_total == 2000
    assert monitor.report()['features']['age']['samples'] == 500  # окно обрезано


# --- эндпоинт GET /drift ------------------------------------------------------


def test_drift_endpoint_no_reference(monkeypatch):
    monkeypatch.setattr(app1, 'DRIFT_MONITOR', DriftMonitor(None))
    response = TestClient(app1.app).get('/drift')
    assert response.status_code == 200
    assert response.json()['status'] == 'no_reference'


def test_drift_endpoint_collecting_then_ok(monkeypatch):
    monitor = DriftMonitor({'age': make_reference(np.arange(100, dtype=float))},
                           window_size=500, min_samples=3)
    monkeypatch.setattr(app1, 'DRIFT_MONITOR', monitor)
    client = TestClient(app1.app)
    info = client.get('/drift').json()
    assert info['status'] == 'collecting'
    assert info['features']['age']['samples'] == 0
    for _ in range(3):
        monitor.observe(age=42.0)
    info = client.get('/drift').json()
    assert info['features']['age']['psi'] is not None
    assert set(info) >= {'status', 'window_size', 'features', 'thresholds'}


def test_predict_feeds_drift_monitor(monkeypatch, tiny_model_bundle, tiny_personal_recs):
    """POST /predict скармливает входные значения монитору (None пропускаются)."""
    from preprocessing import PreprocessingParams

    monkeypatch.setattr(app1, 'MODEL', tiny_model_bundle['model'])
    monkeypatch.setattr(app1, 'PERSONAL_RECS', tiny_personal_recs)
    monkeypatch.setattr(app1, 'PARAMS', PreprocessingParams())
    monkeypatch.setattr(app1, 'LABEL_MAP', dict(tiny_model_bundle['label_map']))
    monitor = DriftMonitor(
        {feature: make_reference(np.linspace(1, 300, 1000))
         for feature in ('age', 'antiguedad', 'renta')},
        window_size=500, min_samples=1)
    monkeypatch.setattr(app1, 'DRIFT_MONITOR', monitor)

    client = TestClient(app1.app)
    response = client.post('/predict', json=make_profile(age=45.0))
    assert response.status_code == 200
    assert monitor.observed_total == 1
    # запрос без числовых полей: наблюдать нечего — счётчик не растёт,
    # медианы обучения в окно не попадают (иначе замоют сдвиг)
    response = client.post('/predict', json={'ncodpers': 1001})
    assert response.status_code == 200
    assert monitor.observed_total == 1
