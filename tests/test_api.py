"""Тесты HTTP-сервиса: happy path, валидация, деградация (CODE_REVIEW §P2.16)."""

import json

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app1
from preprocessing import PreprocessingParams
from tests.conftest import make_profile


def _post_raw_json(client, payload):
    """POST сырым телом через `json.dumps` — как шлёт профили test.ipynb.

    У `json.dumps` по умолчанию `allow_nan=True`, поэтому пандасовские пропуски
    уезжают JSON-токенами `NaN`/`Infinity`, а не `null`.
    """
    return client.post(
        '/predict',
        content=json.dumps(payload),
        headers={'Content-Type': 'application/json'},
    )


@pytest.fixture()
def patched_service(monkeypatch, tiny_model_bundle, tiny_personal_recs):
    """Сервис с игрушечной моделью вместо настоящей saved_model.pkl."""
    monkeypatch.setattr(app1, 'MODEL', tiny_model_bundle['model'])
    monkeypatch.setattr(app1, 'PERSONAL_RECS', tiny_personal_recs)
    monkeypatch.setattr(app1, 'PARAMS', PreprocessingParams())
    monkeypatch.setattr(app1, 'LABEL_MAP', dict(tiny_model_bundle['label_map']))
    return TestClient(app1.app)


def test_health_ok_when_model_loaded(patched_service):
    response = patched_service.get('/health')
    assert response.status_code == 200
    payload = response.json()
    assert payload['status'] == 'ok'
    assert payload['model_loaded'] is True
    assert payload['personal_recs_loaded'] is True


def test_health_degraded_without_model(monkeypatch):
    monkeypatch.setattr(app1, 'MODEL', None)
    response = TestClient(app1.app).get('/health')
    assert response.json()['status'] == 'degraded'
    assert response.json()['model_loaded'] is False


def test_predict_happy_path_schema(patched_service):
    response = patched_service.post('/predict', json=make_profile())
    assert response.status_code == 200
    payload = response.json()
    assert set(payload) == {'prediction', 'product', 'confidence', 'top_k'}
    assert isinstance(payload['prediction'], int)
    assert 0.0 <= payload['confidence'] <= 1.0
    # топ всегда согласован с предсказанием и отсортирован по вероятности
    assert payload['top_k'][0]['code'] == payload['prediction']
    assert payload['top_k'][0]['probability'] == pytest.approx(payload['confidence'])
    probabilities = [item['probability'] for item in payload['top_k']]
    assert probabilities == sorted(probabilities, reverse=True)


def test_predict_matches_offline(patched_service, tiny_model_bundle, tiny_personal_recs):
    """Сервис и офлайн-пайплайн дают одинаковый ответ (train/serve-согласованность)."""
    import preprocessing

    profile = make_profile(ncodpers=1001)
    response = patched_service.post('/predict', json=profile)
    assert response.status_code == 200
    payload = response.json()

    features = preprocessing.prepare_features(profile, PreprocessingParams(), tiny_personal_recs)
    proba = tiny_model_bundle['model'].predict_proba(features)[0]
    best = int(np.argmax(proba))
    assert payload['prediction'] == int(tiny_model_bundle['model'].classes_[best])
    assert payload['confidence'] == pytest.approx(float(proba[best]))


def test_predict_minimal_profile_uses_training_defaults(patched_service):
    """Только ncodpers: пропуски закрываются медианами/модами обучения."""
    response = patched_service.post('/predict', json={'ncodpers': 1001})
    assert response.status_code == 200
    assert isinstance(response.json()['prediction'], int)


def test_predict_ignores_extra_fields(patched_service):
    profile = make_profile()
    profile['future_unknown_field'] = 'ignored'
    assert patched_service.post('/predict', json=profile).status_code == 200


def test_predict_missing_ncodpers_is_422(patched_service):
    response = patched_service.post('/predict', json={'age': 30})
    assert response.status_code == 422


def test_predict_nan_in_string_fields_is_missing_not_422(patched_service):
    """NaN в строковых полях (пандасовские пропуски из test.ipynb) → 200, а не 422."""
    profile = make_profile()
    profile['sexo'] = float('nan')
    profile['tiprel_1mes'] = float('nan')
    profile['canal_entrada'] = float('nan')
    profile['nomprov'] = float('nan')
    response = _post_raw_json(patched_service, profile)
    assert response.status_code == 200
    assert isinstance(response.json()['prediction'], int)


def test_predict_non_finite_in_numeric_fields_is_missing(patched_service):
    """NaN/±Infinity в числовых полях закрываются медианами обучения → 200."""
    profile = make_profile(age=float('nan'), renta=float('inf'), antiguedad=float('-inf'))
    assert _post_raw_json(patched_service, profile).status_code == 200


def test_predict_nan_profile_matches_none_profile(patched_service):
    """NaN семантически равен отсутствию значения: ответы совпадают с None-профилем."""
    nan_profile = make_profile(sexo=float('nan'), age=float('nan'), renta=float('nan'))
    none_profile = make_profile(sexo=None, age=None, renta=None)
    nan_response = _post_raw_json(patched_service, nan_profile)
    none_response = patched_service.post('/predict', json=none_profile)
    assert nan_response.status_code == none_response.status_code == 200
    assert nan_response.json()['prediction'] == none_response.json()['prediction']
    assert nan_response.json()['confidence'] == pytest.approx(none_response.json()['confidence'])


def test_validation_error_containing_nan_is_422_not_500(patched_service):
    """Заведомо битое тело с NaN: 422 с валидным JSON, а не 500.

    Регрессия: стоковый обработчик 422 падал с
    `ValueError: Out of range float values are not JSON compliant`, т.к. клал
    сырой `input` (NaN) в тело через `json.dumps(allow_nan=False)`.
    """
    response = patched_service.post(
        '/predict',
        content='NaN',  # тело вообще не объект — input ошибки и есть NaN
        headers={'Content-Type': 'application/json'},
    )
    assert response.status_code == 422
    assert 'detail' in response.json()


def test_validation_error_nested_nan_is_422_not_500(patched_service):
    """NaN внутри значения неверного типа тоже не должен ронять сериализацию 422."""
    response = patched_service.post(
        '/predict',
        content='{"ncodpers": 1001, "age": {"nested": NaN}}',
        headers={'Content-Type': 'application/json'},
    )
    assert response.status_code == 422
    assert 'detail' in response.json()


def test_predict_without_model_is_503(monkeypatch, tiny_personal_recs):
    monkeypatch.setattr(app1, 'MODEL', None)
    response = TestClient(app1.app).post('/predict', json=make_profile())
    assert response.status_code == 503


def test_predict_without_personal_recs_still_works(patched_service, monkeypatch):
    """Нет ALS-файла: признак вырождается в '0', сервис отвечает 200."""
    monkeypatch.setattr(app1, 'PERSONAL_RECS', None)
    response = patched_service.post('/predict', json=make_profile(ncodpers=999999))
    assert response.status_code == 200


def test_metrics_endpoint_exposes_counters(patched_service):
    patched_service.post('/predict', json=make_profile())
    response = patched_service.get('/metrics')
    assert response.status_code == 200
    assert 'bank_recommender_requests_total' in response.text
    assert 'bank_recommender_predict_latency_seconds' in response.text
