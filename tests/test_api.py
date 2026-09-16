"""Тесты HTTP-сервиса: happy path, валидация, деградация (CODE_REVIEW §P2.16)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import app1
from preprocessing import PreprocessingParams
from tests.conftest import make_profile


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
