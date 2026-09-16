"""Smoke-тесты артефактов репозитория (CODE_REVIEW §P2.16).

Проверяют файлы, которые лежат в git или появляются после `modeling.ipynb`.
Каждый тест скипается, если его файла нет, — набор остаётся зелёным
и на чистой машине без данных.
"""

import json
from pathlib import Path

import pandas as pd
import pytest

ARTIFACTS_DIR = Path('fastapi')

REQUIRED_PARAM_SECTIONS = {'medians', 'modes', 'clip_bounds', 'age_intervals',
                           'date_cohorts', 'aggregates', 'replacer', 'label_map',
                           'feature_columns'}


def test_preprocessing_params_shape():
    path = ARTIFACTS_DIR / 'preprocessing_params.json'
    if not path.exists():
        pytest.skip('нет preprocessing_params.json')
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert REQUIRED_PARAM_SECTIONS <= set(payload), 'в файле нет обязательных секций'
    assert '0' in payload['label_map'], 'в label_map нет класса «покупки не было»'
    assert len(payload['age_intervals']) >= 2
    assert payload['feature_columns']['numeric'], 'пустой список числовых признаков'
    assert payload['feature_columns']['categorical'], 'пустой список категориальных признаков'


def test_personal_recs_indexed_by_ncodpers():
    """Регрессия CODE_REVIEW §1.5: lookup идёт по ID клиента, дубли вычищены."""
    path = ARTIFACTS_DIR / 'personal_als.parquet'
    if not path.exists():
        pytest.skip('нет personal_als.parquet')
    recs = pd.read_parquet(path)
    assert recs.index.name == 'ncodpers', 'parquet сохранён без индекса по клиенту'
    assert int(recs.index.duplicated().sum()) == 0, 'в файле дубли ncodpers'
    assert 'recommended_product_id' in recs.columns
    assert int(recs['recommended_product_id'].notna().sum()) > 0, 'все рекомендации пустые'


def test_model_version_file_shape():
    """Версия модели зафиксирована (CODE_REVIEW §P2.20)."""
    path = ARTIFACTS_DIR / 'model_version.json'
    if not path.exists():
        pytest.skip('нет model_version.json')
    payload = json.loads(path.read_text(encoding='utf-8'))
    assert payload.get('mlflow_run_id'), 'в файле нет run id'
    assert payload.get('created_at'), 'в файле нет даты обучения'
    assert payload.get('metrics'), 'в файле нет метрик'
    assert '0' in payload.get('label_map', {}), 'в файле нет расшифровки классов'


def test_replacer_json_loads():
    """Словарь редких категорий читается и не пуст."""
    for path in (ARTIFACTS_DIR / 'replacer.json', Path('replacer.json')):
        if path.exists():
            payload = json.loads(path.read_text(encoding='utf-8'))
            assert isinstance(payload, dict) and payload
            return
    pytest.skip('нет replacer.json')
