"""Нагрузочный тест микросервиса рекомендаций (можно запускать из CI).

Раньше прогон существовал только как ноутбук `test.ipynb` и выполнялся
вручную (README, «Ограничения»). Теперь вся логика живёт здесь,
а ноутбук — тонкая интерактивная обёртка над этим модулем.

Запуск (сервис должен быть поднят):
    python load_test.py                     # прогон по переменным окружения
    LOAD_TEST_MAX_TIME=120 python load_test.py

Параметры (переменные окружения, см. .env.example):
    LOAD_TEST_URL, LOAD_TEST_WORKERS, LOAD_TEST_MAX_TIME, LOAD_TEST_SAMPLES,
    LOAD_TEST_P95_SLO_MS, LOAD_TEST_MIN_SUCCESS_RATE, LOAD_TEST_FIRST_ROWS,
    LOAD_TEST_DATA_PATH, LOAD_TEST_REPORT_DIR.

Результаты:
    artifacts/load_test_report.html, artifacts/load_test_report.png.
    Код возврата 0 — SLO выполнены, 1 — нарушены (AssertionError) или
    прогон не состоялся.
"""

from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import matplotlib

matplotlib.use('Agg')  # без GUI-бэкенда — прогон в CI/docker

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:  # seaborn для графиков опционален (в CI достаточно HTML)
    import seaborn as sns
except ImportError:  # pragma: no cover
    sns = None

import requests

# --- Параметры прогона (переопределяются переменными окружения) ---
URL = os.getenv('LOAD_TEST_URL', 'http://localhost:8079/predict')
WORKERS = int(os.getenv('LOAD_TEST_WORKERS', '50'))
MAX_TIME = float(os.getenv('LOAD_TEST_MAX_TIME', '50'))
N_PROFILES = int(os.getenv('LOAD_TEST_SAMPLES', '100'))
FIRST_ROWS = int(os.getenv('LOAD_TEST_FIRST_ROWS', '100000'))
DATA_PATH = os.getenv('LOAD_TEST_DATA_PATH', 'data/train_ver2.csv')
REPORT_DIR = Path(os.getenv('LOAD_TEST_REPORT_DIR', 'artifacts'))
# SLO: прогон падает, если p95 выше или доля успешных ответов ниже порога
P95_SLO_MS = float(os.getenv('LOAD_TEST_P95_SLO_MS', '5000'))
MIN_SUCCESS_RATE = float(os.getenv('LOAD_TEST_MIN_SUCCESS_RATE', '99.0'))


class NumpyEncoder(json.JSONEncoder):
    """json-серилизатор для numpy-типов в профилях из датасета."""

    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super().default(obj)


def load_label_classes(params_path='fastapi/preprocessing_params.json'):
    """Допустимые классы ответа из артефакта обучения (для проверки схемы)."""
    try:
        with open(params_path, encoding='utf-8') as file:
            return {int(code) for code in json.load(file)['label_map']}
    except (OSError, KeyError):
        print('preprocessing_params.json не найден — проверяю только базовые классы')
        return {0, 99}  # минимум: классы «нет покупки» и «редкий продукт»


def load_profiles(path=DATA_PATH, n=N_PROFILES, seed=42, first_rows=FIRST_ROWS):
    """Реальные профили клиентов — целые строки датасета.

    Раньше профиль собирался из несовместимых значений разных строк
    (возраст 116 + сегмент «UNIVERSITARIO») — такие запросы нерепрезентативны
    (CODE_REVIEW §5.3). Читаем только первые строки: для выборки достаточно.
    """
    sample = pd.read_csv(path, nrows=first_rows).sample(
        min(n, first_rows), random_state=seed)
    return sample.to_dict('records')


def check_response_schema(payload, label_map):
    """Проверяет схему ответа POST /predict; возвращает текст ошибки или None."""
    if not isinstance(payload, dict):
        return f'ответ не dict: {type(payload)}'
    for key in ('prediction', 'product', 'confidence', 'top_k'):
        if key not in payload:
            return f'нет ключа {key!r} в ответе: {payload}'
    if not isinstance(payload['prediction'], int):
        return f"prediction не int: {payload['prediction']!r}"
    if payload['prediction'] not in label_map:
        return f"неизвестный класс {payload['prediction']}"
    if not isinstance(payload['top_k'], list) or not payload['top_k']:
        return 'top_k пуст или не список'
    if payload['top_k'][0]['code'] != payload['prediction']:
        return 'top_k[0] не совпадает с prediction'
    probabilities = [item['probability'] for item in payload['top_k']]
    if probabilities != sorted(probabilities, reverse=True):
        return 'top_k не отсортирован по убыванию вероятности'
    return None


def send_request(url, profile, label_map, timeout=30):
    """Один запрос к сервису: возвращает (успех, латентность, ошибку)."""
    started = time.time()
    try:
        response = requests.post(
            url,
            headers={'Content-Type': 'application/json'},
            data=json.dumps(profile, cls=NumpyEncoder),
            timeout=timeout,
        )
        latency = (time.time() - started) * 1000  # мс
        if response.status_code != 200:
            return {'status': response.status_code, 'latency': latency,
                    'success': False, 'error': f'HTTP {response.status_code}'}
        schema_error = check_response_schema(response.json(), label_map)
        return {'status': 200, 'latency': latency,
                'success': schema_error is None, 'error': schema_error}
    except requests.RequestException as exc:
        return {'status': 0, 'latency': (time.time() - started) * 1000,
                'success': False, 'error': f'{type(exc).__name__}: {exc}'}


def run_load_test(profiles, url, max_time, workers, label_map, verbose=True):
    """Шлёт запросы параллельно через пул воркеров, пока не выйдет время.

    Раньше Executor создавался, но запросы шли последовательно в цикле `while`
    (~1.8 RPS), и p95/p99 ничего не значили (CODE_REVIEW §5.1). Теперь фьючерсы
    реально выполняются конкурентно, а очередь ограничена числом воркеров.
    """
    results, profile_index, submitted = [], 0, set()
    started = time.time()
    with ThreadPoolExecutor(max_workers=workers) as executor:
        while (time.time() - started) < max_time or submitted:
            while len(submitted) < workers and (time.time() - started) < max_time:
                profile = profiles[profile_index % len(profiles)]
                future = executor.submit(send_request, url, profile, label_map)
                submitted.add(future)
                profile_index += 1
            done, submitted = set(), submitted
            for future in as_completed(submitted, timeout=max_time):
                done.add(future)
                try:
                    results.append(future.result())
                except Exception as exc:  # падение воркера — тоже результат
                    results.append({'status': 0, 'latency': 0.0, 'success': False,
                                    'error': f'{type(exc).__name__}: {exc}'})
                submitted = submitted - done
                break  # добрать очередь до workers на следующей итерации
            if verbose and profile_index % 50 == 0:
                print(f'  отправлено {profile_index}, готово {len(results)}')
    frame = pd.DataFrame(results)
    frame.attrs['wall_seconds'] = time.time() - started
    return frame


def generate_report(results_df, *, url=URL, workers=WORKERS,
                    p95_slo_ms=P95_SLO_MS, min_success_rate=MIN_SUCCESS_RATE,
                    report_dir=REPORT_DIR, eps=1e-9):
    """Считает метрики, рисует графики, пишет HTML-отчёт.

    Возвращает dict с метриками и флагом slo_ok. Нарушение SLO — не
    AssertionError (assert отключается под -O), а явный ValueError ниже.
    """
    ok = float(results_df['success'].mean() * 100)
    latencies = results_df.loc[results_df['success'], 'latency']
    if latencies.empty:
        latencies = pd.Series([0.0])
    wall_seconds = results_df.attrs.get('wall_seconds', 0)
    p95 = float(np.percentile(latencies, 95))
    p99 = float(np.percentile(latencies, 99))

    report = {
        'total_requests': len(results_df),
        'success_rate': ok,
        'avg_latency': float(latencies.mean()),
        'p95_latency': p95,
        'p99_latency': p99,
        'rps': len(results_df) / wall_seconds if wall_seconds else None,
        'error_distribution': results_df['error'].value_counts().to_dict(),
    }

    report_dir.mkdir(parents=True, exist_ok=True)
    png_path = report_dir / 'load_test_report.png'
    html_path = report_dir / 'load_test_report.html'

    plt.figure(figsize=(15, 5))
    plt.subplot(131)
    if sns is not None:
        sns.histplot(results_df['latency'], bins=50)
    else:  # pragma: no cover
        plt.hist(results_df['latency'], bins=50)
    plt.title('Latency Distribution')
    plt.subplot(132)
    results_df['status'].value_counts().plot(kind='bar')
    plt.title('Status Code Distribution')
    plt.subplot(133)
    pd.Series([ok, 100 - ok], index=['Success', 'Error']).plot(
        kind='pie', autopct='%1.1f%%')
    plt.title('Success Rate')
    plt.tight_layout()
    plt.savefig(png_path)
    plt.close()

    html_report = f"""
    <html>
        <body>
            <h1>Load Test Report</h1>
            <p>URL: {url}, workers: {workers}, SLO: p95 &lt; {p95_slo_ms} мс,
               success_rate &gt;= {min_success_rate}%</p>
            <pre>{pd.DataFrame([report]).to_html()}</pre>
            <h2>Charts</h2>
            <img src="{png_path.name}" width="100%">
        </body>
    </html>
    """
    html_path.write_text(html_report, encoding='utf-8')
    print(f'Отчёты сохранены: {html_path}, {png_path}')

    # --- SLO-проверки: с допуском eps, без assert -------------------------
    failures = []
    if ok + eps < min_success_rate:
        failures.append(
            f'success_rate={ok:.4f}% < {min_success_rate}% '
            f'(не хватило {min_success_rate - ok:.4f} п.п.)'
        )
    if p95 + eps >= p95_slo_ms:
        failures.append(
            f'p95={p95:.0f} мс >= {p95_slo_ms} мс'
        )
    if failures:
        raise ValueError('SLO нарушен: ' + '; '.join(failures))

    return report


def main() -> int:
    """Полный прогон: профили → нагрузка → отчёт. Код возврата — статус SLO."""
    print(f'URL: {URL}, workers: {WORKERS}, max_time: {MAX_TIME}с, профилей: {N_PROFILES}')
    print(f'SLO: p95 < {P95_SLO_MS} мс, success_rate >= {MIN_SUCCESS_RATE}%')
    label_map = load_label_classes()
    profiles = load_profiles()
    print(f'Загружено профилей: {len(profiles)}')

    print('Starting load test...')
    results = run_load_test(profiles, URL, max_time=MAX_TIME, workers=WORKERS,
                            label_map=label_map)
    print('Test completed, generating report...')
    try:
        report = generate_report(results)
    except ValueError as exc:
        print(f'LOAD TEST FAILED: {exc}', file=sys.stderr)
        return 1
    print(pd.DataFrame([report]).to_string(index=False))
    return 0


if __name__ == '__main__':
    try:
        sys.exit(main())
    except (AssertionError, ValueError) as exc:
        print(f'LOAD TEST FAILED: {exc}', file=sys.stderr)
        sys.exit(1)
