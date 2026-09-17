# Рекомендательная система банковских продуктов

Проект строит рекомендательную систему для банковских продуктов на датасете
**Santander Product Recommendation**: по профилю клиента и текущему набору продуктов
модель оценивает, какой продукт клиент может приобрести в следующем периоде.

В репозитории есть полный контур: загрузка данных → EDA → обучение модели → офлайн-
проверка инференса → FastAPI-сервис → нагрузочный тест → мониторинг Prometheus/Grafana.

## Структура репозитория

| Путь | Назначение |
|---|---|
| `loader.ipynb` | Загрузка `train_ver2.csv` из Kaggle в `data/` и проверка целостности файла. |
| `eda.ipynb` | Исследовательский анализ данных, выводы по пропускам, выбросам, категориям и продуктам. |
| `recommendations_analysis.md` | Краткий аналитический отчёт по EDA и итоговому качеству модели. |
| `modeling.ipynb` | Основной ноутбук обучения: таргет, ALS-признак, временное разбиение, train-only предобработка, Optuna, MLflow, экспорт артефактов. |
| `rec_sys.ipynb` | Офлайн-проверка инференса на одном реальном клиенте без запуска HTTP-сервиса. |
| `test.ipynb` | Интерактивная обёртка над нагрузочным тестом. |
| `preprocessing.py` | Единый модуль предобработки для обучения, офлайн-инференса и FastAPI. |
| `drift.py` | PSI-мониторинг дрейфа входных признаков `age`, `antiguedad`, `renta`. |
| `app1.py` | FastAPI-сервис: `POST /predict`, `GET /health`, `GET /metrics`, `GET /drift`. |
| `load_test.py` | Нагрузочный тест сервиса с проверкой схемы ответа и SLO; пишет отчёты в `artifacts/`. |
| `tests/` | Pytest-набор для предобработки, API, дрейфа и smoke-проверки артефактов. |
| `artifacts/` | Последние сохранённые отчёты: качество модели, важности признаков, ALS-метрики, нагрузочный отчёт. |
| `fastapi/` | Dockerfile, docker compose, Prometheus/Grafana-конфиги и артефакты сервиса (`preprocessing_params.json`, `model_version.json`, `replacer.json`, `personal_als.parquet`). |
| `.env.example` | Шаблон переменных окружения для Kaggle, MLflow, сервиса, мониторинга и нагрузочного теста. |

Канонический порядок запуска ноутбуков:

```text
loader.ipynb → eda.ipynb → modeling.ipynb → rec_sys.ipynb → test.ipynb
```

## Подготовка окружения

Требуется Python 3.10+.

```bash
python -m venv .venv
source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Для разработки и тестов дополнительно:

```bash
pip install -r requirements-dev.txt
pytest
ruff check .
```

## Данные

Датасет берётся из Kaggle Competition
[Santander Product Recommendation](https://www.kaggle.com/competitions/santander-product-recommendation/data).

1. На странице соревнования примите правила (*Rules → I Understand and Accept*).
2. В Kaggle откройте *Settings → API → Create New Token* и скачайте `kaggle.json`.
3. Либо положите файл в `~/.kaggle/kaggle.json`, либо создайте `.env`:

   ```bash
   cp .env.example .env
   # затем заполните KAGGLE_USERNAME и KAGGLE_KEY
   ```

4. Запустите `loader.ipynb`. Он скачает архив соревнования, найдёт `train_ver2.csv`,
   сохранит его в `data/` и проверит размер/колонки/число строк.

`data/` не коммитится: исходный CSV весит около 2.14 ГБ.

## Как устроена модель

### Целевая переменная

Модель решает многоклассовую задачу: предсказывает первый продукт, который клиент
приобретёт в 2016 году.

- `0` — покупки нет;
- коды `3`, `5`, `8`, `9`, `12`, `13`, `18`, `19`, `20`, `22`, `23`, `24` — продукты из шорт-листа;
- `99` (`other`) — первым куплен продукт вне шорт-листа.

Шорт-лист строится по покупкам 2015 года: в отдельные классы попадают продукты с
долей покупок не ниже 1%, затем берутся первые 12 продуктов. В последнем запуске
13 продуктов прошли порог 1%, но из-за ограничения `MAX_TARGET_PRODUCTS=12` продукт
`ind_fond_fin_ult1` попал в `other`.

### Предобработка и признаки

Предобработка выполняется единым кодом из `preprocessing.py`:

1. удаляются малоинформативные признаки `indrel`, `indext`, `nomprov`, `tipodom`;
2. числовые признаки (`age`, `antiguedad`, `renta`) приводятся к числам;
3. пропуски заполняются медианами/модами обучающей части;
4. возраст переводится в интервалы, даты — во временные когорты;
5. редкие категории сворачиваются в `other`;
6. `renta` и `antiguedad` клиппируются по квантилям;
7. добавляются арифметические признаки и групповые агрегаты;
8. добавляется `total_products` — число продуктов у клиента;
9. добавляется персональная ALS-рекомендация `recommended_product_id`.

Все статистики предобработки обучаются только на train-части и сохраняются в
`fastapi/preprocessing_params.json`. Сервис использует тот же файл, поэтому инференс
повторяет обучающий пайплайн.

### Разбиение и обучение

В актуальном ноутбуке `modeling.ipynb` используется временное разбиение по дате
привлечения клиента `fecha_alta`:

- train — клиенты, привлечённые до cutoff;
- test — более новые клиенты;
- cutoff последнего запуска: `2013-10-09`;
- размерности последнего запуска: train `(644744, 57)`, test `(275674, 57)`.

Гиперпараметры `RandomForestClassifier(class_weight='balanced')` подбираются Optuna
по кросс-валидации на train. Test используется один раз — для финальной оценки.

## Качество рекомендательной системы

Актуальные числа зафиксированы в `artifacts/classification_report.txt`,
`artifacts/als_metrics.csv` и выводах `modeling.ipynb`.

### Главное резюме

Модель **существенно лучше случайного ранжирования продуктов**. Для PR-AUC базовый
уровень случайного ранжирования равен доле класса в test (`share` в отчёте). По
продуктовым классам с ненулевым support средний PR-AUC составляет **0.296** против
средней базовой доли **0.0093**, то есть примерно **в 31.8 раза выше бейзлайна**.

При этом accuracy интерпретировать нужно осторожно: 90.46% test — это класс
`no_purchase`. Если всегда предсказывать «покупки нет», accuracy будет около 90.46%.
Текущая модель даёт **93.40% accuracy** (`+2.94 п.п.`), но для рекомендаций важнее
per-class PR-AUC, recall/precision по продуктам и lift относительно доли класса.

### Метрики финального классификатора

| Метрика | Значение последнего запуска |
|---|---:|
| CV ROC-AUC macro на train | 0.9622 |
| ROC-AUC macro OVR на test | 0.9594 |
| Accuracy на test | 0.9340 |
| Precision macro | 0.3235 |
| Recall macro | 0.3362 |
| F1 macro | 0.2816 |
| PR-AUC macro | 0.3437 |

### Lift по ключевым продуктам относительно случайного бейзлайна

| Продукт | Support test | Доля класса | PR-AUC | Lift к случайному PR-AUC |
|---|---:|---:|---:|---:|
| `ind_recibo_ult1` | 10 543 | 3.824% | 0.584 | 15.3× |
| `ind_cco_fin_ult1` | 3 450 | 1.252% | 0.502 | 40.1× |
| `ind_nomina_ult1` | 3 600 | 1.306% | 0.413 | 31.7× |
| `ind_ecue_fin_ult1` | 2 731 | 0.991% | 0.296 | 29.9× |
| `ind_cno_fin_ult1` | 2 566 | 0.931% | 0.234 | 25.2× |
| `ind_nom_pens_ult1` | 496 | 0.180% | 0.743 | 412.9× |

`ind_nom_pens_ult1` показывает максимальный lift, но support у него небольшой —
такие значения нужно подтверждать на последующих периодах. Наиболее устойчивые
практические выводы дают массовые продукты с тысячами объектов в test.

### ALS как самостоятельный baseline и как признак

ALS строит персональные top-5 рекомендации по истории 2015 года и используется в
финальном классификаторе как признак `recommended_product_id`.

| Scope | Users | precision@5 | recall@5 | hit_rate@5 |
|---|---:|---:|---:|---:|
| Клиенты, присутствующие и в 2015, и в 2016 | 94 091 | 3.18% | 9.60% | 13.36% |
| Все покупатели 2016 года, включая cold-start | 115 867 | 2.58% | 7.79% | 10.85% |

Вывод: ALS сам по себе — слабый, но полезный baseline. Финальная модель использует
ALS-рекомендацию вместе с анкетными, продуктовыми и агрегатными признаками; по PR-AUC
она даёт кратный lift относительно случайного бейзлайна по продуктам.

## Артефакты обучения

`modeling.ipynb` создаёт/обновляет:

- `fastapi/saved_model.pkl` — sklearn-пайплайн для сервиса и `rec_sys.ipynb`;
- `fastapi/preprocessing_params.json` — параметры предобработки;
- `fastapi/model_version.json` — run id, дата, split, гиперпараметры и метрики;
- `fastapi/replacer.json` — словарь редких категорий;
- `fastapi/personal_als.parquet` — персональные ALS-рекомендации по `ncodpers`;
- `artifacts/classification_report.txt` — полный отчёт качества;
- `artifacts/feature_importances.csv` — важности отобранных признаков;
- `artifacts/als_metrics.csv` — top-k метрики ALS.

`*.pkl` игнорируется Git, поэтому после свежего клона `fastapi/saved_model.pkl` может
отсутствовать. В таком случае сервис поднимется, но `POST /predict` вернёт `503` до
запуска `modeling.ipynb` или ручной передачи файла модели.

## MLflow

MLflow нужен для истории экспериментов, но сервис читает локальные артефакты из
`fastapi/`.

```bash
mlflow server \
  --host 127.0.0.1 \
  --port 5000 \
  --backend-store-uri sqlite:///mlflow.db \
  --default-artifact-root ./mlruns_artifacts
```

Переменные `MLFLOW_TRACKING_HOST` и `MLFLOW_TRACKING_PORT` можно задать в `.env`.

## FastAPI-сервис

### Локальный запуск

```bash
uvicorn app1:app --host 0.0.0.0 --port 8079
```

Проверка состояния:

```bash
curl http://localhost:8079/health
```

Пример запроса:

```bash
curl -X POST "http://localhost:8079/predict" \
  -H "Content-Type: application/json" \
  -d '{
        "ncodpers": 1049144,
        "fecha_dato": "2015-01-28",
        "ind_empleado": "N",
        "pais_residencia": "ES",
        "sexo": "V",
        "age": 24,
        "fecha_alta": "2012-08-10",
        "ind_nuevo": 0,
        "antiguedad": 35,
        "indrel_1mes": "1.0",
        "tiprel_1mes": "I",
        "indresi": "S",
        "conyuemp": null,
        "canal_entrada": "KHE",
        "indfall": "N",
        "cod_prov": 28,
        "ind_actividad_cliente": 0,
        "renta": 101850,
        "segmento": "03 - UNIVERSITARIO",
        "ind_ahor_fin_ult1": 0,
        "ind_aval_fin_ult1": 0,
        "ind_cco_fin_ult1": 1,
        "ind_cder_fin_ult1": 0,
        "ind_cno_fin_ult1": 0,
        "ind_ctju_fin_ult1": 0,
        "ind_ctma_fin_ult1": 0,
        "ind_ctop_fin_ult1": 0,
        "ind_ctpp_fin_ult1": 0,
        "ind_deco_fin_ult1": 0,
        "ind_deme_fin_ult1": 0,
        "ind_dela_fin_ult1": 0,
        "ind_ecue_fin_ult1": 0,
        "ind_fond_fin_ult1": 0,
        "ind_hip_fin_ult1": 0,
        "ind_plan_fin_ult1": 0,
        "ind_pres_fin_ult1": 0,
        "ind_reca_fin_ult1": 0,
        "ind_tjcr_fin_ult1": 0,
        "ind_valo_fin_ult1": 0,
        "ind_viv_fin_ult1": 0,
        "ind_nomina_ult1": 0,
        "ind_nom_pens_ult1": 0,
        "ind_recibo_ult1": 0
      }'
```

Ответ содержит top-k продуктов:

```json
{
  "prediction": 0,
  "product": "no_purchase",
  "confidence": 0.891,
  "top_k": [
    {"code": 0, "product": "no_purchase", "probability": 0.891},
    {"code": 24, "product": "ind_recibo_ult1", "probability": 0.025}
  ]
}
```

## Нагрузочное тестирование

Сервис должен быть поднят на `LOAD_TEST_URL` (по умолчанию `http://localhost:8079/predict`).

```bash
python load_test.py
```

Параметры задаются через `.env`/переменные окружения:

```bash
LOAD_TEST_WORKERS=50 \
LOAD_TEST_MAX_TIME=50 \
LOAD_TEST_P95_SLO_MS=5000 \
LOAD_TEST_MIN_SUCCESS_RATE=99.0 \
python load_test.py
```

Последний сохранённый отчёт (`artifacts/load_test_report.html`):

| total_requests | success_rate | avg_latency | p95_latency | p99_latency | RPS |
|---:|---:|---:|---:|---:|---:|
| 1 408 | 99.01% | 1 822 мс | 2 483 мс | 2 671 мс | 27.56 |

SLO последнего прогона выполнены: `success_rate ≥ 99%`, `p95 < 5000 мс`.
В отчёте зафиксировано 14 ответов `HTTP 422` на 1 408 запросов; это валидируемые
ошибки входного профиля, а не 5xx-падения сервиса.

## Prometheus и Grafana

Инфраструктура лежит в `fastapi/docker-compose.yaml` и поднимает три контейнера:

1. `ml_service` — FastAPI-сервис на `8079`;
2. `prometheus` — сбор метрик на `9090`;
3. `grafana` — дашборд на `3000`.

### Быстрый запуск всего стека

```bash
cp .env.example .env        # опционально: можно оставить значения по умолчанию
# отредактируйте .env, если нужны другие порты или пароль Grafana

docker compose --env-file .env -f fastapi/docker-compose.yaml up --build
```

Если `.env` не нужен, можно запустить с дефолтами:

```bash
docker compose -f fastapi/docker-compose.yaml up --build
```

Адреса по умолчанию:

| Компонент | URL | Примечание |
|---|---|---|
| FastAPI | `http://localhost:8079` | `/health`, `/predict`, `/metrics`, `/drift` |
| Prometheus | `http://localhost:9090` | UI для запросов PromQL и статуса алертов |
| Grafana | `http://localhost:3000` | логин/пароль из `.env`, по умолчанию `admin` / `admin` |

Переменные портов:

```env
VM_PORT=8079
THE_PORT=8079
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
GRAFANA_USER=admin
GRAFANA_PASS=admin
```

Для нелокального окружения обязательно смените `GRAFANA_PASS`. Внутренний `THE_PORT`
лучше оставлять `8079`: Prometheus по умолчанию скрейпит `ml-service:8079`. Если меняете
`THE_PORT`, одновременно поменяйте target в `fastapi/prometheus/prometheus.yml`.

### Что проверить после запуска

```bash
curl http://localhost:8079/health
curl http://localhost:8079/metrics | grep bank_recommender
```

Затем создайте трафик, иначе графики будут пустыми:

```bash
LOAD_TEST_MAX_TIME=60 LOAD_TEST_WORKERS=20 python load_test.py
```

### Как пользоваться Prometheus

1. Откройте `http://localhost:9090`.
2. Перейдите в **Status → Targets**. У job `bank-recommender` должен быть статус `UP`.
   Если статус `DOWN`, проверьте, что контейнер `ml_service` запущен и что target в
   `fastapi/prometheus/prometheus.yml` равен `ml-service:8079`.
3. На вкладке **Graph** выполните полезные PromQL-запросы:

   ```promql
   # RPS по всем эндпоинтам
   sum(rate(bank_recommender_requests_total[1m]))

   # p95 латентности /predict
   histogram_quantile(0.95,
     sum(rate(bank_recommender_predict_latency_seconds_bucket[5m])) by (le)
   )

   # доля 5xx-ошибок
   sum(rate(bank_recommender_requests_total{status=~"5.."}[5m]))
     /
   sum(rate(bank_recommender_requests_total[5m]))

   # распределение предсказаний по классам
   sum by (prediction) (rate(bank_recommender_predictions_total[5m]))

   # PSI-дрейф по признакам, если в preprocessing_params.json есть drift_reference
   bank_recommender_drift_psi

   # CPU и память процесса сервиса
   rate(process_cpu_seconds_total[5m])
   process_resident_memory_bytes
   ```

4. Алерты смотрите в **Status → Rules** и **Alerts**. Подключены правила:
   - `BankRecommenderDown` — сервис не скрейпится 2 минуты;
   - `BankRecommenderHighLatencyP95` — p95 `/predict` выше 2 секунд 10 минут;
   - `BankRecommenderHighErrorRate` — доля 5xx выше 1%;
   - `BankRecommenderFeatureDrift` — PSI выше 0.25.

Alertmanager в compose не добавлен: сработки видны в UI Prometheus. Для уведомлений
в почту/Slack/Telegram добавьте контейнер Alertmanager и блок `alerting` в
`fastapi/prometheus/prometheus.yml`.

### Как пользоваться Grafana

1. Откройте `http://localhost:3000`.
2. Войдите под `GRAFANA_USER` / `GRAFANA_PASS` (`admin` / `admin` по умолчанию).
3. Откройте **Dashboards → Bank Recommender**. Дашборд автоматически загружается из
   `fastapi/grafana.json`, а datasource Prometheus — из
   `fastapi/grafana/provisioning/datasources/prometheus.yml`.
4. На дашборде доступны панели:
   - p95 латентности `/predict`;
   - success rate / error rate;
   - RPS;
   - распределение предсказаний по классам;
   - CPU и память процесса;
   - PSI-дрейф входных признаков.
5. Если панель пустая, проверьте:
   - был ли трафик после запуска (`load_test.py`);
   - в Prometheus target `bank-recommender` находится в состоянии `UP`;
   - datasource в Grafana называется `Prometheus` и имеет UID `prometheus`.
6. Если дашборд не появился автоматически: **Dashboards → New → Import → Upload JSON**,
   выберите `fastapi/grafana.json`, datasource — `Prometheus`.

### Дрейф входных признаков

`GET /drift` возвращает JSON-отчёт по PSI. Метрика в Prometheus называется
`bank_recommender_drift_psi{feature="..."}`.

Важно: PSI появляется только если в `fastapi/preprocessing_params.json` есть непустая
секция `drift_reference` и накоплено не меньше `DRIFT_MIN_SAMPLES` наблюдений. Если секции
нет или она равна `null`, `/drift` вернёт `status: "no_reference"`; пересоздайте артефакты
запуском `modeling.ipynb`.

## Ограничения текущей версии

- Модель выдаёт один основной класс и top-k вероятностей, а не оптимизирует отдельный
  ранжировщик всех 24 продуктов.
- Редкие продукты объединены в `other`, поэтому отдельные рекомендации для них не
  интерпретируются как самостоятельные продуктовые классы.
- `fastapi/saved_model.pkl` не хранится в Git и должен быть создан локальным обучающим
  запуском или доставлен из внешнего model registry.
- Без непустого `drift_reference` в `preprocessing_params.json` дрейф-мониторинг работает в
  пассивном режиме.
