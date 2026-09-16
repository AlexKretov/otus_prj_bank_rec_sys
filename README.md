# Рекомендательная система банковских продуктов

## Задача
Создание рекомендательной системы для банковских продуктов.

## Описание
Имеется набор данных по клиентам банка, где на ряд дат 2015 года приведены основные характеристики клиентов
и перечень банковских продуктов, которыми они обладают. Необходимо создать рекомендательную систему,
которая рекомендует клиентам банка продукты, которые могут быть им интересны.
В качестве верификации используются данные из 2016 года, приведённые в том же файле.

На основе обученной модели сделан микросервис (FastAPI), который принимает профиль клиента
по HTTP-запросу и возвращает предсказание.

## Структура репозитория

| Файл | Назначение |
|---|---|
| `loader.ipynb` | Загрузка датасета `train_ver2.csv` из соревнования Kaggle в каталог `data/` |
| `eda.ipynb` | Исследовательский анализ данных и выводы |
| `modeling.ipynb` | Основной ноутбук: предобработка (статистики только на train), временное разбиение, ALS-признак, обучение и тюнинг модели, артефакты, выводы |
| `rec_sys.ipynb` | Офлайн-проверка инференса: предобработка одного клиента и предсказание сохранённой моделью |
| `load_test.py` | Нагрузочный тест сервиса (запускается из CI); формирует `artifacts/load_test_report.html` / `artifacts/load_test_report.png` |
| `test.ipynb` | Тонкая интерактивная обёртка над `load_test.py` |
| `preprocessing.py` | Общий модуль предобработки: используют и `modeling.ipynb` (fit/apply), и сервис |
| `drift.py` | PSI-мониторинг дрейфа входных признаков (скользящее окно против эталона обучения) |
| `app1.py` | Микросервис предсказаний (FastAPI): `POST /predict`, `GET /health`, `GET /metrics`, `GET /drift` |
| `tests/` | Pytest-набор: предобработка (в т.ч. fit/apply, временное разбиение), дрейф-мониторинг, API, smoke-тесты артефактов |
| `requirements-dev.txt` | Зависимости разработчика (`pytest`, `httpx`, `ruff`) |
| `pyproject.toml` | Конфиги `pytest` и `ruff` |
| `.env.example` | Шаблон `.env`: ключи Kaggle, настройки MLflow, сервиса и мониторинга |
| `recommendations_analysis.md` | Итоговый аналитический отчёт по данным (см. также выводы в `eda.ipynb`) |
| `CODE_REVIEW.md` | Результаты code review и план улучшений проекта |
| `columns.txt` | Человекочитаемые названия колонок датасета |
| `artifacts/` | Отчёты последних запусков: `classification_report.txt`, `feature_importances.csv`, `als_metrics.csv`, `load_test_report.html` / `.png` |
| `replacer.json` | Старая копия словаря редких категорий; актуальный файл генерируется в `fastapi/replacer.json` |
| `fastapi/` | Артефакты для сервиса (`saved_model.pkl`, `preprocessing_params.json`, `model_version.json`, `replacer.json`, `personal_als.parquet`) и инфраструктура (`Dockerfile`, `docker-compose.yaml`, `prometheus/`, `grafana.json`) |
| `image.png` | Скриншот примера дашборда Grafana |

Ноутбуки запускаются по порядку: `loader.ipynb` → `eda.ipynb` → `modeling.ipynb` →
`rec_sys.ipynb` (офлайн-проверка) → `test.ipynb` (нагрузочный прогон поднятого сервиса).
Канонические ноутбуки — только эти пять; курсовых «сшивок» (`full_final_prj`,
`mle_final_prj2` из CODE_REVIEW §4.5) в репозитории нет.

## Построение модели

Сначала проводится исследовательский анализ данных (`eda.ipynb`, в нём же основные выводы),
затем в `modeling.ipynb` строится и оценивается модель.

### Подготовка среды
Требуется Python ≥ 3.10.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Данные
Датасет берётся из соревнования Kaggle
**[Santander Product Recommendation](https://www.kaggle.com/competitions/santander-product-recommendation/data)**.

1. Получите API-токен: Kaggle → *Settings* → *API* → *Create New Token* (скачается `kaggle.json`).
2. Примите правила соревнования на вкладке *Data → Rules* — иначе Kaggle не отдаёт файлы.
3. Создайте `.env` по образцу `.env.example` и укажите в нём `KAGGLE_USERNAME` и `KAGGLE_KEY`
   (альтернатива — положить `kaggle.json` в `~/.kaggle/kaggle.json`).
4. Запустите `loader.ipynb` — он скачает `train_ver2.csv` (≈2.3 ГБ, 13 647 309 строк × 48 колонок)
   и сохранит его в `data/`.

Датасет в репозиторий не входит (каталог `data/` указан в `.gitignore`).

### Целевая переменная
Модель — многоклассовый классификатор: он предсказывает, какой продукт клиент купит в 2016 году.
Класс `0` — «покупки не было», остальные классы — коды продуктов из шорт-листа,
`other` (`99`) — если первым куплен продукт, не попавший в шорт-лист.

Шорт-лист — продукты, на которые в 2015 году пришлось не меньше 1% покупок
(у большинства из 24 продуктов доля покупателей меньше 0.5%, поэтому классы по ним
состояли бы из одного-двух объектов). Правило отбора, состав шорт-листа и расшифровка
классов печатаются в `modeling.ipynb` и сохраняются в `fastapi/preprocessing_params.json`
(ключ `label_map`).

### Предобработка данных
Предобработка включает в себя:
1. Удаление малоинформативных признаков (`indrel`, `indext`, `nomprov`, `tipodom`).
2. Замена возраста (`age`) на возрастные корзины.
3. Замена временных переменных (`fecha_alta`, `ult_fec_cli_1t`) на интервалы.
4. Устранение экстремумов в числовых переменных (`renta`, `antiguedad`): всё, что менее 1% квантиля
   и более 96% квантиля, заменяется на соответствующее предельное значение.
5. Сворачивание маргинальных групп (менее 1%) в одну категорию `other` по всем категориальным показателям.
6. Все статистики (медианы, моды, границы клиппинга, возрастные бины, когорты дат, наборы редких
   категорий и групповые агрегаты) **обучаются только на train-части**
   (`fit_preprocessing_params` в `preprocessing.py`) — утечки до сплита нет (см. «Валидация»).
7. Сохранение посчитанных параметров (плюс агрегаты, расшифровка классов и эталонные
   гистограммы `drift_reference` для мониторинга дрейфа) в
   `fastapi/preprocessing_params.json` — этот файл читает сервис, поэтому предобработка на проде
   совпадает с обучением (та же функция `apply_preprocessing` применяется к train/test
   и к профилю запроса).

### Синтез новых признаков
1. Арифметические признаки (`antiguedad ± × / renta`, логарифмы, корни) — pandas/numpy.
   Изначально считались featuretools-DFS: на инференсе это стоило ~250 мс на запрос и
   падало гонкой потоков (`KeyError: 'DataFrame main does not exist'` в каждом четвёртом
   конкурентном запросе → HTTP 500 → SLO success_rate ~70%). Текущая реализация
   побитово эквивалентна DFS (проверяется тестом на паритет с featuretools).
2. Общее количество продуктов на последнюю дату 2015 года для каждого клиента.
3. Персональные рекомендации на основе ALS-модели (признак `recommended_product_id`).

### Моделирование
Модель построена с помощью sklearn в формате pipeline: кодировка категориальных признаков,
нормализация числовых, отбор признаков. Основную предсказывающую функцию выполняет
`RandomForestClassifier` с `class_weight='balanced'`.

**Валидация.** Разбиение train/test — **временное** (`temporal_split` в `preprocessing.py`)
по дате привлечения клиента (`fecha_alta`): в train — клиенты, привлечённые до cutoff-даты
(≈70%), в test — более новые. По дате последнего среза разбивать нельзя: она тянет за собой
отток (клиенты с ранней последней датой просто ушли из банка, и «покупок 2016» у них нет
априори). Профили одного клиента не смешиваются между частями, поэтому метрики
соответствуют честному временному сценарию (модель не подглядывает в будущее).
Тюнинг гиперпараметров (Optuna) скорится по **кросс-валидации на train**; тестовая выборка
используется один раз — для финальной оценки. Обучение, метрики, параметры разбиения
и артефакты логируются в MLflow.

### Запуск MLflow (опционально)
```bash
mlflow server \
    --host 127.0.0.1 \
    --port 5000 \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlruns_artifacts
```
Далее ноутбуки запускаются по порядку: `eda.ipynb` → `modeling.ipynb` → `rec_sys.ipynb`.

### Проверка модели
Отчёт последнего запуска — `artifacts/classification_report.txt` (per-class таблица, macro-
и взвешенные метрики, PR-AUC по каждому классу), важности признаков —
`artifacts/feature_importances.csv`, качество ALS — `artifacts/als_metrics.csv`
(precision@k, recall@k, hit_rate в двух скоупах: клиенты обоих годов и все покупатели
2016 года, включая cold-start).

> **Важно при интерпретации:** классы сильно несбалансированы (доля покупок ≈ 2%),
> поэтому accuracy и взвешенные метрики описывают в основном класс «покупки не было».
> Ориентируйтесь на per-class метрики, macro-метрики и PR-AUC (average precision) по
> классам-покупкам: при таком дисбалансе PR-AUC честнее ROC-AUC, а базовый уровень
> случайного ранжирования равен доле класса (колонка `share` в отчёте). Методологические
> оговорки собраны в `CODE_REVIEW.md` и в разделе «Ограничения» ниже.

## Развёртывание микросервиса

Для запуска сервиса нужны артефакты (все создаются в `modeling.ipynb` в каталоге `fastapi/`):
- `fastapi/saved_model.pkl` — обученный sklearn-пайплайн (`joblib.dump`), канонический экспорт;
- `fastapi/preprocessing_params.json` — параметры предобработки, посчитанные при обучении
  (медианы, моды, границы клиппинга, возрастные бины, когорты дат, агрегаты, расшифровка классов);
- `fastapi/model_version.json` — версия модели: run id, дата, гиперпараметры, метрики, классы;
- `fastapi/replacer.json` — словарь сворачивания редких категорий в `other`;
- `fastapi/personal_als.parquet` — персональные ALS-рекомендации, индексированные по `ncodpers`.

Каталог артефактов переопределяется переменной окружения `ARTIFACTS_DIR` (см. `.env.example`).
Если `preprocessing_params.json` отсутствует, сервис работает на legacy-константах
из `preprocessing.py` и предупреждает об этом в логе — такие константы могли разойтись
с обучающим запуском.

Запуск напрямую:

```bash
uvicorn app1:app --host 0.0.0.0 --port 8079
# с C-реализациями event loop / HTTP-парсера (как в docker-образе):
uvicorn app1:app --host 0.0.0.0 --port 8079 --loop uvloop --http httptools --no-access-log
```

Запуск через docker compose (из корня репозитория, вместе с Prometheus и Grafana):

```bash
docker compose -f fastapi/docker-compose.yaml up --build
```

Производительность: предобработка одного запроса — ~18 мс CPU (pandas/numpy,
без featuretools). При 50 конкурентных клиентах на 2 vCPU сервис держит
~26 RPS с p95 ≈ 2.2 с; SLO нагрузочного теста (`success_rate ≥ 99%`,
`p95 < 5000 мс`) выполняется с запасом. Раньше featuretools-DFS на каждый
запрос стоил ~250 мс и падал гонкой потоков (`KeyError: 'DataFrame main does
not exist'` в каждом четвёртом конкурентном запросе → HTTP 500 →
success_rate ~70%). Пул потоков эндпоинтов ограничен
(`THREAD_POOL_TOKENS`, по умолчанию 10): меньше потоков — меньше
переключений GIL при CPU-bound обработке.

Сервис принимает `POST /predict` с JSON-профилем клиента (поля как в `columns.txt`,
обязательно только `ncodpers`, неизвестные поля игнорируются) и отвечает
`{"prediction": <код класса>, "product": <название продукта>, "confidence": <вероятность>,
"top_k": [...]}`. `GET /health` возвращает статус сервиса и список загруженных артефактов.
Если модель не найдена, `/predict` отвечает `503`; ошибки валидации — `422` с именем поля.

### Мониторинг

`GET /metrics` отдаёт метрики в формате Prometheus: счётчик запросов
`bank_recommender_requests_total`, гистограмму латентности
`bank_recommender_predict_latency_seconds`, счётчик предсказаний по классам
`bank_recommender_predictions_total` и gauge дрейфа `bank_recommender_drift_psi`
(плюс стандартные `process_*`).
Конфиг скрейпинга — `fastapi/prometheus/prometheus.yml`. Дашборд Grafana
(`fastapi/grafana.json`: p95 латентности, CPU/память процесса, распределение
предсказаний) загружается автоматически: compose маунтит его и каталог
`fastapi/grafana/provisioning/` (датасорс Prometheus + провайдер дашбордов),
поэтому после `docker compose up` дашборд «Bank Recommender» уже на месте —
импортировать руками ничего не нужно. Порты и учётные данные задаются
переменными окружения (см. `.env.example`: `VM_PORT`, `THE_PORT`,
`PROMETHEUS_PORT`, `GRAFANA_PORT`, `GRAFANA_USER`, `GRAFANA_PASS`).
Пример дашборда см. в `image.png`.

**Алерты.** Правила лежат в `fastapi/prometheus/alerts.yml` и подключены через
`rule_files`: простой сервиса (`up == 0`), p95 латентности > 2 с, доля 5xx > 1%
и дрейф признаков (PSI > 0.25). Alertmanager не настроен — сработки видны в UI
Prometheus (*Status → Alerts*) и через `/api/v1/alerts`; для уведомлений в почту
или мессенджер добавьте alertmanager в `docker-compose.yaml` и блок `alerting`
в `prometheus.yml`.

**Дрейф-мониторинг.** Сервис копит сырые значения `age`/`antiguedad`/`renta`
из запросов в скользящем окне (`drift.py`) и считает PSI против эталонных
гистограмм обучающего запуска (секция `drift_reference` в
`preprocessing_params.json`). Ко всему добавлены: JSON-отчёт `GET /drift`,
метрика `bank_recommender_drift_psi` для Prometheus и алерт
`BankRecommenderFeatureDrift`. Пропуски в запросе не наблюдаются — иначе
медианы обучения «вымывали» бы сдвиг. Пока в файле нет `drift_reference`
(артефакт прошлого запуска), монитор пассивен (`/drift` → `no_reference`).
Размер окна и порог — `DRIFT_WINDOW_SIZE` / `DRIFT_MIN_SAMPLES`.

### Версия модели

Канонический экспорт — `fastapi/saved_model.pkl` (его читают и сервис, и
`rec_sys.ipynb`); MLflow-модель (`runs:/<run_id>/model`) — только для истории
экспериментов. Версия зафиксирована в `fastapi/model_version.json` (запуск до
введения временного разбиения и train-only статистик из P3; переобучение
обновит метрики и секцию `split`):

| Параметр | Значение |
|---|---|
| MLflow run id | `e200be8d83f34a8dac0ad6a3eb785c42` (эксперимент `RecSys_Modeling`) |
| Дата обучения | 2026-09-16 |
| train / test | 715 129 / 306 484 объектов, 57 признаков (стратифицированное разбиение) |
| Гиперпараметры | `n_estimators=32`, `max_depth=None`, `min_samples_split=20` |
| CV ROC-AUC (train) | 0.9014 |
| ROC-AUC ovr macro (test) | 0.9316 |
| F1 macro / precision macro / recall macro | 0.4684 / 0.4009 / 0.6050 |
| PR-AUC macro | 0.4210 |

Полный per-class отчёт — `artifacts/classification_report.txt`, метрики ALS —
`artifacts/als_metrics.csv` (precision@5 ≈ 0.043, recall@5 ≈ 0.138,
hit_rate ≈ 0.180 — старая индексация ALS, см. CODE_REVIEW §P3.5).

### Тестовый запрос к микросервису

```bash
curl -X POST "http://localhost:8079/predict" \
  -H "Content-Type: application/json" \
  -d '{"fecha_dato": "2015-05-28", "ncodpers": 444579, "ind_empleado": null, "pais_residencia": "NI", "sexo": "H", "age": 116, "fecha_alta": "1998-07-01", "ind_nuevo": 1.0, "antiguedad": 213, "indrel": 1.0, "ult_fec_cli_1t": "2015-11-24", "indrel_1mes": "4.0", "tiprel_1mes": "R", "indresi": "S", "indext": "N", "conyuemp": null, "canal_entrada": "KCE", "indfall": "S", "tipodom": 1.0, "cod_prov": 23.0, "nomprov": "SALAMANCA", "ind_actividad_cliente": 0.0, "renta": 63830.06999999999, "segmento": "03 - UNIVERSITARIO", "ind_ahor_fin_ult1": 1, "ind_aval_fin_ult1": 0, "ind_cco_fin_ult1": 1, "ind_cder_fin_ult1": 1, "ind_cno_fin_ult1": 1, "ind_ctju_fin_ult1": 0, "ind_ctma_fin_ult1": 1, "ind_ctop_fin_ult1": 1, "ind_ctpp_fin_ult1": 1, "ind_deco_fin_ult1": 0, "ind_deme_fin_ult1": 1, "ind_dela_fin_ult1": 0, "ind_ecue_fin_ult1": 0, "ind_fond_fin_ult1": 0, "ind_hip_fin_ult1": 1, "ind_plan_fin_ult1": 1, "ind_pres_fin_ult1": 1, "ind_reca_fin_ult1": 1, "ind_tjcr_fin_ult1": 0, "ind_valo_fin_ult1": 0, "ind_viv_fin_ult1": 1, "ind_nomina_ult1": 0.0, "ind_nom_pens_ult1": 1.0, "ind_recibo_ult1": 1}'
```

## Нагрузочное тестирование

Запуск (сервис должен быть поднят на порту 8079):

```bash
python load_test.py        # полный прогон; код возврата 0 — SLO выполнены, 1 — нарушены
```

Тест шлёт запросы параллельно пулом воркеров, использует реальные профили клиентов
из датасета, проверяет схему каждого ответа и SLO — по итогам формируются
`artifacts/load_test_report.html` и `artifacts/load_test_report.png`.
Код возврата по SLO позволяет гонять тест в CI без ноутбука; для интерактивной
работы остаётся `test.ipynb` (тонкая обёртка над тем же модулем).
Параметры задаются переменными окружения (см. `.env.example`): `LOAD_TEST_URL`,
`LOAD_TEST_WORKERS`, `LOAD_TEST_MAX_TIME`, `LOAD_TEST_SAMPLES`,
`LOAD_TEST_P95_SLO_MS`, `LOAD_TEST_MIN_SUCCESS_RATE`, `LOAD_TEST_DATA_PATH`,
`LOAD_TEST_FIRST_ROWS`, `LOAD_TEST_REPORT_DIR`.

## Разработка

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest          # unit-тесты предобработки, API и smoke-тесты артефактов
ruff check .    # линтер (pyflakes + pycodestyle)
```

Чтобы не коммитить вывод ячеек ноутбуков (`nbstripout`, CODE_REVIEW §7.1):

```bash
pip install nbstripout && nbstripout --install
```

(`.gitattributes` уже содержит фильтр; без установленного `nbstripout` git его игнорирует.)

## Ограничения и известные проблемы

Проведено code review (`CODE_REVIEW.md`); планы P0–P3 выполнены. В P3 устранены
оставшиеся огрехи из этого списка:

- **Предобработка (исправлено в P3).** Раньше медианы, моды, квантили клиппинга, наборы
  редких категорий, когорты дат и агрегаты считались на всём датасете до разбиения
  (умеренная утечка). Теперь все статистики обучаются только на train
  (`fit_preprocessing_params` в `preprocessing.py`), а к test и к продовым профилям
  применяется одна и та же функция `apply_preprocessing`/`prepare_features`.
- **Валидация (исправлено в P3).** Разбиение стратифицированное заменено временным
  (`temporal_split` по дате привлечения `fecha_alta`: train — старые клиенты,
  test — новые; разбиение по дате среза оказалось вырожденным из-за оттока —
  см. CODE_REVIEW §P3.3). Профили одного клиента не смешиваются — метрики больше
  не оптимистичны за счёт «подглядывания». Качество ALS теперь считается в двух скоупах: на клиентах обоих
  годов (как раньше) и на всех покупателях 2016 года, включая cold-start, —
  честная оценка (`artifacts/als_metrics.csv`, колонка `scope`). Попутно исправлена
  индексация ALS-матрицы (строки — user_map-индексы, как и вызовы `recommend`;
  раньше использовались сырые `ncodpers`, и рекомендации приписывались чужим
  клиентам — см. CODE_REVIEW §3.7).
- **Редкие категории (исправлено в P3).** На проде сворачивание редких категорий
  сравнивается в строковом домене — как при обучении; раньше float-значения
  (`cod_prov`, `ind_nuevo` и др.) не матчились со строковыми ключами словаря
  и не сворачивались (OHE молча занулял такие категории).
- **Инфраструктура (исправлено в P3).** Добавлены алерты Prometheus
  (`fastapi/prometheus/alerts.yml`: простой, p95 > 2 с, 5xx > 1%, дрейф) и
  дрейф-мониторинг признаков (PSI по `age`/`antiguedad`/`renta`: `GET /drift`,
  метрика `bank_recommender_drift_psi`, алерт `BankRecommenderFeatureDrift`).
  Alertmanager не настроен: сработки смотрятся в UI Prometheus; для уведомлений
  — подключите alertmanager. Нагрузочный прогон вынесен в `load_test.py`
  с кодом возврата по SLO — можно гонять в CI. Отчёты перенесены в `artifacts/`.
- **Зависимости (исправлено в P3).** `requirements.txt` не резолвился
  (`mlflow==2.7.1` требовал `numpy<2` и `pyarrow<14` при пинах
  `numpy==2.2.5`/`pyarrow==19.0.1`) — mlflow обновлён до 2.22.1.

Остающиеся оговорки:

- **Таргет.** Целевая переменная многоклассовая: код первого купленного в 2016 году продукта
  из шорт-листа, `0` — покупки не было, `other` — куплен редкий продукт. Шорт-лист ограничен
  продуктами с долей покупок ≥ 1%: редкие продукты предсказываются только как `other`,
  а не как отдельный класс. Модель предсказывает один продукт, а не ранжирование всего списка.
- **Качество ALS** (используется и как признак `recommended_product_id`, и как бейзлайн)
  остаётся базовым — фактические значения precision@k / recall@k см. в
  `artifacts/als_metrics.csv`.
- **Закоммиченные артефакты** (`fastapi/preprocessing_params.json`,
  `fastapi/model_version.json`, `fastapi/personal_als.parquet`, отчёты в `artifacts/`)
  получены предыдущим запуском — ещё до P3 (стратифицированное разбиение, статистики
  на всём датасете, новая секция `drift_reference` отсутствует, поэтому
  `/drift` → `no_reference` до переобучения). Прогон актуального `modeling.ipynb`
  на реальных данных обновит их (как в P1 — изменения вступают в силу после
  переобучения; модель и сервис при этом совместимы в обе стороны).
- **Артефакты обучения** (`fastapi/saved_model.pkl` и др.) в репозитории лежат частично:
  модель нужно получить, прогнав `loader.ipynb` → `modeling.ipynb`
  (см. «Данные» и «Развёртывание»). По-хорошему конвейер артефактов должен идти из
  model registry/пайплайна, а не из git.
