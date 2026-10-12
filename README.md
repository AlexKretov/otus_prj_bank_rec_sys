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
| `eda.ipynb` | Исследовательский анализ данных: пропуски, выбросы, категории, продукты, дубликаты. Только выводы по самим данным — метрик модели в нём нет. |
| `recommendations_analysis.md` | Итоговый аналитический отчёт (данные, таргет, протоколы оценки, сравнение моделей, top-K метрики, абляция ALS). Целиком генерируется из `modeling.ipynb` и логируется в MLflow. |
| `modeling.ipynb` | Основной ноутбук обучения: таргет по всем покупкам, ALS-признак, два протокола разбиения, train-only предобработка, подбор RF и CatBoost, top-K метрики, абляция ALS, MLflow, экспорт артефактов. |
| `rec_sys.ipynb` | Офлайн-проверка инференса на одном реальном клиенте без запуска HTTP-сервиса. |
| `test.ipynb` | Интерактивная обёртка над нагрузочным тестом. |
| `preprocessing.py` | Единый модуль предобработки для обучения, офлайн-инференса и FastAPI. |
| `dataset.py` | Данные и таргет: события покупок, шорт-лист продуктов, панель «состояние снимка → покупка в следующем снимке», ALS-матрица и ALS-признак по снимкам. |
| `experiments.py` | Плечи сравнения моделей (Random Forest / CatBoost), общий для обоих протокол подбора Optuna, оценка на тесте, метрики ранжирования и абляция ALS-признака. |
| `recsys_metrics.py` | Метрики рекомендаций: precision@K, recall@K, NDCG@K, hit rate, бейзлайны случайного списка и популярности, метрики ALS-списков. |
| `drift.py` | PSI-мониторинг дрейфа входных признаков `age`, `antiguedad`, `renta`. |
| `app1.py` | FastAPI-сервис: `POST /predict`, `GET /health`, `GET /metrics`, `GET /drift`. |
| `load_test.py` | Нагрузочный тест сервиса с проверкой схемы ответа и SLO; пишет отчёты в `artifacts/`. |
| `tests/` | Pytest-набор: предобработка, таргет и панель, эксперименты и абляция, метрики рекомендаций, сквозной прогон пайплайна, API, дрейф, smoke-проверка артефактов. |
| `artifacts/` | Последние сохранённые отчёты обучения: качество модели, важности признаков, сравнение моделей, top-K метрики, абляция ALS, сравнение протоколов, диагностика таргета, ALS-метрики, нагрузочный отчёт. |
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

## Ключевые выводы EDA

Полный разбор — в `eda.ipynb` (выводы там касаются только самих данных).
Исходный файл: **13 647 309 строк, 48 колонок**.

### Качество данных

- Пропуски распределены неравномерно: `conyuemp` практически пустой,
  `ult_fec_cli_1t` заполнен лишь у небольшой доли клиентов, `renta` — около 20%
  пропусков и тяжёлый правый хвост.
- Есть аномалии в числовых признаках: возраст до 164 лет, экстремальные доходы,
  некорректные значения стажа (`antiguedad = -999999`).
- Явных дубликатов нет; неявных дубликатов в названиях провинций и каналов тоже нет,
  поэтому из пары `cod_prov`/`nomprov` достаточно оставить код.

### Профиль клиентской базы

- Около **99.6%** клиентов — резиденты Испании; крупнейшие провинции Madrid и Barcelona.
- Около **57.8%** клиентов неактивны.
- Привлечение сконцентрировано в нескольких каналах: `KHE`, `KAT`, `KFC`.
- Возрастной профиль смещён к молодым клиентам.

### Продукты

- Потребление крайне неравномерно: текущий счёт доминирует, а у большинства продуктов
  доля покупок меньше процента.
- В истории клиента покупка нового продукта — редкое событие: большинство клиентов
  долго живёт с одним базовым продуктом.
- В динамике по срезам видно снижение доли текущих и особых счетов, нишевые продукты
  остаются почти невостребованными.

### Что это даёт моделированию

Данные пригодны для рекомендательной системы, но требуют устойчивой предобработки:
заполнения пропусков, клиппинга `renta`/`antiguedad`, перевода возраста в интервалы,
сворачивания редких категорий в `other` и удаления малоинформативных колонок.
Сильный дисбаланс продуктов напрямую определяет состав таргета (шорт-лист + `other`)
и выбор метрик (per-class PR-AUC и lift вместо accuracy).

## Как устроена модель

### Целевая переменная

Строка датасета — состояние клиента в снимке `t` (месячный срез), таргет — **первый
продукт**, который клиент приобрёл в снимке `t + 1` (событие «продукт появился у
клиента впервые», `0 -> 1`). В панель попадают **все клиенты снимка**, а не только те,
у кого есть история для ALS: покупатель без истории получает заглушку `0` в признаке
`recommended_product_id`, но не может получить метку «покупки не было».

Классы:

- `0` — в целевом снимке покупки не было;
- коды `3`, `5`, `8`, `9`, `12`, `13`, `18`, `19`, `20`, `22`, `23`, `24` — продукты шорт-листа;
- `99` (`other`) — куплен продукт вне шорт-листа.

Шорт-лист строится по покупкам периода до cutoff (доля покупок не ниже 1%, затем
первые `MAX_TARGET_PRODUCTS=12` продуктов). Редкие классы сворачиваются в `other` по
статистике train, а не по всей выборке. Покупки клиентов, которых не было в предыдущем
снимке (новые и вернувшиеся клиенты), панель предсказать не может: их долю фиксирует
`artifacts/target_diagnostics.csv`, и её нужно читать вместе с метриками.

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
9. добавляется персональная ALS-рекомендация `recommended_product_id` (модель ALS
   переобучается по снимкам строго на прошлых событиях, `dataset.als_feature_lookup`).

Все статистики предобработки и агрегаты обучаются только на train-части и сохраняются в
`fastapi/preprocessing_params.json`. Сервис использует тот же файл, поэтому инференс
повторяет обучающий пайплайн.

### Протоколы оценки

- **Календарный (основной).** Строки панели делятся по дате снимка: train — снимки до
  `TEST_CUTOFF`, test — более поздние снимки. Это постановка «обучение на ранних
  периодах, проверка на последующих», то есть прогноз в будущее. Последний прогон:
  снимки `2015-02-28 … 2016-04-28`, cutoff `2016-01-01`,
  train `(8 399 876, 57)`, test `(3 690 523, 57)`.
- **Когортный (дополнительный).** Деление по дате привлечения клиента `fecha_alta`
  (`preprocessing.temporal_split`): train — клиенты, привлечённые раньше, test — новые.
  Отвечает на вопрос о переносимости модели между клиентскими когортами. Метрики двух
  протоколов **не сравнимы между собой** (разный состав теста), поэтому публикуются
  рядом в `artifacts/split_comparison.csv`, а не вместо друг друга.
- Статистики предобработки, агрегаты и порог сворачивания редких классов считаются
  только на train-части, поэтому тест не влияет ни на признаки, ни на состав классов.

### Сравнение моделей и обучение

В `modeling.ipynb` подбираются два плеча с одинаковым бюджетом
(`OPTUNA_TRIALS`, `OPTUNA_TIMEOUT`, `CV_SPLITS`; кросс-валидация на train) и
оцениваются одной функцией на одном тесте:

| Плечо | Кодирование признаков | Особенности |
|---|---|---|
| Random Forest | one-hot (`OneHotEncoder`, `handle_unknown='ignore'`) + масштабирование числовых | встроенный отбор признаков (`SelectFromModel`), `class_weight='balanced'` |
| CatBoost | нативное кодирование категориальных признаков (`cat_features`, без one-hot) | устойчив к большому числу категорий, свой `class_weights` |

Финальная модель выбирается по CV ROC-AUC macro; тест используется один раз — для
финальной оценки. Результаты складываются в `artifacts/model_comparison.csv`
и логируются в MLflow (обе модели — в один run с префиксами `random_forest_*`
и `catboost_*`).

## Качество рекомендательной системы

Числа ниже — последний полный прогон на датасете Kaggle (run
`c0de098c36604315ab771d51c7ecb033`, календарный протокол, cutoff `2016-01-01`).
Те же значения лежат в `artifacts/classification_report.txt`,
`artifacts/als_metrics.csv` и `fastapi/model_version.json`; полный пересчёт — запуском
`modeling.ipynb`.

### Главное резюме

Модель **существенно лучше случайного ранжирования продуктов**. Для PR-AUC базовый
уровень случайного ранжирования равен доле класса в test (`share` в отчёте): по
продуктовым классам средний PR-AUC равен **0.0318** против средней доли класса
**0.0026**, то есть примерно **в 12.2 раза** выше бейзлайна.

Accuracy интерпретировать нужно осторожно: 96.79% test — это класс `no_purchase`, поэтому
модель с метрикой `accuracy = 0.4142` намеренно жертвует точностью в пользу редких
продуктовых классов (`class_weight='balanced'`). Для рекомендаций важнее per-class
PR-AUC, recall/precision по продуктам и метрики персональных top-K списков.

### Метрики финального классификатора

| Метрика | Значение последнего запуска |
|---|---:|
| CV ROC-AUC macro на train | 0.7391 |
| ROC-AUC macro OVR на test | 0.7729 |
| Accuracy на test | 0.4142 |
| Precision macro | 0.0835 |
| Recall macro | 0.2663 |
| F1 macro | 0.0597 |
| PR-AUC macro | 0.0983 |

### Lift по ключевым продуктам относительно случайного бейзлайна

| Продукт | Support test | Доля класса | PR-AUC | Lift к случайному PR-AUC |
|---|---:|---:|---:|---:|
| `ind_recibo_ult1` | 37 870 | 1.026% | 0.0209 | 2.0× |
| `ind_nomina_ult1` | 18 655 | 0.506% | 0.1100 | 21.7× |
| `ind_cco_fin_ult1` | 16 420 | 0.445% | 0.0206 | 4.6× |
| `ind_tjcr_fin_ult1` | 16 038 | 0.435% | 0.0275 | 6.3× |
| `ind_cno_fin_ult1` | 9 341 | 0.253% | 0.0734 | 29.0× |
| `ind_ecue_fin_ult1` | 8 658 | 0.235% | 0.0104 | 4.4× |
| `ind_nom_pens_ult1` | 3 795 | 0.103% | 0.1144 | 111.3× |
| `other` (вне шорт-листа) | 2 739 | 0.074% | 0.0042 | 5.7× |

Максимальный lift у `ind_nom_pens_ult1`, но support небольшой — такие значения нужно
подтверждать на последующих периодах. Наиболее устойчивые практические выводы дают
массовые продукты с десятками тысяч объектов в test.

### Метрики персональных списков: precision@K, recall@K, NDCG@K

Помимо macro-метрик ноутбук считает метрики ранжирования списка рекомендаций
(модуль `recsys_metrics.py`): для каждой строки теста строится top-K продуктов
шорт-листа по вероятности модели, релевантный объект — фактически купленный продукт.

- `precision@K` — доля угаданных продуктов в списке из K позиций;
- `recall@K` — найден ли купленный продукт (релевантный объект один, поэтому здесь
  recall@K совпадает с hit rate@K);
- `NDCG@K` — попадание с поправкой на позицию продукта в списке.

Скоупы: все строки теста, все покупатели (покупка вне шорт-листа — промах) и
покупатели шорт-листа. Рядом считаются бейзлайны на тех же кандидатах: случайный
список и «показать всем топ-K популярных продуктов обучающего периода»; разрыв
с популярностным бейзлайном — это и есть эффект персонализации. Таблица складывается
в `artifacts/topk_metrics.csv` и логируется в MLflow (`topk_precision_at_*`,
`topk_recall_at_*`, `topk_ndcg_at_*`).

### Вклад ALS-признака (абляция)

Чтобы обосновать гибридную архитектуру, а не декларировать её, ноутбук обучает две
версии финальной модели на одном и том же train и одних гиперпараметрах:

- `with_als` — полный набор признаков;
- `without_als` — тот же набор без `recommended_product_id`.

Обе версии оцениваются одной функцией на одном тесте, включая метрики top-K. Результат
складывается в `artifacts/als_ablation.csv` и логируется в MLflow с префиксом
`ablation_*`: видно, сколько именно даёт коллаборативная фильтрация поверх анкетных,
продуктовых и агрегатных признаков.

### ALS как самостоятельный baseline и как признак

ALS строит персональные top-5 рекомендации по истории до `ALS_HISTORY_CUTOFF` и
используется в финальном классификаторе как признак `recommended_product_id`.

| Scope | Users | precision@5 | recall@5 | hit_rate@5 |
|---|---:|---:|---:|---:|
| Клиенты, присутствующие и в 2015, и в 2016 | 94 091 | 3.18% | 9.60% | 13.36% |
| Все покупатели 2016 года, включая cold-start | 115 867 | 2.58% | 7.79% | 10.85% |

Вывод: ALS сам по себе — слабый, но полезный baseline: у клиентов без истории
рекомендации нет вовсе, и это честно видно во втором скоупе. Финальная модель
использует ALS-рекомендацию вместе с остальными признаками, а её вклад измеряется
абляцией (см. выше).

## Артефакты обучения

`modeling.ipynb` создаёт/обновляет:

- `fastapi/saved_model.pkl` — sklearn-пайплайн для сервиса и `rec_sys.ipynb`;
- `fastapi/preprocessing_params.json` — параметры предобработки;
- `fastapi/model_version.json` — run id, дата, таргет, оба протокола, победившее плечо
  сравнения моделей, гиперпараметры, метрики, top-K и абляция ALS;
- `fastapi/replacer.json` — словарь редких категорий;
- `fastapi/personal_als.parquet` — персональные ALS-рекомендации по `ncodpers`;
- `artifacts/classification_report.txt` — полный отчёт качества (метрики, top-K, абляция, per-class PR-AUC);
- `artifacts/feature_importances.csv` — важности отобранных признаков;
- `artifacts/model_comparison.csv` — сравнение Random Forest и CatBoost на одном тесте;
- `artifacts/topk_metrics.csv` — precision@K / recall@K / NDCG@K финальной модели и бейзлайнов;
- `artifacts/als_ablation.csv` — модель с ALS-признаком и без него;
- `artifacts/split_comparison.csv` — календарный и когортный протоколы;
- `artifacts/target_diagnostics.csv` — сколько покупателей целевого снимка панель не видит;
- `artifacts/als_metrics.csv` — top-k метрики ALS-списков;
- `recommendations_analysis.md` — итоговый аналитический отчёт (также логируется в MLflow
  как артефакт обучения в `analysis/`).

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
- Панель предсказывает покупку только у клиентов, которые были в предыдущем снимке:
  покупки новых и вернувшихся клиентов (`artifacts/target_diagnostics.csv`) закрываются
  отдельной моделью холодного старта.
- Метрики top-K считаются по одному купленному продукту на строку: при нескольких
  покупках в месяце правило детерминированное (минимальный код продукта).
- Качество по продуктам с малым support (сотни объектов в test) нестабильно между
  периодами, поэтому решения по таким продуктам принимаются по нескольким прогонам.
- Бюджет подбора гиперпараметров задан временем (`OPTUNA_TIMEOUT`); при уменьшении
  бюджета сравнение моделей остаётся честным (одинаковые условия), но может не найти
  лучшие параметры. Артефакты `artifacts/*.csv`, `artifacts/classification_report.txt`,
  `recommendations_analysis.md` и JSON-файлы в `fastapi/` полностью перезаписываются
  прогоном `modeling.ipynb` — числа в них относятся к последнему запуску.
