# Образ микросервиса рекомендаций.
# Сборка из корня репозитория (контекст — корень):
#   docker build -f fastapi/Dockerfile .
# Проще — через compose (см. README.md):
#   docker compose -f fastapi/docker-compose.yaml up --build
FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

WORKDIR /app

COPY fastapi/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY app1.py preprocessing.py ./
# Артефакты обучения (создаёт modeling.ipynb). Если saved_model.pkl нет,
# сервис всё равно поднимется: /predict ответит 503, /health — degraded.
COPY fastapi/ ./fastapi/

EXPOSE 8079

# uvloop + httptools: C-реализации event loop и HTTP-парсера — без них
# конкурентные запросы упираются в чистопитонный h11 и «карусель» GIL.
# Access-log выключен: наблюдение за запросами — через /metrics (Prometheus),
# а логирование каждого запроса заметно ест латентность под нагрузкой.
CMD uvicorn app1:app --host 0.0.0.0 --port ${THE_PORT:-8079} \
    --loop uvloop --http httptools --no-access-log
