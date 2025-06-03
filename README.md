# Отслеживание бездомных собак

Проект для обработки изображений и видеопотоков с целью детекции бездомных собак. Включает Telegram-бота и API.

## Стек

- YOLO
- FastAPI
- PostgreSQL
- SQLAlchemy
- Docker
- OpenCV
- PyTorch

## Telegram-бот

Запуск:

```bash
python src/detection_cycle.py
```

## API

Запуск:

```bash
python -m uvicorn src.api:app --reload
```

Доступные эндпоинты:

- GET /docs/ - swagger документация
- POST /detection/image/ - загрузка изображения для обработки
- POST /detection/video/ - загрузка видео
- POST /detection/url_stream/ - загрузка URL видеопотока, возвращает SSE-поток результатов, кадры можно получить из /frame/
- GET /frame/ - получение кадра по ID

## Установка

```bash
pip install -r requirements.txt
```
