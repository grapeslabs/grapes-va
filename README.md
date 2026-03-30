# docker_face

Система распознавания лиц: захват видео с камер, детекция, идентификация, запись событий.

## Установка и запуск

### 1. Установить Git

### 2. Установить Docker

### 3. Проверить доступ к гиту

```
ssh git@git.pinspb.ru
```

Должно вернуть:

```
PTY allocation request failed on channel 0
```

### 4. Клонировать репозиторий

```
git clone git@git.pinspb.ru:a.lishchina/docker_fr.git
cd docker_face
```

### 5. Создать файл окружения

```
cp .env.example .env
```

### 6. Настроить .env

# PostgreSQL Connection Settings
PG_HOST=localhost
PG_PORT=5432
PG_USER=postgres
PG_PASSWORD=postgres
PG_DATABASE = fr

# API settings
PACS_API_URL=http://localhost:5000
API_PORT=5000

# Storage paths
PERSON_AVATARS_PATH=
PHOTOS_PATH=
THUMBNAIL_PATH=
SAVE_PHOTOS=true

# Logging Settings
LOGGING_TYPE= local - локальное логирование, sentry - local + отправка важных событий в sentry

# Sentry - заполнить при использовании (формат: https://key@sentry.host/project_id)
SENTRY_DSN=
SENTRY_ENVIRONMENT=development
SENTRY_RELEASE=docker_face@1.0.0
SENTRY_TRACES_SAMPLE_RATE=1.0
SENTRY_PROFILES_SAMPLE_RATE=0
SENTRY_DEBUG=false

### 7. Запустить

```
docker-compose up -d
```

### 8. Проверить

```
docker-compose ps
```

Должны работать 3 контейнера: `core`, `api`, `postgres`

### 9. Логи, остановка, удаление

```
docker-compose logs <имя_контейнера>
docker-compose down
docker-compose down -v
```

## API

Базовый URL: `http://localhost:5000`

Формат ответа:

```json
{"ok": true, ...данные...}
{"ok": false, "info_msg": "текст ошибки"}
```

### Камеры

#### POST /api/c1/create

Создать камеру.

Тело (JSON):

```json
{
  "stream_to_parse": "rtsp://admin:pass@192.168.1.100:554/stream",
  "user_id": "1",
  "cam_id": "cam001",
  "name": "Камера 1",
  "face_width_min": 50,
  "timedelay": 333,
  "motion_min_area": 500,
  "motion_threshold": 25,
  "motion_record_after_time": 3
}
```

`stream_to_parse` и `user_id` — обязательные.

Ответ (201):

```json
{"ok": true, "status": "success", "data": {...}}
```

#### GET /api/c1/list

Список камер.

Параметры: `user_id` (опционально)

```
curl "http://localhost:5000/api/c1/list?user_id=1"
```

Ответ (200):

```json
{"ok": true, "tasks": {"queue": {...}, "suspended": {...}}}
```

#### POST /api/c1/suspend

Приостановить камеру.

Тело (JSON):

```json
{"cam_id": "cam001"}
```

Ответ (200):

```json
{"ok": true, "status": "success", "cam_id": "cam001"}
```

### Персоны

#### POST /api/v1/person/add

Добавить или обновить персону. Отправка файлов через `multipart/form-data`.

```
curl -X POST http://localhost:5000/api/v1/person/add \
  -F "user_id=1" \
  -F "person_id=ivanov" \
  -F "desc=Иванов Иван" \
  -F "photos=@photo1.jpg" \
  -F "photos=@photo2.jpg"
```

`user_id` — обязательный.

Ответ (201):

```json
{
  "ok": true,
  "person_id": "ivanov",
  "photo_id": ["abc123"],
  "quality": [95],
  "is_update": false,
  "info_msg": "Person added successfully"
}
```

#### GET /api/v1/person/getinfo

Информация о персоне.

```
curl "http://localhost:5000/api/v1/person/getinfo?user_id=1&person_id=ivanov"
```

`user_id` — обязательный.

Ответ (200):

```json
{"ok": true, "percones": [...], "count_percones": 1}
```

#### GET /api/v1/person/getphoto

Получить фото.

```
curl "http://localhost:5000/api/v1/person/getphoto?user_id=1&person_id=ivanov&photo_id=abc123"
```

Ответ (200):

```json
{"ok": true, "Photos": [...], "count_photos": 1}
```

#### DELETE /api/v1/person/del

Удалить персону и все её фото.

```
curl -X DELETE "http://localhost:5000/api/v1/person/del?user_id=1&person_id=ivanov"
```

`user_id` и `person_id` — обязательные.

Ответ (200):

```json
{
  "ok": true,
  "percone_count_delete": 1,
  "photo_count_delete": 2,
  "info_msg": "Person deleted"
}
```

#### DELETE /api/v1/person/delphoto

Удалить одно фото.

```
curl -X DELETE "http://localhost:5000/api/v1/person/delphoto?user_id=1&person_id=ivanov&photo_id=abc123"
```

`person_id` и `photo_id` — обязательные.

Ответ (200):

```json
{"ok": true, "photo_count_delete": 1, "info_msg": "Photo deleted"}
```

### События

#### GET /api/events

Список событий распознавания.

```
curl "http://localhost:5000/api/events?limit=50&offset=0&camera_id=cam001&from=2026-01-01&to=2026-12-31"
```

Ответ (200):

```json
{"ok": true, "events": [...], "count": 10}
```

## Коды ошибок

| Код | Причина |
|-----|---------|
| 400 | Невалидный JSON / отсутствуют обязательные поля / лица не найдены на фото |
| 404 | Камера / персона / фото не найдены |
| 500 | Ошибка сервера (БД, внутренняя ошибка) |

Примеры ответов с ошибками:

```json
{"ok": false, "info_msg": "Invalid JSON"}
{"ok": false, "info_msg": "Обязательные поля: stream_to_parse, user_id"}
{"ok": false, "info_msg": "user_id is required"}
{"ok": false, "info_msg": "cam_id required"}
{"ok": false, "info_msg": "No valid faces in photos"}
{"ok": false, "info_msg": "Задание с cam_id 'xxx' не найдено"}
{"ok": false, "info_msg": "Photo not found"}
{"ok": false, "info_msg": "user_id and person_id required"}
```

## Устранение неполадок

| Проблема | Решение |
|----------|---------|
| postgres не стартует | `docker-compose logs postgres`, проверить порт 5432 |
| api/core не подключается к БД | Проверить `PG_USER`, `PG_PASSWORD` в `.env` |
| core не видит камеру | Проверить `stream_to_parse` в БД, сетевую доступность RTSP |
| Ошибка инициализации камеры | Проверить наличие поддержки ffmpeg кодека в системе |
| pgvector не найден | Образ: `pgvector/pgvector:pg15` |
| Мало распознанных лиц | Проверить освещение, качество камеры |
| Много ложных срабатываний | Уменьшить `EUCLIDEAN_THRESHOLD` в `.env` (например 0.4) |
| Sentry не работает | Проверить `SENTRY_DSN`, `LOGGING_TYPE=sentry` |
