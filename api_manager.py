"""
pacs_api.py
Единое API, которое слушает события от PACS (камеры и персоны).
Камеры хранятся в памяти (словарь), персоны – в базе FR.
"""

import os
import uuid
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import cv2
from dotenv import load_dotenv

from libs.pinfacekirjasto.PinFace import PinFace
from libs.DbLibrary import FRDatabase
from libs.loglib import capture_message, shutdown
from psycopg2.extras import RealDictCursor
import atexit

load_dotenv()

try:
    pFace = PinFace(ffmode="mtcnn", frmode="adaface")
    capture_message("info", "PinFace initialized successfully")
except Exception as e:
    capture_message("error", f"Failed to initialize PinFace: {e}")
    raise SystemExit(1)

atexit.register(shutdown)

MODE = os.getenv("MODE", "pacs")
DEBUG_MODE = str(os.getenv("DEBUG_MODE", "false")).lower() == "true"
API_PORT = int(os.getenv("API_PORT", "5000"))
PERSON_AVATARS_PATH = os.getenv("PERSON_AVATARS_PATH", "./person_photos")
MIN_WIDTH_PHOTO = int(os.getenv("MIN_WIDTH_PHOTO", "50"))

# Параметры детектора движения по умолчанию
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "500"))
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "25"))
MOTION_RECORD_AFTER_TIME = int(os.getenv("MOTION_RECORD_AFTER_TIME", "3"))

SAVE_PHOTOS = os.getenv("SAVE_PHOTOS", "false").lower() == "true"
PHOTOS_PATH = os.getenv("PHOTOS_PATH", "./person_avatars")

try:
    fr_db = FRDatabase()
    capture_message("info", "Database connection established")
except Exception as e:
    capture_message("error", f"Failed to connect to database: {e}")
    raise SystemExit(1)


# ===== Применение миграций структуры БД ===== k3
migrations_file = os.path.join(os.path.dirname(__file__), "libs", "update_schema.py")
if os.path.isfile(migrations_file):
    try:
        from libs.update_schema import apply_migrations

        apply_migrations(fr_db, MODE, capture_message)
    except ImportError as e:
        capture_message("error", f"Не удалось импортировать apply_migrations: {e}")
    except Exception as e:
        capture_message("error", f"Ошибка при применении миграций: {e}")
else:
    # capture_message("debug", "Файл миграций отсутствует (возможно, уже применён)")
    pass
# ===== Конец блока миграций ===== k3


if MODE == "pin":
    try:
        from libs.FrFile import FrFile

        fr_fl = FrFile()
        capture_message("info", "Camera files manager established")
    except Exception as e:
        capture_message("error", f"Failed camera files manager: {e}")
        raise SystemExit(1)


app = Flask(__name__)
CORS(app)


def make_response(
    ok: bool, data: dict = None, message: str = "", status_code: int = 200
):
    resp = {"ok": ok}
    if data:
        resp.update(data)
    if message:
        resp["info_msg"] = message
    return jsonify(resp), status_code


def generate_short_id() -> str:
    return str(uuid.uuid4())[24:]

def mask_rtsp_credentials(url: str) -> str:
    """
    Маскирует логин и пароль в RTSP URL, заменяя их на символы '#'.

    Аргументы:
        url: RTSP-адрес вида rtsp://логин:пароль@хост/путь

    Возвращает:
        Адрес с замаскированными логином и паролем.
        При неверном формате возвращает исходную строку.

    Пример:
        >>> mask_rtsp_credentials("rtsp://admin:123456@192.168.1.64:554/ISAPI/Streaming/Channels/101")
        'rtsp://#####:######@192.168.1.64:554/ISAPI/Streaming/Channels/101'
    """
    if not url:
        return url
    if not url.startswith("rtsp://"):
        return url
    at = url.find('@')
    if at == -1:
        return url
    col = url.find(':', 7)
    if col == -1 or col > at:
        return url
    return f"rtsp://{'#' * (col - 7)}:{'#' * (at - col - 1)}{url[at:]}"

def camera_to_nested(cam: dict, mask_rtsp: bool = True) -> dict:
    """Преобразует плоский JSON камеры во вложенный формат"""
    data = {
        "stream_info": {
            "id": cam.get("cam_id"),
            "url": cam.get("stream_to_parse", ""),
            "name": cam.get("name"),
            "description": cam.get("description"),
            "timedelay": cam.get("timedelay", 333),
            "resize": cam.get("resize"),
        },
        "user_info": {
            "id": cam.get("user_id"),
            "mail": cam.get("user_mail"),
        },
        "detection_face": {
            "is_detection": cam.get("is_detection", True),
            "is_recognize": cam.get("is_recognize", True),
            "min_width_photo": cam.get("face_width_min", 50),
            "face_width_max": cam.get("face_width_max", 45),
            "min_area": cam.get("motion_min_area", 500),
            "threshold": cam.get("motion_threshold", 25),
            "moving_duration_after": cam.get("motion_record_after_time", 3),
            "cache_face_time": cam.get("cache_face_time", 30),
            "cache_face_max": cam.get("cache_face_max", 20),
            "zone": cam.get("crop_params"),
            "timedelay": cam.get("timedelay", 333),
            "resize": cam.get("resize"),
        },
        "detection_figure": {
            "is_active": cam.get("detection_figure_active", False),
            "direction": cam.get("detection_figure_direction", "LRBTA"),
            "zones": cam.get("detection_figure_zones", []),
        },
        "debug": {
            "write_thumbnails": cam.get("write_thumbnails", False),
            "write_frame": cam.get("write_frame", False),
        },
    }

    if mask_rtsp:
        url = data['stream_info']['stream_to_parse']
        data['stream_info']['stream_to_parse'] = mask_rtsp_credentials(url)

    return data

@app.errorhandler(404)
def abort_404(e):
    """Обработчик для 404 ошибок."""
    capture_message("info", f"Use 404 page {str(e)}")
    return make_response(False, message="Use 404 page", status_code=404)


# Проверка работоспособности api


@app.route("/api/c1/test", methods=["POST", "GET"])
@app.route("/api/v1/test", methods=["POST", "GET"])
@app.route("/api/v1/person/test", methods=["POST", "GET"])
def api_test():
    return jsonify({"ok": True})


@app.route("/api/c1/create", methods=["POST"])
@app.route("/api/v1/camera/create", methods=["POST"])
def create_camera():
    data = request.get_json(silent=True)
    if not data:
        return make_response(False, message="Invalid JSON", status_code=400)

    # === Секция stream_info ===
    stream_info = data.get("stream_info", {})
    stream_to_parse = stream_info.get("url")
    if not stream_to_parse:
        stream_to_parse = data.get("stream_to_parse")

    cam_id = stream_info.get("id") or data.get("cam_id", generate_short_id())
    name = stream_info.get("name") or data.get(
        "name", stream_info.get("description", cam_id)
    )

    # === Секция user_info ===
    user_info = data.get("user_info", {})
    user_id = user_info.get("id") or data.get("user_id", "1")
    user_mail = user_info.get("mail")

    if not stream_to_parse or not user_id:
        return make_response(
            False,
            message="Обязательные поля: stream_info.url, user_info.id",
            status_code=400,
        )

    # === Секция detection_face ===
    detection_face = data.get("detection_face", {})
    face_width_min = detection_face.get("min_width_photo", MIN_WIDTH_PHOTO)
    face_width_max = detection_face.get("face_width_max", 45)
    timedelay = detection_face.get("timedelay", stream_info.get("timedelay", 333))
    resize = detection_face.get("resize") or stream_info.get("resize")
    detection_zone = detection_face.get("zone")
    is_detection = detection_face.get("is_detection", True)
    is_recognize = detection_face.get("is_recognize", True)
    moving_duration_after = detection_face.get("moving_duration_after", 4)
    cache_face_time = detection_face.get("cache_face_time", 30)
    cache_face_max = detection_face.get("cache_face_max", 20)
    threshold = detection_face.get("threshold", MOTION_THRESHOLD)
    min_area = detection_face.get("min_area", MOTION_MIN_AREA)

    # === Секция detection_figure ===
    detection_figure = data.get("detection_figure", {})
    detection_figure_active = detection_figure.get("is_active", False)
    detection_figure_direction = detection_figure.get("direction", "LRBTA")
    detection_figure_zones = detection_figure.get("zones", [])

    # === Секция debug ===
    debug_info = data.get("debug", {})
    write_thumbnails = debug_info.get("write_thumbnails", False)
    write_frame = debug_info.get("write_frame", False)

    camera_data = {
        "cam_id": cam_id,
        "name": name,
        "desc": stream_info.get("description", name),
        "stream_to_parse": stream_to_parse,
        "user_id": user_id,
        "user_mail": user_mail,
        "face_width_min": face_width_min,
        "face_width_max": face_width_max,
        "timedelay": timedelay,
        "resize": resize,
        "crop_params": detection_zone,
        "extraqueue": data.get("extraqueue", 1),
        "status": "active",
        "motion_min_area": min_area,
        "motion_threshold": threshold,
        "motion_record_after_time": moving_duration_after,
        "is_detection": is_detection,
        "is_recognize": is_recognize,
        "cache_face_time": cache_face_time,
        "cache_face_max": cache_face_max,
        "detection_figure_active": detection_figure_active,
        "detection_figure_direction": detection_figure_direction,
        "detection_figure_zones": detection_figure_zones,
        "write_thumbnails": write_thumbnails,
        "write_frame": write_frame,
    }

    # Сохраняем в БД
    if MODE == "pacs":
        fr_db.add_camera(camera_data)
    elif MODE == "pin":
        fr_fl.add_camera(camera_data)

    capture_message("info", f"Camera {cam_id} created/updated in DB")

    return make_response(
        True,
        {"status": "success", "filename": f"{cam_id}.json", "data": camera_data},
        status_code=201,
    )


@app.route("/api/c1/list", methods=["GET"])
@app.route("/api/v1/camera/list", methods=["GET"])
def list_cameras():
    user_id_filter = request.args.get("user_id")

    if MODE == "pacs":
        cameras = fr_db.get_all_cameras(user_id_filter)
    elif MODE == "pin":
        cameras = fr_fl.get_all_cameras(user_id_filter)

    tasks = {"queue": {}, "suspended": {}}
    for cam in cameras:
        tasks["queue"][cam["cam_id"]] = {
            "filename": f"{cam['cam_id']}.json",
            "folder": "queue",
            **camera_to_nested(cam),
        }

    return make_response(True, {"tasks": tasks})


@app.route("/api/c1/suspend", methods=["POST"])
@app.route("/api/v1/camera/suspend", methods=["POST"])
def suspend_camera():
    data = request.get_json(silent=True)
    if not data:
        return make_response(False, message="Invalid JSON", status_code=400)

    cam_id = data.get("cam_id")
    if not cam_id:
        return make_response(False, message="cam_id required", status_code=400)

    if MODE == "pacs":
        # Помечаем камеру как suspended в БД
        if fr_db.suspend_camera(cam_id):
            capture_message("info", f"Camera {cam_id} suspended")
            return make_response(
                True,
                {"status": "success", "filename": f"{cam_id}.json", "cam_id": cam_id},
            )
    elif MODE == "pin":
        # удаление камеры через сервер пин
        if fr_fl.suspend_camera(cam_id):
            capture_message("info", f"Camera {cam_id} suspended")
            return make_response(
                True,
                {"status": "success", "filename": f"{cam_id}.json", "cam_id": cam_id},
            )
    else:
        return make_response(
            False, message=f"Задание с cam_id '{cam_id}' не найдено", status_code=404
        )


@app.route("/api/v1/person/add", methods=["POST"])
def add_person():
    user_id = request.form.get("user_id")
    desc = request.form.get("desc", "")
    photo_files = request.files.getlist("photos")

    # person_id = request.form.get("person_id", generate_short_id())
    person_id = request.form.get("person_id")  # может быть None
    is_external_person_id = person_id is not None  # запоминаем, был ли передан
    if not is_external_person_id:
        person_id = generate_short_id()

    capture_message(
        "info",
        f"Добавление персоны: user_id={user_id}, person_id={person_id}, файлов={len(photo_files)}",
    )

    if not user_id:
        capture_message("error", "user_id is required")
        return make_response(False, message="user_id is required", status_code=400)

    percone_dttm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    is_update = False
    old_photo_paths = []

    with fr_db._get_cursor() as cursor:
        cursor.execute(
            "SELECT person_id FROM percone WHERE person_id = %s", (person_id,)
        )
        existing = cursor.fetchone()

        if existing:
            is_update = True
            cursor.execute(
                """
                UPDATE percone
                SET description = %s, tag = %s, percone_dttm = %s, view_percone = %s
                WHERE person_id = %s
            """,
                (desc, desc, percone_dttm, True, person_id),
            )

            if photo_files:
                cursor.execute(
                    "SELECT filein FROM photo WHERE person_id = %s", (person_id,)
                )
                old_photo_paths = [row[0] for row in cursor.fetchall()]
                cursor.execute("DELETE FROM photo WHERE person_id = %s", (person_id,))
        else:
            cursor.execute(
                """
                INSERT INTO percone (user_id, person_id, description, tag, percone_dttm, view_percone)
                VALUES (%s, %s, %s, %s, %s, %s)
            """,
                (user_id, person_id, desc, desc, percone_dttm, True),
            )

        photo_ids = []
        qualities = []
        files_saved = 0

        if photo_files:
            for photo_file in photo_files:
                img_bytes = photo_file.read()
                try:
                    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
                except Exception as e:
                    capture_message("error", f"Ошибка открытия изображения: {e}")
                    # Сохраняем оригинал даже при ошибке
                    if SAVE_PHOTOS:
                        avatar_filename = f"error_{person_id}_{generate_short_id()}.jpg"
                        avatar_path = os.path.join(PHOTOS_PATH, avatar_filename)
                        with open(avatar_path, "wb") as f:
                            f.write(img_bytes)
                    continue

                # Поиск лиц на изображении
                bboxes, faces, _ = pFace.face_detection(img)

                # Сохраняем аватар с рамками если нужно
                if SAVE_PHOTOS:
                    img_cv = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

                    # Рисуем рамки вокруг всех найденных лиц
                    for bbox in bboxes:
                        left = int(bbox[0])
                        top = int(bbox[1])
                        right = int(bbox[2])
                        bottom = int(bbox[3])
                        cv2.rectangle(
                            img_cv,
                            (left - 10, top - 10),
                            (right + 10, bottom + 10),
                            (0, 255, 0),
                            2,
                        )

                    avatar_filename = f"{person_id}_{generate_short_id()}.jpg"
                    avatar_path = os.path.join(PHOTOS_PATH, avatar_filename)
                    cv2.imwrite(avatar_path, img_cv)

                    if not faces:
                        capture_message(
                            "warning",
                            f"Все найденные лица на {photo_file.filename} меньше минимальной ширины {MIN_WIDTH_PHOTO}px",
                        )
                        continue

                if not faces:
                    capture_message(
                        "warning",
                        f"Лица не найдены на изображении {photo_file.filename}",
                    )
                    continue

                if len(faces) > 1:
                    capture_message(
                        "warning",
                        f"Найдено более одного лица ({len(faces)}) на изображении {photo_file.filename}. Поиск лица с максимальной шириной",
                    )

                    # Находим индекс лица с максимальной шириной
                    max_width = -1
                    best_index = -1

                    for idx, bbox in enumerate(bboxes):
                        width = int(bbox[2]) - int(bbox[0])
                        if width > max_width:
                            max_width = width
                            best_index = idx

                    faces = [faces[best_index]]
                    bboxes = [bboxes[best_index]]

                    capture_message(
                        "debug", f"Выбрано лицо #{best_index+1}, ширина: {max_width}"
                    )

                # Изображение подходит - строим вектор для найденного лица
                face = faces[0]
                embeddings = pFace.face_recognition(faces=[face])

                if not embeddings:
                    capture_message(
                        "warning",
                        f"Не удалось построить вектор для лица на {photo_file.filename}",
                    )
                    continue

                photo_id = generate_short_id()
                filename = f"{person_id}_{photo_id}.jpg"
                file_path = os.path.join(PERSON_AVATARS_PATH, filename)
                os.makedirs(PERSON_AVATARS_PATH, exist_ok=True)

                # Сохраняем только вырезанное лицо
                face_cv = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                cv2.imwrite(file_path, face_cv)
                files_saved += 1

                vector_full = [round(float(x), 5) for x in embeddings[0]]

                cursor.execute(
                    """
                    INSERT INTO photo (filein, person_id, photo_id, quality, photo_dttm, vector, view_photo)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s)
                """,
                    (
                        file_path,
                        person_id,
                        photo_id,
                        95,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        vector_full,
                        True,
                    ),
                )

                photo_ids.append(photo_id)
                qualities.append(95)
                capture_message(
                    "info",
                    f"Фото сохранено: person_id={person_id}, photo_id={photo_id}, файл={filename}",
                )

        # ===== НОВЫЙ БЛОК: обновление unknown, если person_id передан извне =====
        if is_external_person_id:
            cursor.execute(
                "UPDATE unknown SET view_percone = false WHERE uuid = %s AND view_percone = true",
                (person_id,),
            )
            if cursor.rowcount:
                capture_message(
                    "info", f"Unknown record {person_id} marked as converted"
                )
        # ===== КОНЕЦ НОВОГО БЛОКА =====

        if not is_update and not photo_ids:
            return make_response(
                False, message="No valid faces in photos", status_code=400
            )

    for path in old_photo_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            capture_message("warning", f"Ошибка удаления файла {path}: {e}")

    response = {
        "person_id": person_id,
        "photo_id": photo_ids,
        "quality": qualities,
        "user_id": user_id,
        "info_msg": (
            "Person updated successfully" if is_update else "Person added successfully"
        ),
        "is_update": is_update,
    }

    action = "обновлена" if is_update else "добавлена"
    capture_message(
        "info",
        f"Персона {action}: person_id={person_id}, фото={len(photo_ids)}, user_id={user_id}",
    )

    return make_response(True, response, status_code=201)


@app.route("/api/v1/person/getinfo", methods=["GET"])
def get_person_info():
    user_id = request.args.get("user_id")
    person_id = request.args.get("person_id")

    if not user_id:
        return make_response(False, message="user_id is required", status_code=400)

    rows = fr_db.get_person_info(user_id, person_id)
    percones = []
    for row in rows:
        percones.append(
            {
                "user_id": row["user_id"],
                "person_id": row["person_id"],
                "description": row["description"],
                "tags": row["tag"],
                "dttm": row["percone_dttm"],
                "photos": row["photos"] or [],
                "count_photos": len(row["photos"] or []),
            }
        )

    return make_response(True, {"percones": percones, "count_percones": len(percones)})


@app.route("/api/v1/person/getphoto", methods=["GET"])
def get_photo_info():
    user_id = request.args.get("user_id", "1")
    person_id = request.args.get("person_id")
    photo_id = request.args.get("photo_id")

    if not user_id:
        return make_response(False, message="user_id is required", status_code=400)

    photos = fr_db.get_photo_info(user_id, person_id, photo_id)
    return make_response(True, {"Photos": photos, "count_photos": len(photos)})


@app.route("/api/v1/person/del", methods=["DELETE"])
def delete_person():
    user_id = request.args.get("user_id")
    person_id = request.args.get("person_id")

    if not user_id or not person_id:
        return make_response(
            False, message="user_id and person_id required", status_code=400
        )

    # Сначала удаляем все фото персоны
    with fr_db._get_cursor() as cursor:
        # Получаем пути к файлам фото перед удалением
        cursor.execute(
            "SELECT filein, photo_id FROM photo WHERE person_id = %s", (person_id,)
        )
        photos = cursor.fetchall()

        # Удаляем записи из photo
        cursor.execute("DELETE FROM photo WHERE person_id = %s", (person_id,))
        photo_deleted = cursor.rowcount

        # Удаляем запись из percone
        cursor.execute("DELETE FROM percone WHERE person_id = %s", (person_id,))
        percone_deleted = cursor.rowcount

        # Удаляем физические файлы
        for photo in photos:
            file_path = photo[0]
            if file_path and os.path.exists(file_path):
                try:
                    os.remove(file_path)
                except Exception as e:
                    capture_message("error", f"Ошибка удаления файла {file_path}: {e}")

    if percone_deleted:
        capture_message(
            "info",
            f"Персона удалена: person_id={person_id}, фото={photo_deleted}, user_id={user_id}",
        )
    else:
        capture_message(
            "warning", f"Персона не найдена для удаления: person_id={person_id}"
        )

    return make_response(
        True,
        {
            "percone_count_delete": percone_deleted,
            "photo_count_delete": photo_deleted,
            "person_id": person_id,
            "user_id": user_id,
            "info_msg": "Person deleted" if percone_deleted else "Person not found",
        },
    )


@app.route("/api/v1/person/delphoto", methods=["DELETE"])
def delete_photo():
    user_id = request.args.get("user_id", "1")
    person_id = request.args.get("person_id")
    photo_id = request.args.get("photo_id")

    if not user_id or not person_id or not photo_id:
        return make_response(
            False, message="user_id, person_id, photo_id required", status_code=400
        )

    photos = fr_db.get_photo_info(user_id, person_id, photo_id)
    if not photos:
        return make_response(False, message="Photo not found", status_code=404)

    deleted, file_path = fr_db.delete_photo(photo_id)
    if file_path and os.path.exists(file_path):
        os.remove(file_path)

    return make_response(
        True,
        {
            "photo_count_delete": deleted,
            "person_id": person_id,
            "photo_id": photo_id,
            "user_id": user_id,
            "info_msg": "Photo deleted" if deleted else "Photo not found",
        },
    )


@app.route("/api/events", methods=["GET"])
def get_events():
    if MODE != "pacs":
        return make_response(
            False, message=f"Not using endpoints in mode {MODE}", status_code=404
        )

    limit = request.args.get("limit", default=100, type=int)
    offset = request.args.get("offset", default=0, type=int)
    camera_id = request.args.get("camera_id")
    from_date = request.args.get("from")
    to_date = request.args.get("to")

    query = "SELECT * FROM analytics_events WHERE 1=1"
    params = []
    if camera_id:
        query += " AND camera_id = %s"
        params.append(camera_id)
    if from_date:
        query += " AND datetime >= %s"
        params.append(from_date)
    if to_date:
        query += " AND datetime <= %s"
        params.append(to_date)
    query += " ORDER BY datetime DESC LIMIT %s OFFSET %s"
    params.extend([limit, offset])

    try:
        with fr_db._get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
        return make_response(True, {"events": rows, "count": len(rows)})
    except Exception as e:
        capture_message("error", f"Ошибка получения событий: {e}")
        return make_response(False, message=str(e), status_code=500)


if __name__ == "__main__":
    capture_message("info", f"{MODE} API starting...", force_sentry=True)

    if SAVE_PHOTOS:
        try:
            os.makedirs(PHOTOS_PATH, exist_ok=True)
        except Exception as e:
            capture_message("warning", f"Ошибка создания папки: {e}")

    capture_message("info", f"{MODE} API started on port {API_PORT}", force_sentry=True)
    app.run(host="0.0.0.0", port=API_PORT, debug=DEBUG_MODE)
