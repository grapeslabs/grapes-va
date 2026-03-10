"""
pacs_api.py
Единое API, которое слушает события от PACS (камеры и персоны).
Камеры хранятся в памяти (словарь), персоны – в базе FR.
"""

import os
import uuid
import base64
import logging
from datetime import datetime
from flask import Flask, request, jsonify
from flask_cors import CORS
from PIL import Image
import io
import numpy as np
import cv2

from libs.pinfacekirjasto.PinFace import PinFace
from libs.DbLibrary import FRDatabase
from libs.color_logger import ColorLogger

pFace = PinFace(ffmode="mtcnn", frmode="adaface")

logger = ColorLogger("PACS_API", log_file="pacs_api.log", level=logging.INFO)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
API_PORT = int(os.getenv("API_PORT", "5000"))
PERSON_PHOTOS_PATH = os.getenv("PERSON_PHOTOS_PATH", "./person_photos")
MIN_WIDTH_PHOTO = int(os.getenv("MIN_WIDTH_PHOTO", "50"))

# Параметры детектора движения по умолчанию
MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "500"))
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "25"))
MOTION_RECORD_AFTER_TIME = int(os.getenv("MOTION_RECORD_AFTER_TIME", "3"))

fr_db = FRDatabase()

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


@app.route("/api/c1/create", methods=["POST"])
def create_camera():
    data = request.get_json(silent=True)
    if not data:
        return make_response(False, message="Invalid JSON", status_code=400)

    stream_to_parse = data.get("stream_to_parse")
    user_id = data.get("user_id")
    if not stream_to_parse or not user_id:
        return make_response(
            False,
            message="Обязательные поля: stream_to_parse, user_id",
            status_code=400,
        )

    cam_id = data.get("cam_id", generate_short_id())
    name = data.get("name", data.get("description", cam_id))

    camera_data = {
        "cam_id": cam_id,
        "name": name,
        "desc": data.get("description", name),
        "stream_to_parse": stream_to_parse,
        "user_id": user_id,
        "face_width_max": data.get("face_width_max", MIN_WIDTH_PHOTO),
        "timedelay": data.get("timedelay", 333),
        "resize": data.get("resize"),
        "crop_params": data.get("crop_params"),
        "extraqueue": data.get("extraqueue", 1),
        "status": "active",
        "motion_min_area": data.get("motion_min_area", MOTION_MIN_AREA),
        "motion_threshold": data.get("motion_threshold", MOTION_THRESHOLD),
        "motion_record_after_time": data.get(
            "motion_record_after_time", MOTION_RECORD_AFTER_TIME
        ),
    }

    # Сохраняем в БД
    fr_db.add_camera(camera_data)
    logger.info(f"Camera {cam_id} created/updated in DB")
    logger.info(f"Camera data: {camera_data}")

    return make_response(
        True,
        {"status": "success", "filename": f"{cam_id}.json", "data": camera_data},
        status_code=201,
    )


@app.route("/api/c1/list", methods=["GET"])
def list_cameras():
    user_id_filter = request.args.get("user_id")

    # Получаем камеры из БД
    cameras = fr_db.get_all_cameras(user_id_filter)

    tasks = {"queue": {}, "suspended": {}}
    for cam in cameras:
        tasks["queue"][cam["cam_id"]] = {
            "filename": f"{cam['cam_id']}.json",
            "folder": "queue",
            **cam,
        }

    return make_response(True, {"tasks": tasks})


@app.route("/api/c1/suspend", methods=["POST"])
def suspend_camera():
    data = request.get_json(silent=True)
    if not data:
        return make_response(False, message="Invalid JSON", status_code=400)

    cam_id = data.get("cam_id")
    if not cam_id:
        return make_response(False, message="cam_id required", status_code=400)

    # Помечаем камеру как suspended в БД
    if fr_db.suspend_camera(cam_id):
        logger.info(f"Camera {cam_id} suspended")
        return make_response(
            True, {"status": "success", "filename": f"{cam_id}.json", "cam_id": cam_id}
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
    person_id = request.form.get("person_id", generate_short_id())

    print(f"Параметры запроса:")
    print(f"  - user_id: {user_id}")
    print(f"  - desc: {desc}")
    print(f"  - person_id: {person_id}")
    print(f"  - количество файлов: {len(photo_files)}")

    if not user_id:
        print("Ошибка: user_id is required")
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
                    img = Image.open(io.BytesIO(img_bytes))
                    img = img.convert("RGB")
                except Exception:
                    continue

                img_resized = img.resize((112, 112), Image.Resampling.LANCZOS)
                faces = [img_resized]
                embeddings = pFace.face_recognition(faces=faces)

                if not embeddings:
                    continue

                photo_id = generate_short_id()
                filename = f"{person_id}_{photo_id}.jpg"
                file_path = os.path.join(PERSON_PHOTOS_PATH, filename)
                os.makedirs(PERSON_PHOTOS_PATH, exist_ok=True)

                with open(file_path, "wb") as f:
                    f.write(img_bytes)
                files_saved += 1

                vector_full = embeddings[0].tolist()
                vector_128 = (
                    vector_full[:128] if len(vector_full) >= 128 else vector_full
                )

                cursor.execute(
                    """
                    INSERT INTO photo (filein, person_id, photo_id, quality, photo_dttm, vector, vector128, view_photo)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s::vector, %s)
                """,
                    (
                        file_path,
                        person_id,
                        photo_id,
                        95,
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        vector_full,
                        vector_128,
                        True,
                    ),
                )

                photo_ids.append(photo_id)
                qualities.append(95)

        if not is_update and not photo_ids:
            raise Exception("No valid faces in photos")

    for path in old_photo_paths:
        try:
            if os.path.exists(path):
                os.remove(path)
        except Exception as e:
            print(f"Ошибка удаления файла {path}: {e}")

    if not is_update and not photo_ids:
        return make_response(False, message="No valid faces in photos", status_code=400)

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
    user_id = request.args.get("user_id")
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
                    logger.info(f"Удален файл: {file_path}")
                except Exception as e:
                    logger.error(f"Ошибка удаления файла {file_path}: {e}")

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
    user_id = request.args.get("user_id")
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
        logger.error(f"Ошибка получения событий: {e}")
        return make_response(False, message=str(e), status_code=500)


if __name__ == "__main__":
    logger.info(f"Запуск объединённого API на порту {API_PORT}")
    app.run(host="0.0.0.0", port=API_PORT, debug=DEBUG_MODE)
