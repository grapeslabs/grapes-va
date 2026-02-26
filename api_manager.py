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

camera_storage = {}


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
    name = data.get("name", data.get("desc", cam_id))

    camera_data = {
        "cam_id": cam_id,
        "name": name,
        "desc": data.get("desc", name),
        "stream_to_parse": stream_to_parse,
        "user_id": user_id,
        "face_width_max": data.get("face_width_max", MIN_WIDTH_PHOTO),
        "timedelay": data.get("timedelay", 333),
        "resize": data.get("resize"),
        "crop_params": data.get("crop_params"),
        "extraqueue": data.get("extraqueue", 1),
        "status": "active",
        # Параметры детектора движения
        "motion_min_area": data.get("motion_min_area", MOTION_MIN_AREA),
        "motion_threshold": data.get("motion_threshold", MOTION_THRESHOLD),
        "motion_record_after_time": data.get(
            "motion_record_after_time", MOTION_RECORD_AFTER_TIME
        ),
    }

    camera_storage[cam_id] = camera_data
    logger.info(f"Camera {cam_id} created/updated")
    logger.info(f"Camera data: {camera_data}")

    return make_response(
        True,
        {"status": "success", "filename": f"{cam_id}.json", "data": camera_data},
        status_code=201,
    )


@app.route("/api/c1/list", methods=["GET"])
def list_cameras():
    cam_id_filter = request.args.get("cam_id")
    user_id_filter = request.args.get("user_id")

    tasks = {"queue": {}, "suspended": {}}
    for cid, cam in camera_storage.items():
        if cam_id_filter and cid != cam_id_filter:
            continue
        if user_id_filter and str(cam["user_id"]) != user_id_filter:
            continue
        tasks["queue"][cid] = {"filename": f"{cid}.json", "folder": "queue", **cam}

    return make_response(True, {"tasks": tasks})


@app.route("/api/c1/suspend", methods=["POST"])
def suspend_camera():
    data = request.get_json(silent=True)
    if not data:
        return make_response(False, message="Invalid JSON", status_code=400)

    cam_id = data.get("cam_id")
    if not cam_id:
        return make_response(False, message="cam_id required", status_code=400)

    if cam_id not in camera_storage:
        return make_response(
            False, message=f"Задание с cam_id '{cam_id}' не найдено", status_code=404
        )

    del camera_storage[cam_id]
    logger.info(f"Camera {cam_id} suspended")
    return make_response(
        True, {"status": "success", "filename": f"{cam_id}.json", "cam_id": cam_id}
    )


@app.route("/api/v1/person/add", methods=["POST"])
def add_person():
    user_id = request.form.get("user_id")
    desc = request.form.get("desc", "")
    photo_file = request.files.get("photo")

    if not user_id:
        return make_response(False, message="user_id is required", status_code=400)
    if not photo_file:
        return make_response(False, message="photo is required", status_code=400)

    img_bytes = photo_file.read()
    img = Image.open(io.BytesIO(img_bytes)).convert("RGB")
    faces = [img]
    embeddings = pFace.face_recognition(faces=faces)
    if not embeddings:
        return make_response(False, message="No face detected", status_code=400)

    vector_full = embeddings[0].tolist()
    vector_128 = vector_full[:128] if len(vector_full) >= 128 else vector_full

    person_id = generate_short_id()
    photo_id = generate_short_id()

    filename = f"{person_id}_{photo_id}.jpg"
    file_path = os.path.join(PERSON_PHOTOS_PATH, filename)
    os.makedirs(PERSON_PHOTOS_PATH, exist_ok=True)
    with open(file_path, "wb") as f:
        f.write(img_bytes)

    percone_dttm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with fr_db._get_cursor() as cursor:
        cursor.execute(
            """
            INSERT INTO percone (user_id, person_id, description, tag, percone_dttm, view_percone)
            VALUES (%s, %s, %s, %s, %s, %s)
        """,
            (user_id, person_id, desc, desc, percone_dttm, True),
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

    return make_response(
        True,
        {
            "person_id": person_id,
            "photo_id": [photo_id],
            "quality": [95],
            "user_id": user_id,
            "info_msg": "Person added successfully",
        },
        status_code=201,
    )


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

    percone_deleted, photo_deleted = fr_db.delete_person(person_id)

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
