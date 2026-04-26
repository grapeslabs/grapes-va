"""
pacs_core.py
Основной сервис: захват видео с камер, детекция лиц, идентификация, запись событий.
Получает список камер через API PACS.
Использует детектор движения для снижения нагрузки.
"""

import os
import time
import cv2
import threading
import logging
import uuid
import queue
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

from libs.pinfacekirjasto.PinFace import PinFace
from libs.detect_motion import MotionDetector
from libs.loglib import capture_message, shutdown
from libs.DbLibrary import FRDatabase
from libs.camstream import VideoCapture
from libs.FixedSizeList import FixedSizeList
import atexit

load_dotenv()


try:
    pFace = PinFace(ffmode="mtcnn", frmode="adaface", fmode=['sface'])
    capture_message("info", "PinFace initialized successfully")
except Exception as e:
    capture_message("error", f"Failed to initialize PinFace: {e}")
    raise SystemExit(1)

atexit.register(shutdown)

# FixedSizeList для коротких векторов sface (128)
face_short_cache = FixedSizeList(max_size=50, time_len=5)
EUCLIDIAN_DISTANCE_SEPARATING = 0.9

try:
    db = FRDatabase()
    capture_message("info", "Database connection established")
except Exception as e:
    capture_message("error", f"Failed to connect to database: {e}")
    raise SystemExit(1)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
MODE = os.getenv("MODE", "pacs")
MIN_WIDTH_PHOTO = int(os.getenv("MIN_WIDTH_PHOTO", "50"))
EUCLIDEAN_THRESHOLD = float(os.getenv("EUCLIDEAN_THRESHOLD", "0.6"))
MAX_FACES_IN_LIST = int(os.getenv("MAX_FACES_IN_LIST", "25"))
THUMBNAIL_PATH = os.getenv("THUMBNAIL_PATH", "input/thumbnails/")
PACS_API_URL = os.getenv("PACS_API_URL", "http://localhost:5000")
CAMERA_POLL_INTERVAL = int(os.getenv("CAMERA_POLL_INTERVAL", "5"))

os.makedirs(THUMBNAIL_PATH, exist_ok=True)

migrations_file = os.path.join(os.path.dirname(__file__), "libs", "update_schema.py")

if os.path.isfile(migrations_file):
    try:
        from libs.update_schema import apply_migrations
        apply_migrations(db, MODE, capture_message)
    except ImportError as e:
        capture_message("error", f"Не удалось импортировать apply_migrations: {e}")
    except Exception as e:
        capture_message("error", f"Ошибка при применении миграций: {e}")
else:
    #capture_message("debug", "Файл миграций отсутствует (возможно, уже применён)")
    pass

cameras = {}
cameras_lock = threading.Lock()
shared_queue = None


def fetch_cameras_from_db():
    """Читает камеры напрямую из БД"""
    try:
        cameras_list = []
        rows = db.get_all_cameras()  # используем новый метод

        for cam in rows:
            cameras_list.append(
                {
                    "cam_id": cam["cam_id"],
                    "name": cam["name"],
                    "stream_to_parse": cam["stream_to_parse"],
                    "user_id": cam.get("user_id", "1"),
                    "face_width_min": cam.get("face_width_min", MIN_WIDTH_PHOTO),
                    "timedelay": cam.get("timedelay", 333),
                    "motion_min_area": cam.get("motion_min_area", 500),
                    "motion_threshold": cam.get("motion_threshold", 25),
                    "motion_record_after_time": cam.get("motion_record_after_time", 3),
                }
            )

        return cameras_list
    except Exception as e:
        capture_message("error", f"Ошибка при запросе камер из БД: {e}")
        return []


def queue_worker(q, stop_event):
    processed_count = 0

    while not stop_event.is_set():
        try:
            data = q.get(timeout=0.3)
            if data is None:
                break

            processed_count += 1
            timestamp_num, camera_id, camera_name, user_id, faces, face_widths, _ = data
            timestamp = datetime.fromtimestamp(timestamp_num)

            for idx, face in enumerate(faces):
                face_width = face_widths[idx]

                # 1. Строим короткий вектор sface (128)
                embedding_short = pFace.face_recognition(faces=[face], frmode='sface')[0]

                # 2. Проверяем в FixedSizeList - новое лицо или уже было
                face_distance_min, _, _, _ = face_short_cache.get_EVmin(embedding_short)

                if face_distance_min > EUCLIDIAN_DISTANCE_SEPARATING:
                    # Лицо новое - добавляем в кэш
                    face_short_cache.add(embedding_short.copy())

                    # Строим длинный вектор adaface (512)
                    embedding_full = pFace.face_recognition(faces=[face])[0]

                    # Сохраняем thumbnail
                    event_uuid = str(uuid.uuid4())
                    filename = f"{event_uuid}.jpeg"
                    full_path = os.path.join(THUMBNAIL_PATH, filename)
                    face_cv = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                    cv2.imwrite(full_path, face_cv)

                    dtime_str = datetime.fromtimestamp(timestamp_num).strftime("%Y%m%d-%H%M%S-%f")

                    # Обработка распознавания
                    item, debug_text = db.process_face_recognition(
                        user_id="1",
                        embedding=embedding_full,
                        dtime=dtime_str,
                        camera_id=camera_id,
                        max_distance=0.9,
                        is_real=False,
                        is_multiple=False,
                        percent_unknown=79.5,
                        image_data=full_path,
                    )

                    recognized = item["data"]["person"].get("facerecognized", False)
                    if recognized:
                        person_id = item["data"]["person"].get("id")
                        is_unknown = False
                        unknown_uuid = None
                    else:
                        person_id = None
                        is_unknown = True
                        unknown_uuid = item["data"]["person"].get("unknown_uuid")

                    # Логирование
                    percent = item["data"]["person"].get("percent")
                    if recognized:
                        capture_message("info", f"Распознано: person_id={person_id}, камера={camera_name}, процент={percent}")
                    else:
                        capture_message("info", f"Неизвестное лицо: uuid={unknown_uuid}, камера={camera_name}, процент={percent}")

                    # Запись в БД
                    event_data = {
                        "event_id": event_uuid,
                        "datetime": timestamp,
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "person_id": person_id,
                        "is_unknown": is_unknown,
                        "unknown_uuid": unknown_uuid,
                        "face_width": face_width,
                        "snapshot_path": full_path,
                        "user_id": user_id,
                    }

                    if not DEBUG_MODE:
                        if db.log_event(event_data):
                            status = "распознан" if recognized else "неизвестный"
                            capture_message("info", f"Ивент записан: event={event_uuid}, {status}, камера={camera_name}, person_id={person_id or 'N/A'}")
                        else:
                            capture_message("error", f"Ошибка записи события {event_uuid} в БД")
                else:
                    # Лицо уже было в последние 5 секунд - пропускаем
                    capture_message("debug", f"Лицо пропущено (уже было), камера={camera_name}")
                    continue

        except queue.Empty:
            continue
        except Exception as e:
            capture_message("error", f"Ошибка в queue_worker: {e}")


class CameraProcessor(threading.Thread):
    def __init__(self, cam_config, shared_queue):
        super().__init__()
        self.config = cam_config
        self.running = True
        self.camera_id = cam_config["cam_id"]
        self.cam_name = cam_config["name"]
        self.stream_url = cam_config["stream_to_parse"]

        if len(self.stream_url) <= 3:
            self.stream_url = int(self.stream_url)

        self.user_id = cam_config.get("user_id", "1")
        self.face_width_min = cam_config.get("face_width_min", MIN_WIDTH_PHOTO)
        self.timedelay = cam_config.get("timedelay", 333) / 1000.0
        self.shared_queue = shared_queue
        self.frames_processed = 0
        self.faces_detected = 0
        # Параметры детектора движения
        self.motion_min_area = cam_config.get("motion_min_area", 500)
        self.motion_threshold = cam_config.get("motion_threshold", 25)
        self.motion_record_after_time = cam_config.get("motion_record_after_time", 3)

        self.events_dir_path = f"events/{self.cam_name}"

        if not os.path.exists(self.events_dir_path):
            os.mkdir(self.events_dir_path)

    def run(self):
        try:
            cap = VideoCapture(self.stream_url, self.camera_id)
        except Exception as e:
            capture_message(
                "error", f"Ошибка создания VideoCapture для камеры {self.cam_name}: {e}"
            )
            return

        # Инициализация детектора движения
        motion_detector = MotionDetector(
            min_area=self.motion_min_area,
            threshold=self.motion_threshold,
            record_after_time=self.motion_record_after_time,
        )

        while self.running:

            frame = cap.read()

            if frame is None:
                time.sleep(0.1)
                continue

            # Обработка движения
            motion, recording_active, area, area_box = motion_detector.process_frame(
                frame
            )

            # Если запись не активна (нет движения и не в периоде дожития) – пропускаем кадр
            if not recording_active:
                time.sleep(0.05)
                continue

            # Далее – только при активной записи
            self.frames_processed += 1
            start_time = time.time()

            bboxes, faces, _ = pFace.face_detection(frame)

            if faces:
                event_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                filtered_faces = []
                face_widths = []
                for b, f in zip(bboxes, faces):
                    w = int(b[2] - b[0])
                    if w >= self.face_width_min:
                        filtered_faces.append(f)
                        face_widths.append(w)

                if filtered_faces:
                    self.faces_detected += len(filtered_faces)
                    self.shared_queue.put_nowait(
                        (
                            time.time(),
                            self.camera_id,
                            self.cam_name,
                            self.user_id,
                            filtered_faces,
                            face_widths,
                            self.config.get("stream_id", ""),
                        )
                    )
                    cv2.imwrite(
                        f"{self.events_dir_path}/{self.camera_id}_{event_timestamp}.jpg",
                        frame,
                    )

            elapsed = time.time() - start_time
            remaining = self.timedelay - elapsed
            if remaining > 0:
                time.sleep(remaining)

        cap.stop()


def camera_polling_thread(stop_event):
    while not stop_event.is_set():
        try:
            new_cams = fetch_cameras_from_db()
            with cameras_lock:
                current_ids = set(cameras.keys())
                new_ids = {cam["cam_id"] for cam in new_cams}

                for cam_id in current_ids - new_ids:
                    capture_message("info", f"Камера {cam_id} удалена")
                    cameras[cam_id].running = False
                    cameras[cam_id].join(timeout=5)
                    del cameras[cam_id]

                for cam in new_cams:
                    cam_id = cam["cam_id"]
                    if cam_id not in cameras:
                        capture_message("info", f"Камера {cam['name']} добавлена")
                        proc = CameraProcessor(cam, shared_queue)
                        proc.start()
                        cameras[cam_id] = proc
                    else:
                        old = cameras[cam_id].config
                        if (
                            old["stream_to_parse"] != cam["stream_to_parse"]
                            or old.get("face_width_min") != cam.get("face_width_min")
                            or old.get("timedelay") != cam.get("timedelay")
                            or old.get("motion_min_area") != cam.get("motion_min_area")
                            or old.get("motion_threshold")
                            != cam.get("motion_threshold")
                            or old.get("motion_record_after_time")
                            != cam.get("motion_record_after_time")
                        ):
                            capture_message("info", f"Камера {cam_id} изменена")
                            cameras[cam_id].running = False
                            cameras[cam_id].join(timeout=5)
                            proc = CameraProcessor(cam, shared_queue)
                            proc.start()
                            cameras[cam_id] = proc
        except Exception as e:
            capture_message("error", f"Ошибка в потоке опроса камер: {e}")

        stop_event.wait(timeout=CAMERA_POLL_INTERVAL)


def main():
    capture_message("info", "PACS Core starting...", force_sentry=True)
    if not os.path.exists("events"):
        os.mkdir("events")

    global shared_queue
    shared_queue = queue.Queue(maxsize=1000)
    stop_event = threading.Event()

    queue_thread = threading.Thread(
        target=queue_worker, args=(shared_queue, stop_event)
    )
    queue_thread.start()

    poll_thread = threading.Thread(target=camera_polling_thread, args=(stop_event,))
    poll_thread.start()

    capture_message("info", "PACS Core started successfully", force_sentry=True)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        capture_message("info", "PACS Core shutting down...", force_sentry=True)
        stop_event.set()

        with cameras_lock:
            for proc in cameras.values():
                proc.running = False
            for proc in cameras.values():
                proc.join(timeout=5)

        queue_thread.join()
        poll_thread.join()
        db.close_all_connections()
        capture_message("info", "PACS Core stopped", force_sentry=True)


if __name__ == "__main__":
    main()
