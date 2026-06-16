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
import json
import base64
from datetime import datetime
import numpy as np
from dotenv import load_dotenv

from libs.pinfacekirjasto.PinFace import PinFace
from libs.detect_motion import MotionDetector
from libs.PersonDetector import PersonDetector

from libs.loglib import capture_message, shutdown
from libs.DbLibrary import FRDatabase
from libs.camstream import VideoCapture
from libs.FixedSizeList import FixedSizeList
import atexit

load_dotenv()


try:
    pFace = PinFace(ffmode="mtcnn", frmode="adaface", fmode=["sface"])
    capture_message("debug", "PinFace initialized successfully")
except Exception as e:
    capture_message("error", f"Failed to initialize PinFace: {e}")
    raise SystemExit(1)

atexit.register(shutdown)

# FixedSizeList для коротких векторов sface (128)
face_short_cache = FixedSizeList(max_size=50, time_len=5)

try:
    db = FRDatabase()
    capture_message("info", "Database connection established")
except Exception as e:
    capture_message("error", f"Failed to connect to database: {e}")
    raise SystemExit(1)


EUCLIDIAN_DISTANCE_SEPARATING = 0.9
DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
MODE = os.getenv("MODE", "pacs")
MIN_WIDTH_PHOTO = int(os.getenv("MIN_WIDTH_PHOTO", "50"))
EUCLIDEAN_THRESHOLD = float(os.getenv("EUCLIDEAN_THRESHOLD", "0.6"))
MAX_FACES_IN_LIST = int(os.getenv("MAX_FACES_IN_LIST", "25"))
THUMBNAIL_PATH = os.getenv("THUMBNAIL_PATH", "input/thumbnails/")
PACS_API_URL = os.getenv("PACS_API_URL", "http://localhost:5000")
CAMERA_POLL_INTERVAL = int(os.getenv("CAMERA_POLL_INTERVAL", "5"))

DEFAULT_FACE_WIDTH_MAX = int(os.getenv("FACE_WIDTH_MAX", "45"))
DEFAULT_RESIZE = os.getenv("RESIZE")
DEFAULT_IS_DETECTION = os.getenv("IS_DETECTION", "true").lower() == "true"
DEFAULT_IS_RECOGNIZE = os.getenv("IS_RECOGNIZE", "true").lower() == "true"
DEFAULT_CACHE_FACE_TIME = int(os.getenv("CACHE_FACE_TIME", "30"))
DEFAULT_CACHE_FACE_MAX = int(os.getenv("CACHE_FACE_MAX", "20"))
DEFAULT_DETECTION_FIGURE_ACTIVE = (
    os.getenv("DETECTION_FIGURE_ACTIVE", "false").lower() == "true"
)
DEFAULT_DETECTION_FIGURE_DIRECTION = os.getenv("DETECTION_FIGURE_DIRECTION", "LRBTA")
DEFAULT_WRITE_THUMBNAILS = os.getenv("WRITE_THUMBNAILS", "false").lower() == "true"
DEFAULT_WRITE_FRAME = os.getenv("WRITE_FRAME", "false").lower() == "true"
TIMEDELAY = int(os.getenv("TIMEDELAY", 333))

MOTION_MIN_AREA = int(os.getenv("MOTION_MIN_AREA", "500"))
MOTION_THRESHOLD = int(os.getenv("MOTION_THRESHOLD", "25"))
MOTION_RECORD_AFTER_TIME = int(os.getenv("MOTION_RECORD_AFTER_TIME", "3"))
RECONNECT_DELAY = 600  # 10 минут

os.makedirs(THUMBNAIL_PATH, exist_ok=True)

migrations_file = os.path.join(os.path.dirname(__file__), "libs", "update_schema.py")

try:
    # Проверяем, что выбранные режимы допустимы
    assert MODE in ["pacs", "pin"]
except Exception as e:
    # Если возникла ошибка, выводим сообщение и завершаем программу
    s = str(e).strip()  # Преобразуем исключение в строку и убираем лишние пробелы
    capture_message(
        "error",
        f"Failed set MODE {e}; Имя ошибки: {e.__class__.__name__}"
        + (f"; Сообщение об ошибке: {s}" if s else ""),
    )
    raise SystemExit(1)


if os.path.isfile(migrations_file):
    try:
        from libs.update_schema import apply_migrations

        apply_migrations(db, MODE, capture_message)
    except ImportError as e:
        capture_message("error", f"Не удалось импортировать apply_migrations: {e}")
    except Exception as e:
        capture_message("error", f"Ошибка при применении миграций: {e}")
else:
    capture_message("debug", "Файл миграций отсутствует (возможно, уже применён)")
    pass

cameras = {}
cameras_lock = threading.Lock()
shared_queue = None

capture_message("info", f"PACS mode = {MODE}", force_sentry=True)
if MODE == "pacs":
    MODE_SUB = "pacs"
elif MODE == "pin":
    MODE_SUB = "detection"
    import sys

    if len(sys.argv) > 1:
        if sys.argv[1].lower() == "--recognize":
            MODE_SUB = "recognize"
    capture_message("info", f"PACS mode_sub = {MODE_SUB}", force_sentry=True)


# 270526 -->
def figure_queue_worker(q, stop_event):
    while not stop_event.is_set():
        try:
            data = q.get(timeout=0.3)
            if data is None:
                break

            db.log_figure_event(data)
            capture_message(
                "info",
                f"Figure event saved: {data['event_id']}, cam={data['camera_name']}, persons={data['person_count']}",
            )
        except queue.Empty:
            continue
        except Exception as e:
            capture_message("error", f"Figure queue worker error: {e}")


# 270526 <---


def fetch_cameras_from_db():
    """Читает камеры напрямую из БД"""
    try:
        cameras_list = []

        if MODE == "pacs":
            rows = db.get_all_cameras()  # используем новый метод
        elif MODE == "pin":
            # заглушка для pin
            pass

        for cam in rows:
            cameras_list.append(
                {
                    "cam_id": cam["cam_id"],
                    "name": cam["name"],
                    "stream_to_parse": cam["stream_to_parse"],
                    "user_id": cam.get("user_id", "1"),
                    "user_mail": cam.get("user_mail"),
                    "face_width_min": cam.get("face_width_min", MIN_WIDTH_PHOTO),
                    "face_width_max": cam.get("face_width_max", DEFAULT_FACE_WIDTH_MAX),
                    "timedelay": cam.get("timedelay", TIMEDELAY),
                    "resize": cam.get("resize", DEFAULT_RESIZE),
                    "crop_params": cam.get("crop_params"),
                    "motion_min_area": cam.get("motion_min_area", MOTION_MIN_AREA),
                    "motion_threshold": cam.get("motion_threshold", MOTION_THRESHOLD),
                    "motion_record_after_time": cam.get(
                        "motion_record_after_time", MOTION_RECORD_AFTER_TIME
                    ),
                    "is_detection": cam.get("is_detection", DEFAULT_IS_DETECTION),
                    "is_recognize": cam.get("is_recognize", DEFAULT_IS_RECOGNIZE),
                    "cache_face_time": cam.get(
                        "cache_face_time", DEFAULT_CACHE_FACE_TIME
                    ),
                    "cache_face_max": cam.get("cache_face_max", DEFAULT_CACHE_FACE_MAX),
                    "detection_figure_active": cam.get(
                        "detection_figure_active", DEFAULT_DETECTION_FIGURE_ACTIVE
                    ),
                    "detection_figure_direction": cam.get(
                        "detection_figure_direction", DEFAULT_DETECTION_FIGURE_DIRECTION
                    ),
                    "detection_figure_zones": cam.get("detection_figure_zones"),
                    "write_thumbnails": cam.get(
                        "write_thumbnails", DEFAULT_WRITE_THUMBNAILS
                    ),
                    "write_frame": cam.get("write_frame", DEFAULT_WRITE_FRAME),
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

            # Получаем параметры камеры
            cam_proc = cameras.get(camera_id)
            cam_params = cam_proc.config if cam_proc else {}
            is_detection = cam_params.get("is_detection", DEFAULT_IS_DETECTION)
            is_recognize = cam_params.get("is_recognize", DEFAULT_IS_RECOGNIZE)
            detection_figure_active = cam_params.get(
                "detection_figure_active", DEFAULT_DETECTION_FIGURE_ACTIVE
            )

            full_path = None
            event_uuid = str(uuid.uuid4())
            timestamp_str = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
            filename = f"{event_uuid}.jpeg"

            
            if not is_detection and not detection_figure_active:
                continue
            
            for idx, face in enumerate(faces):
                face_width = face_widths[idx]

                # Если только детекция без распознавания
                if is_detection and not is_recognize:
                    face_event_uuid = str(uuid.uuid4())
                    capture_message(
                        "debug",
                        f"Детекция лица без распознавания: камера={camera_name}",
                    )

                    face_cv = cv2.cvtColor(np.array(face), cv2.COLOR_RGB2BGR)
                    face_filename = f"{face_event_uuid}.jpeg"
                    full_path = os.path.join(THUMBNAIL_PATH, face_filename)
                    cv2.imwrite(full_path, face_cv)

                    event_data = {
                        "event_id": face_event_uuid,
                        "datetime": timestamp,
                        "camera_id": camera_id,
                        "camera_name": camera_name,
                        "person_id": None,
                        "is_unknown": True,
                        "unknown_uuid": None,
                        "face_width": face_width,
                        "snapshot_path": full_path,
                        "user_id": user_id,
                        "is_recognize": is_recognize,
                    }

                    try:
                        db.log_event(event_data=event_data)
                        capture_message(
                            "info",
                            f"Ивент записан: event={event_uuid}, камера={camera_name}",
                        )
                    except Exception as e:
                        capture_message("error", f"Событие не записано. Ошибка: {e}")

                    continue

                # 1. Строим короткий вектор sface (128)
                embedding_short = pFace.face_recognition(faces=[face], frmode="sface")[
                    0
                ]

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

                    dtime_str = datetime.fromtimestamp(timestamp_num).strftime(
                        "%Y%m%d-%H%M%S-%f"
                    )

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
                        capture_message(
                            "info",
                            f"Распознано: person_id={person_id}, камера={camera_name}, процент={percent}",
                        )
                    else:
                        capture_message(
                            "info",
                            f"Неизвестное лицо: uuid={unknown_uuid}, камера={camera_name}, процент={percent}",
                        )

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
                        "is_recognize": is_recognize,
                    }

                    if db.log_event(event_data):
                        status = "распознан" if recognized else "неизвестный"
                        capture_message(
                            "info",
                            f"Ивент записан: event={event_uuid}, {status}, камера={camera_name}, person_id={person_id or 'N/A'}",
                        )
                    else:
                        capture_message(
                            "error", f"Ошибка записи события {event_uuid} в БД"
                        )
                else:
                    # Лицо уже было в последние 5 секунд - пропускаем
                    capture_message(
                        "debug", f"Лицо пропущено (уже было), камера={camera_name}"
                    )
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
        self.timedelay = cam_config.get("timedelay", TIMEDELAY) / 1000.0
        self.shared_queue = shared_queue
        self.frames_processed = 0
        self.faces_detected = 0

        # Параметры детектора движения
        self.motion_min_area = cam_config.get("motion_min_area", MOTION_MIN_AREA)
        self.motion_threshold = cam_config.get("motion_threshold", MOTION_THRESHOLD)
        self.motion_record_after_time = cam_config.get(
            "motion_record_after_time", MOTION_RECORD_AFTER_TIME
        )

        # Дополнительные параметры камеры
        self.user_mail = cam_config.get("user_mail")
        self.face_width_max = cam_config.get("face_width_max", DEFAULT_FACE_WIDTH_MAX)
        self.resize = cam_config.get("resize", DEFAULT_RESIZE)
        self.crop_params = cam_config.get("crop_params")
        self.is_detection = cam_config.get("is_detection", DEFAULT_IS_DETECTION)
        self.is_recognize = cam_config.get("is_recognize", DEFAULT_IS_RECOGNIZE)
        self.cache_face_time = cam_config.get(
            "cache_face_time", DEFAULT_CACHE_FACE_TIME
        )
        self.cache_face_max = cam_config.get("cache_face_max", DEFAULT_CACHE_FACE_MAX)
        self.detection_figure_active = cam_config.get(
            "detection_figure_active", DEFAULT_DETECTION_FIGURE_ACTIVE
        )
        self.detection_figure_direction = cam_config.get(
            "detection_figure_direction", DEFAULT_DETECTION_FIGURE_DIRECTION
        )
        self.detection_figure_zones = cam_config.get("detection_figure_zones")

        # 270526 -->
        if not self.detection_figure_zones:
            self.detection_figure_zones = cam_config.get("crop_params")
        # <---

        self.figure_detector = None

        self.write_thumbnails = cam_config.get(
            "write_thumbnails", DEFAULT_WRITE_THUMBNAILS
        )
        self.write_frame = cam_config.get("write_frame", DEFAULT_WRITE_FRAME)

        self.events_dir_path = f"events/{self.cam_name}"
        os.makedirs(self.events_dir_path, exist_ok=True)

    def run(self):
        cap = None
        while self.running:
            try:
                cap = VideoCapture(
                    self.stream_url,
                    self.camera_id,
                    resize=self.resize,
                    crop_params=self.detection_figure_zones,
                )
                break
            except Exception as e:
                capture_message(
                    "error", f"Камера {self.cam_name}: не удалось подключиться: {e}"
                )
                for _ in range(RECONNECT_DELAY):
                    if not self.running:
                        return
                    time.sleep(1)

        if cap is None:
            return

        if not self.is_detection and not self.detection_figure_active:
            capture_message("info", f"Камера {self.cam_name}: is_detection и detection_figure_active отключены, поток останавливается")
            self.running = False
            cap.stop()
            return

        
        # 250526 -->
        if self.detection_figure_active:
            try:
                self.figure_detector = PersonDetector(model_size="m", model_path="pretrained/detection_figure_model.pt")
                capture_message("debug", "Распознавание фигур инициализировано успешно")
            except Exception as e:
                capture_message(
                    "error",
                    f"Ошибка при инициализации FigureDetector, cam_id={self.camera_id}",
                )

        # 250526 <--

        motion_detector = None
        # Инициализация детектора движения
        try:
            motion_detector = MotionDetector(
                min_area=self.motion_min_area,
                threshold=self.motion_threshold,
                record_after_time=self.motion_record_after_time,
            )
        except Exception as e:
            capture_message(
                "error",
                f"Ошибка при инициализации MotionDetector, cam_id={self.camera_id}",
            )
            cap.stop()
            return

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

            # 250526 -->
            if self.detection_figure_active and self.figure_detector:
                boxes_person, annotated_frame = self.figure_detector.detect_in_frame(
                    frame, thickness=2, putTextinfo=True
                )
                person_count = len(boxes_person)
                if person_count > 0:
                    event_uuid = str(uuid.uuid4())
                    timestamp = datetime.now()
                    snapshot_path = ""

                    detection_data = {
                        "boxes": boxes_person,
                        "direction": self.detection_figure_direction,
                        "zones": self.detection_figure_zones,
                    }

                    if self.write_frame:
                        # сохранение кадра на диск
                        filename = f"{event_uuid}.jpg"
                        snapshot_path = os.path.join(self.events_dir_path, filename)
                        cv2.imwrite(snapshot_path, annotated_frame)

                    # кодирование в base64 и добавление в JSON
                    _, buffer = cv2.imencode(".jpg", annotated_frame)
                    detection_data["frame_base64"] = base64.b64encode(buffer).decode(
                        "utf-8"
                    )

                    figure_data = {
                        "event_id": event_uuid,
                        "datetime": timestamp,
                        "camera_id": self.camera_id,
                        "camera_name": self.cam_name,
                        "user_id": self.user_id,
                        "person_count": person_count,
                        "detection_data": detection_data,
                        "snapshot_path": snapshot_path,
                    }
                    figure_queue.put_nowait(figure_data)
            # <--

            if self.is_detection or self.is_recognize:
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
                        if MODE == "pacs":
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
                        elif MODE == "pin":
                            # запись в очередь rabbitMQ
                            pass

                        if DEFAULT_WRITE_FRAME:
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
            if MODE == "pacs":
                new_cams = fetch_cameras_from_db()
            elif MODE == "pin":
                # new_cams = fetch_cameras_from_db_file()
                pass
            with cameras_lock:
                current_ids = set(cameras.keys())
                new_ids = {cam["cam_id"] for cam in new_cams}

                for cam_id in current_ids - new_ids:
                    capture_message("info", f"Камера {cam_id} удалена")
                    cameras[cam_id].running = False
                    cameras[cam_id].join(timeout=5)
                    del cameras[cam_id]

                    if MODE == "pin":  # перенос файла в область удаленных
                        # fetch_cameras_from_db_delete_file()
                        pass

                for cam in new_cams:
                    cam_id = cam["cam_id"]
                    if cam_id not in cameras:

                        if not cam["is_detection"] and not cam.get("detection_figure_active", False):
                            capture_message(
                                "info",
                                f"Камера {cam['name']} не добавлена. Параметры is_detection и detection_figure_active отключены",
                            )
                            continue

                        capture_message("info", f"Камера {cam['name']} добавлена. is_detection = {cam['is_detection']}, is_recognize = {cam['is_recognize']}, detection_figure_active = {cam['detection_figure_active']}")
                        proc = CameraProcessor(cam, shared_queue)
                        proc.start()
                        cameras[cam_id] = proc

                    else:
                        if not cameras[cam_id].is_alive():
                            capture_message("info", f"Камера {cam_id}: поток завершился, очистка")
                            del cameras[cam_id]
                            continue
                        
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
                            or old.get("is_detection") != cam.get("is_detection")
                            or old.get("is_recognize") != cam.get("is_recognize")
                            or old.get("detection_figure_active")
                            != cam.get("detection_figure_active")
                            or old.get("detection_figure_zones")
                            != cam.get("detection_figure_zones")
                            or old.get("resize") != cam.get("resize")
                            or old.get("crop_params") != cam.get("crop_params")
                            or old.get("write_thumbnails")
                            != cam.get("write_thumbnails")
                            or old.get("write_frame") != cam.get("write_frame")
                        ):
                            if not cam["is_detection"] and not cam.get("detection_figure_active", False):
                                capture_message(
                                    "info", f"Камера {cam_id} остановлена: is_detection и detection_figure_active отключены"
                                )
                                cameras[cam_id].running = False
                                cameras[cam_id].join(timeout=5)
                                del cameras[cam_id]
                            else:
                                capture_message(
                                    "info", f"Камера {cam_id} изменена — перезапуск"
                                )
                                cameras[cam_id].running = False
                                cameras[cam_id].join(timeout=5)
                                proc = CameraProcessor(cam, shared_queue)
                                proc.start()
                                cameras[cam_id] = proc
        except Exception as e:
            capture_message("error", f"Ошибка в потоке опроса камер: {e}")

        stop_event.wait(timeout=CAMERA_POLL_INTERVAL)


def mode_pacs():

    global shared_queue
    shared_queue = queue.Queue(maxsize=1000)
    stop_event = threading.Event()

    # 270526 -->
    global figure_queue
    figure_queue = queue.Queue(maxsize=500)
    figure_thread = threading.Thread(
        target=figure_queue_worker, args=(figure_queue, stop_event)
    )
    figure_thread.start()
    # 270526 <--

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
        figure_thread.join()
        poll_thread.join()
        db.close_all_connections()


def mode_pin():
    ### для пина
    pass


if __name__ == "__main__":
    capture_message("info", "PACS Core starting...", force_sentry=True)

    if not os.path.exists("events") and DEFAULT_WRITE_FRAME:
        os.mkdir("events")

    if MODE == "pacs":
        mode_pacs()

    elif MODE == "pin":
        mode_pin()

    capture_message("info", "PACS Core stopped", force_sentry=True)
