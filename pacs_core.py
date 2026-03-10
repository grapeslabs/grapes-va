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
import requests
import queue
from datetime import datetime
from PIL import Image
import numpy as np

from libs.pinfacekirjasto.PinFace import PinFace
from libs.detect_motion import MotionDetector
from libs.color_logger import ColorLogger
from libs.DbLibrary import FRDatabase
from libs.camstream import VideoCapture

pFace = PinFace(ffmode="mtcnn", frmode="adaface")

logger = ColorLogger("PACS_CORE", log_file="pacs_core.log", level=logging.INFO)

DEBUG_MODE = os.getenv("DEBUG_MODE", "false").lower() == "true"
MIN_WIDTH_PHOTO = int(os.getenv("MIN_WIDTH_PHOTO", "50"))
EUCLIDEAN_THRESHOLD = float(os.getenv("EUCLIDEAN_THRESHOLD", "0.6"))
MAX_FACES_IN_LIST = int(os.getenv("MAX_FACES_IN_LIST", "25"))
THUMBNAIL_PATH = os.getenv("THUMBNAIL_PATH", "input/thumbnails/")
PACS_API_URL = os.getenv("PACS_API_URL", "http://localhost:5000")
CAMERA_POLL_INTERVAL = int(os.getenv("CAMERA_POLL_INTERVAL", "5"))

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
FR_DB = os.getenv("FR_DB", "fr")

os.makedirs(THUMBNAIL_PATH, exist_ok=True)

db = FRDatabase()

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
                    "user_id": cam["user_id"],
                    "face_width_max": cam.get("face_width_max", MIN_WIDTH_PHOTO),
                    "timedelay": cam.get("timedelay", 333),
                    "motion_min_area": cam.get("motion_min_area", 500),
                    "motion_threshold": cam.get("motion_threshold", 25),
                    "motion_record_after_time": cam.get("motion_record_after_time", 3),
                }
            )
        logger.info(f"Получено {len(cameras_list)} камер из БД")
        return cameras_list
    except Exception as e:
        logger.error(f"Ошибка при запросе камер из БД: {e}")
        return []


def queue_worker(q, stop_event):
    logger.info("Запуск потока обработчика очереди")
    processed_count = 0
    person_last_seen = {}  # для известных персон (camera_id, person_id)
    unknown_last_seen = {}  # для неизвестных персон (camera_id, unknown_uuid)
    PERSON_TIMEOUT = 3  # секунд

    while not stop_event.is_set():
        try:
            data = q.get(timeout=0.3)
            if data is None:
                logger.info("Получен сигнал остановки обработчика очереди")
                break

            processed_count += 1
            timestamp_num, camera_id, camera_name, user_id, faces, face_widths, _ = data
            timestamp = datetime.fromtimestamp(timestamp_num)
            current_time = time.time()

            embeddings = pFace.face_recognition(faces=faces)

            for idx, emb in enumerate(embeddings):
                face_img = faces[idx]
                face_width = face_widths[idx]
                emb_128 = emb[:128] if len(emb) >= 128 else emb

                # Быстрая проверка - есть ли персона в базе
                person_id = db.find_similar_person(emb_128.tolist())

                # Проверяем таймаут для известной персоны
                if person_id:
                    key = (camera_id, person_id)
                    last_seen = person_last_seen.get(key, 0)
                    if current_time - last_seen < PERSON_TIMEOUT:
                        logger.debug(f"Таймаут для персоны {person_id}, пропуск")
                        continue

                # Только теперь выполняем полную обработку
                event_uuid = str(uuid.uuid4())
                filename = f"{event_uuid}.jpeg"
                full_path = os.path.join(THUMBNAIL_PATH, filename)

                face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
                cv2.imwrite(full_path, face_cv)

                dtime_str = datetime.fromtimestamp(timestamp_num).strftime(
                    "%Y%m%d-%H%M%S-%f"
                )

                item, debug_text = db.process_face_recognition(
                    embedding=emb_128.tolist(),
                    dtime=dtime_str,
                    camera_id=camera_id,
                    user_id=user_id,
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

                    # Обновляем время для распознанной персоны
                    if person_id:
                        person_last_seen[(camera_id, person_id)] = current_time

                else:
                    person_id = None
                    is_unknown = True
                    unknown_uuid = item["data"]["person"].get("unknown_uuid")

                    # Проверяем таймаут для неизвестной персоны
                    if unknown_uuid:
                        key = (camera_id, unknown_uuid)
                        last_seen = unknown_last_seen.get(key, 0)
                        if current_time - last_seen < PERSON_TIMEOUT:
                            logger.debug(
                                f"Таймаут для неизвестной {unknown_uuid}, пропуск"
                            )
                            continue
                        # Обновляем время для неизвестной персоны
                        unknown_last_seen[key] = current_time

                # Логирование и запись в БД
                print(f"  Результат: {debug_text}")
                print(f"  Распознано: {recognized}")
                print(f"  Person ID: {person_id}")
                print(f"  Unknown UUID: {unknown_uuid}")
                print(f"  Процент: {item['data']['person'].get('percent')}")

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
                    "vector128": emb_128.tolist(),
                    "user_id": user_id,
                }

                if not DEBUG_MODE:
                    if db.log_event(event_data):
                        logger.info(f"Событие {event_uuid} записано в БД")
                    else:
                        logger.error(f"Ошибка записи события {event_uuid} в БД")

        except queue.Empty:
            continue
        except Exception as e:
            logger.error(f"Ошибка в queue_worker: {e}")

    logger.info(
        f"Обработчик очереди завершил работу. Всего обработано сообщений: {processed_count}"
    )


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

        self.user_id = cam_config.get("user_id", "")
        self.face_width_max = cam_config.get("face_width_max", MIN_WIDTH_PHOTO)
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
        logger.info(f"Запуск потока камеры: {self.cam_name} ({self.camera_id})")
        logger.info(
            f"Motion params: min_area={self.motion_min_area}, threshold={self.motion_threshold}, record_after={self.motion_record_after_time}"
        )

        # <-- НОВОЕ: параметры для контроля потери сигнала
        max_consecutive_failures = 10  # сколько плохих кадров подряд считать потерей
        consecutive_failures = 0  # счётчик текущих плохих кадров
        max_reconnect_attempts = 5  # максимальное число попыток переподключения
        reconnect_attempts = 0  # счётчик попыток переподключения
        logger.info(f"Попытка создания VideoCapture для камеры {self.cam_name}")
        try:
            cap = VideoCapture(self.stream_url)
            logger.info(
                f"Обьект VideoCapture для камеры {self.cam_name} успешно создан"
            )

        except Exception as e:
            logger.error(f"Ошибка создания VideoCapture: {e}")
            return

        # Инициализация детектора движения
        motion_detector = MotionDetector(
            min_area=self.motion_min_area,
            threshold=self.motion_threshold,
            record_after_time=self.motion_record_after_time,
        )
        logger.info(f"Запуск потока обработки кадров для камеры {self.cam_name}")
        while self.running:
            frame = cap.read()
            if frame is None:
                consecutive_failures += 1

                # Если ещё не достигнут порог, просто ждём и пробуем следующий кадр
                if consecutive_failures < max_consecutive_failures:
                    time.sleep(0.05)
                    continue

                # Достигнут порог – начинаем процедуру переподключения
                reconnect_attempts += 1
                logger.warning(
                    f"Потеря сигнала с камеры {self.cam_name} в течение {consecutive_failures} кадров, "
                    f"попытка переподключения {reconnect_attempts}/{max_reconnect_attempts}"
                )

                if reconnect_attempts >= max_reconnect_attempts:
                    logger.error(
                        f"Не удалось восстановить соединение с камерой {self.cam_name} "
                        f"после {max_reconnect_attempts} попыток, завершение потока"
                    )
                    break

                cap.stop()
                time.sleep(2)

                # Пытаемся открыть камеру заново
                try:
                    cap = VideoCapture(self.stream_url)
                    if cap:
                        logger.info(f"Переподключение к камере {self.cam_name} успешно")
                        # Сбрасываем все счётчики
                        reconnect_attempts = 0
                        consecutive_failures = 0
                        continue
                    else:
                        logger.error(
                            f"Не удалось открыть камеру при переподключении {self.cam_name}"
                        )
                except Exception as e:
                    logger.error(
                        f"Ошибка при переподключении к камере {self.cam_name}: {e}"
                    )

                # Если не удалось открыть, делаем паузу и повторяем попытку переподключения
                time.sleep(1)
                continue

            # Успешное чтение кадра – сбрасываем счётчики ошибок
            consecutive_failures = 0
            reconnect_attempts = 0

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

            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            pil_img = Image.fromarray(frame_rgb)
            bboxes, faces, _ = pFace.face_detection(pil_img)

            if faces:
                event_timestamp = datetime.now().strftime("%d-%m-%Y_%H-%M-%S")
                filtered_faces = []
                face_widths = []
                for b, f in zip(bboxes, faces):
                    w = int(b[2] - b[0])
                    if w >= self.face_width_max:
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
        logger.info(
            f"Поток камеры {self.cam_name} остановлен. Обработано кадров: {self.frames_processed}"
        )


def camera_polling_thread(stop_event):
    logger.info("Запуск потока опроса камер")
    while not stop_event.is_set():
        try:
            new_cams = fetch_cameras_from_db()
            with cameras_lock:
                current_ids = set(cameras.keys())
                new_ids = {cam["cam_id"] for cam in new_cams}

                for cam_id in current_ids - new_ids:
                    logger.info(f"Камера {cam_id} удалена из API")
                    cameras[cam_id].running = False
                    cameras[cam_id].join(timeout=5)
                    del cameras[cam_id]

                for cam in new_cams:
                    cam_id = cam["cam_id"]
                    if cam_id not in cameras:
                        logger.info(f"Обнаружена новая камера: {cam['name']}")
                        proc = CameraProcessor(cam, shared_queue)
                        proc.start()
                        cameras[cam_id] = proc
                    else:
                        old = cameras[cam_id].config
                        if (
                            old["stream_to_parse"] != cam["stream_to_parse"]
                            or old.get("face_width_max") != cam.get("face_width_max")
                            or old.get("timedelay") != cam.get("timedelay")
                            or old.get("motion_min_area") != cam.get("motion_min_area")
                            or old.get("motion_threshold")
                            != cam.get("motion_threshold")
                            or old.get("motion_record_after_time")
                            != cam.get("motion_record_after_time")
                        ):
                            logger.info(f"Камера {cam_id} изменена")
                            cameras[cam_id].running = False
                            cameras[cam_id].join(timeout=5)
                            proc = CameraProcessor(cam, shared_queue)
                            proc.start()
                            cameras[cam_id] = proc
        except Exception as e:
            logger.error(f"Ошибка в потоке опроса камер: {e}")

        stop_event.wait(timeout=CAMERA_POLL_INTERVAL)


def main():
    logger.info("=" * 50)
    logger.info("Инициализация системы PACS Core...")
    logger.info("=" * 50)

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

    logger.info("Система готова к работе")
    logger.info("=" * 50)

    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        logger.info("Получен сигнал остановки, завершение работы...")
        stop_event.set()

        with cameras_lock:
            for proc in cameras.values():
                proc.running = False
            for proc in cameras.values():
                proc.join(timeout=5)

        queue_thread.join()
        poll_thread.join()
        db.close_all_connections()
        logger.info("Система остановлена")


if __name__ == "__main__":
    main()
