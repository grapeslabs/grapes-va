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


def fetch_cameras_from_api():
    try:
        resp = requests.get(f"{PACS_API_URL}/api/c1/list", timeout=5)
        if resp.status_code == 200:
            data = resp.json()
            tasks = data.get("tasks", {}).get("queue", {})
            cameras_list = []
            for cam_id, cam_info in tasks.items():
                cameras_list.append(
                    {
                        "cam_id": cam_id,
                        "name": cam_info.get("name", cam_info.get("desc", cam_id)),
                        "stream_to_parse": cam_info.get("stream_to_parse"),
                        "user_id": cam_info.get("user_id"),
                        "face_width_max": cam_info.get(
                            "face_width_max", MIN_WIDTH_PHOTO
                        ),
                        "timedelay": cam_info.get("timedelay", 333),
                        # Параметры детектора движения
                        "motion_min_area": cam_info.get("motion_min_area", 500),
                        "motion_threshold": cam_info.get("motion_threshold", 25),
                        "motion_record_after_time": cam_info.get(
                            "motion_record_after_time", 3
                        ),
                    }
                )
            logger.info(f"Получено {len(cameras_list)} камер из API")
            return cameras_list
        else:
            logger.error(f"API вернул код {resp.status_code}")
    except Exception as e:
        logger.error(f"Ошибка при запросе камер: {e}")
    return []


def queue_worker(q, stop_event):
    logger.info("Запуск потока обработчика очереди")
    processed_count = 0
    while not stop_event.is_set():
        try:
            data = q.get(timeout=0.3)
            if data is None:
                logger.info("Получен сигнал остановки обработчика очереди")
                break

            processed_count += 1
            timestamp_num, camera_id, camera_name, user_id, faces, face_widths, _ = data
            timestamp = datetime.fromtimestamp(timestamp_num)
            print(f"user_id = {user_id} (type: {type(user_id)})")
            embeddings = pFace.face_recognition(faces=faces)

            for idx, emb in enumerate(embeddings):
                face_img = faces[idx]
                face_width = face_widths[idx]
                event_uuid = str(uuid.uuid4())
                filename = f"{event_uuid}.jpeg"
                full_path = os.path.join(THUMBNAIL_PATH, filename)

                face_cv = cv2.cvtColor(np.array(face_img), cv2.COLOR_RGB2BGR)
                cv2.imwrite(full_path, face_cv)

                dtime_str = datetime.fromtimestamp(timestamp_num).strftime(
                    "%Y%m%d-%H%M%S-%f"
                )

                # Берем первые 128 элементов для вектора
                emb_128 = emb[:128] if len(emb) >= 128 else emb

                # После получения эмбеддинга
                emb_128 = emb[:128] if len(emb) >= 128 else emb
                print(f"\n🔍 РАСПОЗНАВАНИЕ ЛИЦА:")
                print(f"  Вектор (первые 5 значений): {emb_128[:5]}")

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

                print(f"  Результат: {debug_text}")
                print(f"  Распознано: {item['data']['person'].get('facerecognized')}")
                print(f"  Person ID: {item['data']['person'].get('id')}")
                print(f"  Процент: {item['data']['person'].get('percent')}")

                recognized = item["data"]["person"].get("facerecognized", False)
                person_id = item["data"]["person"].get("id") if recognized else None
                is_unknown = not recognized

                event_data = {
                    "event_id": event_uuid,
                    "datetime": timestamp,
                    "camera_id": camera_id,
                    "camera_name": camera_name,
                    "person_id": person_id,
                    "is_unknown": is_unknown,
                    "face_width": face_width,
                    "snapshot_path": full_path,
                    "vector128": emb_128.tolist(),
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

    def run(self):
        logger.info(f"Запуск потока камеры: {self.cam_name} ({self.camera_id})")
        logger.info(
            f"Motion params: min_area={self.motion_min_area}, threshold={self.motion_threshold}, record_after={self.motion_record_after_time}"
        )

        if isinstance(self.stream_url, str) and self.stream_url.isdigit():
            self.stream_url = int(self.stream_url)

        # <-- НОВОЕ: параметры для контроля потери сигнала
        max_consecutive_failures = 10  # сколько плохих кадров подряд считать потерей
        consecutive_failures = 0  # счётчик текущих плохих кадров
        max_reconnect_attempts = 5  # максимальное число попыток переподключения
        reconnect_attempts = 0  # счётчик попыток переподключения

        try:
            cap = cv2.VideoCapture(self.stream_url)
            if not cap.isOpened():
                logger.error(f"Не удалось открыть камеру {self.cam_name}")
                return
        except Exception as e:
            logger.error(f"Ошибка создания VideoCapture: {e}")
            return

        # Инициализация детектора движения
        motion_detector = MotionDetector(
            min_area=self.motion_min_area,
            threshold=self.motion_threshold,
            record_after_time=self.motion_record_after_time,
        )

        while self.running:
            ret, frame = cap.read()
            if not ret or frame is None:
                consecutive_failures += 1
                logger.debug(
                    f"Не удалось прочитать кадр с камеры {self.cam_name}, "
                    f"последовательных ошибок: {consecutive_failures}"
                )

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

                cap.release()
                time.sleep(2)

                # Пытаемся открыть камеру заново
                try:
                    cap = cv2.VideoCapture(self.stream_url)
                    if cap.isOpened():
                        logger.info(f"Переподключение к камере {self.cam_name} успешно")
                        # Сбрасываем все счётчики
                        reconnect_attempts = 0
                        consecutive_failures = 0
                        continue  # переходим к следующей итерации для чтения кадра
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

            elapsed = time.time() - start_time
            remaining = self.timedelay - elapsed
            if remaining > 0:
                time.sleep(remaining)

        cap.release()
        logger.info(
            f"Поток камеры {self.cam_name} остановлен. Обработано кадров: {self.frames_processed}"
        )


def camera_polling_thread(stop_event):
    logger.info("Запуск потока опроса камер")
    while not stop_event.is_set():
        try:
            new_cams = fetch_cameras_from_api()
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
