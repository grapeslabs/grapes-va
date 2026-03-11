"""
DbLibrary.py
Модуль для работы с базой данных FR (персоны, фото, события, неизвестные лица).
Использует pgvector и хранимые функции find_similar_faces, find_or_create_by_vector.
"""

import os
import uuid
import base64
import numpy as np
import time
import threading
from datetime import datetime, timezone
from typing import List, Dict, Optional, Tuple, Iterator
from contextlib import contextmanager
import json
from functools import lru_cache

import psycopg2
from psycopg2 import pool, OperationalError, InterfaceError
from psycopg2.extras import DictCursor, RealDictCursor

PG_HOST = os.getenv("PG_HOST", "localhost")
PG_PORT = os.getenv("PG_PORT", "5432")
PG_USER = os.getenv("PG_USER", "postgres")
PG_PASSWORD = os.getenv("PG_PASSWORD", "postgres")
FR_DB = os.getenv("FR_DB", "fr")
PERSON_PHOTOS_PATH = os.getenv("PERSON_PHOTOS_PATH", "/data/person_photos")

EUCLIDEAN_THRESHOLD = float(os.getenv("EUCLIDEAN_THRESHOLD", "0.6"))
UNKNOWN_THRESHOLD = float(os.getenv("UNKNOWN_THRESHOLD", "0.8"))


class FRDatabase:
    EUCLIDEAN_THRESHOLD_PERCENT = 1.8
    INTERPOLATION_POINTS = 18
    _DISTANCE_VALUES = np.linspace(
        0, EUCLIDEAN_THRESHOLD_PERCENT, num=INTERPOLATION_POINTS, endpoint=True
    )
    _SIMILARITY_VALUES = np.array(
        [
            1.0,
            0.99,
            0.95,
            0.92,
            0.90,
            0.86,
            0.84,
            0.83,
            0.82,
            0.81,
            0.80,
            0.79,
            0.78,
            0.70,
            0.6,
            0.3,
            0.1,
            0,
        ]
    )
    _PERCENT_VALUES = _SIMILARITY_VALUES * 100
    _REVERSE_DISTANCE_VALUES = _DISTANCE_VALUES

    _connection_pool = None
    _pool_lock = threading.Lock()
    _semaphore = threading.Semaphore(10)

    def __init__(self):
        self.db_config = {
            "user": PG_USER,
            "password": PG_PASSWORD,
            "database": FR_DB,
            "host": PG_HOST,
            "port": PG_PORT,
            "application_name": "FRDatabase",
        }
        self._initialize_pool()

    def _initialize_pool(self):
        if FRDatabase._connection_pool is not None:
            return
        with FRDatabase._pool_lock:
            if FRDatabase._connection_pool is None:
                try:
                    FRDatabase._connection_pool = pool.ThreadedConnectionPool(
                        minconn=5, maxconn=20, **self.db_config
                    )
                    print("Пул соединений PostgreSQL (FR) инициализирован")
                except Exception as e:
                    raise RuntimeError(f"Ошибка создания пула: {e}")

    @contextmanager
    def _get_connection(self) -> Iterator[psycopg2.extensions.connection]:
        """Контекстный менеджер для безопасного получения соединения из пула"""
        conn = None
        with FRDatabase._semaphore:
            for attempt in range(3):
                try:
                    conn = FRDatabase._connection_pool.getconn()
                    # Проверка работоспособности
                    with conn.cursor() as cur:
                        cur.execute("SELECT 1")
                        if cur.fetchone()[0] != 1:
                            raise OperationalError("Проверочный запрос не прошёл")

                    # Возвращаем соединение через yield
                    yield conn

                    # После использования возвращаем в пул
                    if conn and not conn.closed:
                        FRDatabase._connection_pool.putconn(conn)
                    return

                except (OperationalError, InterfaceError) as e:
                    if conn:
                        FRDatabase._connection_pool.putconn(conn, close=True)
                        conn = None
                    if attempt == 2:
                        raise OperationalError(
                            f"Не удалось получить рабочее соединение после 3 попыток: {e}"
                        )
                    time.sleep(0.5 * (attempt + 1))
                except Exception as e:
                    if conn:
                        FRDatabase._connection_pool.putconn(conn)
                        conn = None
                    raise e
            raise OperationalError("Не удалось получить рабочее соединение")

    @contextmanager
    def _get_cursor(self, cursor_factory=None):
        """Контекстный менеджер для работы с курсором"""
        with self._get_connection() as conn:
            cursor = (
                conn.cursor(cursor_factory=cursor_factory)
                if cursor_factory
                else conn.cursor()
            )
            try:
                yield cursor
                conn.commit()
            except Exception:
                conn.rollback()
                raise
            finally:
                cursor.close()

    @staticmethod
    def _embedding_to_str(embedding: List[float]) -> str:
        return "[" + ",".join(f"{x:.6f}" for x in embedding) + "]"

    @staticmethod
    @lru_cache(maxsize=128)
    def _percentage_fcb(euclidean_distance: float) -> float:
        if 0 <= euclidean_distance <= FRDatabase.EUCLIDEAN_THRESHOLD_PERCENT:
            return (
                np.interp(
                    euclidean_distance,
                    FRDatabase._DISTANCE_VALUES,
                    FRDatabase._SIMILARITY_VALUES,
                )
                * 100
            )
        return 0.0

    @staticmethod
    def _format_datetime(dtime: str) -> str:
        try:
            date_part, time_part, _ = dtime.split("-")
            return f"{date_part[:4]}-{date_part[4:6]}-{date_part[6:8]}T{time_part[:2]}:{time_part[2:4]}:{time_part[4:6]}+03:00"
        except Exception:
            return datetime.now().isoformat()

    def get_all_persons_vectors(self) -> List[Tuple[str, List[float]]]:
        with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT person_id, vector
                FROM photo
                WHERE vector IS NOT NULL
            """
            )
            rows = cursor.fetchall()
            return [(row["person_id"], list(row["vector"])) for row in rows]

    def find_similar_person(
        self, embedding: List[float], threshold: float = EUCLIDEAN_THRESHOLD
    ) -> Optional[str]:
        emb_str = self._embedding_to_str(embedding)
        with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                """
                SELECT person_id, vector <-> %s::vector AS dist
                FROM photo
                WHERE vector IS NOT NULL
                ORDER BY vector <-> %s::vector
                LIMIT 1
            """,
                (emb_str, emb_str),
            )
            row = cursor.fetchone()
            if row and row["dist"] < threshold:
                return row["person_id"]
            return None

    def add_person(
        self, user_id: str = "1", person_id: str = None, description: str = ""
    ) -> str:
        if person_id is None:
            person_id = str(uuid.uuid4())
        percone_dttm = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_cursor() as cursor:
            cursor.execute(
                """
                INSERT INTO percone (user_id, person_id, description, percone_dttm, view_percone)
                VALUES (%s, %s, %s, %s, %s)
            """,
                (user_id, person_id, description, percone_dttm, True),
            )
        return person_id

    def add_photos_to_person(self, person_id: str, photos_data: List[Dict]) -> bool:
        with self._get_cursor() as cursor:
            cursor.execute("SELECT 1 FROM percone WHERE person_id = %s", (person_id,))
            if not cursor.fetchone():
                return False

            for ph in photos_data:
                photo_id = ph.get("photo_id", str(uuid.uuid4()))
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"{person_id}_{timestamp}_{photo_id}.jpg"
                file_path = os.path.join(PERSON_PHOTOS_PATH, filename)

                try:
                    img_data = base64.b64decode(ph["base64"])
                    os.makedirs(os.path.dirname(file_path), exist_ok=True)
                    with open(file_path, "wb") as f:
                        f.write(img_data)
                except Exception as e:
                    print(f"Ошибка сохранения фото {filename}: {e}")
                    continue

                cursor.execute(
                    """
                    INSERT INTO photo
                        (filein, person_id, photo_id, quality, photo_dttm, vector, checksum, view_photo)
                    VALUES (%s, %s, %s, %s, %s, %s::vector, %s, %s)
                """,
                    (
                        file_path,
                        person_id,
                        photo_id,
                        ph.get("quality"),
                        datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                        ph["vector"],
                        ph.get("checksum"),
                        True,
                    ),
                )
            return True

    def get_person_info(self, user_id: str, person_id: str) -> List[Dict]:
        with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
            if person_id:
                cursor.execute(
                    """
                    SELECT p.user_id, p.person_id, p.description, p.tag, p.percone_dttm,
                           (SELECT array_agg(ph.photo_id) FROM photo ph WHERE ph.person_id = p.person_id AND ph.view_photo) as photos
                    FROM percone p
                    WHERE p.user_id = %s AND p.person_id = %s AND p.view_percone
                """,
                    (user_id, person_id),
                )
            else:
                cursor.execute(
                    """
                    SELECT p.user_id, p.person_id, p.description, p.tag, p.percone_dttm,
                           (SELECT array_agg(ph.photo_id) FROM photo ph WHERE ph.person_id = p.person_id AND ph.view_photo) as photos
                    FROM percone p
                    WHERE p.user_id = %s AND p.view_percone
                """,
                    (user_id,),
                )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def get_photo_info(
        self, user_id: str = "1", person_id: str = None, photo_id: str = None
    ) -> List[Dict]:
        query = """
            SELECT ph.photo_id, ph.filein, ph.quality, ph.photo_dttm, p.person_id
            FROM photo ph
            JOIN percone p ON ph.person_id = p.person_id
            WHERE p.view_percone AND ph.view_photo
        """
        params = []
        if user_id:
            query += " AND p.user_id = %s"
            params.append(user_id)
        if person_id:
            query += " AND ph.person_id = %s"
            params.append(person_id)
        if photo_id:
            query += " AND ph.photo_id = %s"
            params.append(photo_id)

        with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(query, params)
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def add_camera(self, camera_data: Dict) -> str:
        """Добавляет или обновляет камеру в БД"""
        query = """
            INSERT INTO cameras (
                cam_id, name, description, stream_to_parse, user_id, face_width_max,
                timedelay, resize, crop_params, extraqueue, status,
                motion_min_area, motion_threshold, motion_record_after_time,
                created_at, updated_at
            ) VALUES (
                %s, %s, %s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s, NOW(), NOW()
            ) ON CONFLICT (cam_id) DO UPDATE SET
                name = EXCLUDED.name,
                description = EXCLUDED.description,
                stream_to_parse = EXCLUDED.stream_to_parse,
                user_id = EXCLUDED.user_id,
                face_width_max = EXCLUDED.face_width_max,
                timedelay = EXCLUDED.timedelay,
                resize = EXCLUDED.resize,
                crop_params = EXCLUDED.crop_params,
                extraqueue = EXCLUDED.extraqueue,
                status = EXCLUDED.status,
                motion_min_area = EXCLUDED.motion_min_area,
                motion_threshold = EXCLUDED.motion_threshold,
                motion_record_after_time = EXCLUDED.motion_record_after_time,
                updated_at = NOW()
        """

        params = (
            camera_data["cam_id"],
            camera_data["name"],
            camera_data.get("description", camera_data["name"]),
            camera_data["stream_to_parse"],
            camera_data["user_id"],
            camera_data.get("face_width_max", 50),
            camera_data.get("timedelay", 333),
            camera_data.get("resize"),
            (
                json.dumps(camera_data.get("crop_params"))
                if camera_data.get("crop_params")
                else None
            ),
            camera_data.get("extraqueue", 1),
            camera_data.get("status", "active"),
            camera_data.get("motion_min_area", 500),
            camera_data.get("motion_threshold", 25),
            camera_data.get("motion_record_after_time", 3),
        )

        try:
            with self._get_cursor() as cursor:
                cursor.execute(query, params)
            return camera_data["cam_id"]
        except Exception as e:
            print(f"Ошибка в add_camera: {e}")
            import traceback

            traceback.print_exc()
        raise

    def get_all_cameras(self, user_id: str = "1") -> List[Dict]:
        """Получает список камер из БД"""
        with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
            if user_id:
                cursor.execute(
                    "SELECT * FROM cameras WHERE user_id = %s AND status = 'active' ORDER BY created_at",
                    (user_id,),
                )
            else:
                cursor.execute(
                    "SELECT * FROM cameras WHERE status = 'active' ORDER BY created_at"
                )
            return [dict(row) for row in cursor.fetchall()]

    def get_camera(self, cam_id: str) -> Optional[Dict]:
        """Получает камеру по ID"""
        with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT * FROM cameras WHERE cam_id = %s", (cam_id,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def suspend_camera(self, cam_id: str) -> bool:
        """Помечает камеру как suspended (или удаляет)"""
        with self._get_cursor() as cursor:
            cursor.execute(
                "UPDATE cameras SET status = 'suspended', updated_at = NOW() WHERE cam_id = %s",
                (cam_id,),
            )
            return cursor.rowcount > 0

    def delete_camera(self, cam_id: str) -> bool:
        """Полностью удаляет камеру из БД"""
        with self._get_cursor() as cursor:
            cursor.execute("DELETE FROM cameras WHERE cam_id = %s", (cam_id,))
            return cursor.rowcount > 0

    def delete_person(self, person_id: str) -> Tuple[int, int]:
        with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT filein FROM photo WHERE person_id = %s", (person_id,)
            )
            files = [row["filein"] for row in cursor.fetchall()]

            cursor.execute("DELETE FROM photo WHERE person_id = %s", (person_id,))
            photo_deleted = cursor.rowcount

            cursor.execute("DELETE FROM percone WHERE person_id = %s", (person_id,))
            percone_deleted = cursor.rowcount

        # Удаляем файлы с диска
        for fp in files:
            try:
                if os.path.exists(fp):
                    os.remove(fp)
            except Exception as e:
                print(f"Ошибка удаления файла {fp}: {e}")

        return percone_deleted, photo_deleted

    def delete_photo(self, photo_id: str) -> Tuple[int, str]:
        with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute("SELECT filein FROM photo WHERE photo_id = %s", (photo_id,))
            row = cursor.fetchone()
            file_path = row["filein"] if row else None
            cursor.execute("DELETE FROM photo WHERE photo_id = %s", (photo_id,))
            deleted = cursor.rowcount
        return deleted, file_path

    def find_similar_faces(
        self, user_id: str, embedding: List[float], limit: int = 3
    ) -> List[Dict]:
        emb_str = self._embedding_to_str(embedding)
        with self._get_cursor(cursor_factory=RealDictCursor) as cursor:
            cursor.execute(
                "SELECT * FROM find_similar_faces(%s, %s::vector, %s)",
                (user_id, emb_str, limit),
            )
            rows = cursor.fetchall()
            return [
                {
                    "user_id": row["user_id"],
                    "person_id": row["person_id"],
                    "photo_id": row["photo_id"],
                    "description": row["description"],
                    "tag": row["tag"],
                    "distance": float(row["distance"]),
                }
                for row in rows
            ]

    def find_or_create_vector(
        self, embedding: List[float], max_distance: float = 0.9, image_data: str = ""
    ) -> Dict:
        emb_str = self._embedding_to_str(embedding)
        new_uuid = str(uuid.uuid4())
        try:
            with self._get_cursor(cursor_factory=DictCursor) as cursor:
                cursor.execute(
                    "SELECT * FROM find_or_create_by_vector(%s, %s::vector, %s, %s)",
                    (new_uuid, emb_str, max_distance, image_data),
                )
                result = cursor.fetchone()
                return {
                    "id": result["result_uuid"],
                    "action": result["action_type"],
                    "distance": (
                        result["distance"]
                        if result["distance"] is not None
                        else max_distance
                    ),
                    "status": "success",
                }
        except Exception as e:
            return {
                "status": "error",
                "message": str(e),
                "id": None,
                "action": None,
                "distance": max_distance,
            }

    def process_face_recognition(
        self,
        embedding: List[float],
        dtime: str,
        camera_id: str,
        user_id: str = "1",
        max_distance: float = 0.9,
        is_real: bool = False,
        is_multiple: bool = False,
        percent_unknown: float = 95,
        image_data: str = "",
    ) -> Tuple[Dict, str]:
        try:
            results = self.find_similar_faces(
                user_id=user_id, embedding=embedding, limit=1
            )

            if not results:
                distance = 2.0
                person_id = None
            else:
                distance = results[0]["distance"]
                person_id = results[0]["person_id"]

            percent = self._percentage_fcb(distance)

            item = {
                "type": "face_begin",
                "version": "0.8",
                "id": str(uuid.uuid1()),
                "datetime": self._format_datetime(dtime),
                "user_id": user_id,
                "camera_id": camera_id,
                "data": {
                    "photobank_type": "local",
                    "person": {
                        "id": person_id,
                        "percent": round(percent, 2),
                        "is_real": is_real,
                        "is_multiple": is_multiple,
                    },
                },
            }

            if percent > percent_unknown:
                item["data"]["person"].update(
                    {
                        "facerecognized": True,
                        "id": person_id,
                        "description": (
                            results[0].get("description", "") if results else ""
                        ),
                        "person_unknown": False,
                        "percent_unknown": 0,
                        "person_unknown_new": False,
                    }
                )
                datacuttxt = (
                    f"{results[0].get('description', '')} ({percent:.2f})"
                    if results
                    else f"Unknown ({percent:.2f})"
                )
            else:
                unk_result = self.find_or_create_vector(
                    embedding=embedding,
                    max_distance=max_distance,
                    image_data=image_data,
                )
                distance2 = unk_result.get("distance") or max_distance
                percent2 = self._percentage_fcb(distance2)

                # Сохраняем unknown_uuid в поле data для передачи в log_event
                unk_id = unk_result.get("id", "")

                item["data"]["person"].update(
                    {
                        "facerecognized": False,
                        "id": unk_id,
                        "description": "",
                        "person_unknown": True,
                        "percent_unknown": round(percent2, 2),
                        "person_unknown_new": unk_result.get("action") == "created",
                        "unknown_uuid": unk_id,  # добавляем для передачи
                    }
                )
                action_text = (
                    "new unknown"
                    if unk_result.get("action") == "created"
                    else "existing unknown"
                )
                datacuttxt = (
                    f"{action_text} (Поиск по базе {percent:.2f} ({distance:.2f}) < {percent_unknown} -> "
                    f"Поиск по неизв. {percent2:.2f}% ({distance2:.2f} / {max_distance})"
                )
            return item, datacuttxt
        except Exception as e:
            import traceback

            error_line = traceback.extract_tb(e.__traceback__)[-1].lineno
            raise RuntimeError(
                f"Face recognition processing failed: {str(e)} at line {error_line}"
            )

    def log_event(self, event_data: Dict) -> bool:
        try:
            # Базовые данные
            data_json = {
                "camera_name": event_data.get("camera_name"),
                "face_width": event_data.get("face_width"),
                "snapshot_path": event_data.get("snapshot_path"),
            }

            # Добавляем unknown_uuid, если это неизвестное лицо
            if event_data.get("is_unknown") and event_data.get("unknown_uuid"):
                data_json["unknown_uuid"] = event_data.get("unknown_uuid")

            # Убираем None значения
            data_json = {k: v for k, v in data_json.items() if v is not None}

            person_photobank_id = event_data.get("person_id") or ""
            is_unknown = event_data.get("is_unknown", person_photobank_id == "")

            with self._get_cursor() as cursor:
                cursor.execute(
                    """
                    INSERT INTO analytics_events
                        (datetime, camera_id, type, person_photobank_id, event_id, created_at, data, is_unknown)
                    VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s)
                    """,
                    (
                        event_data["datetime"],
                        event_data["camera_id"],
                        "face_detected",
                        person_photobank_id,
                        event_data["event_id"],
                        datetime.now(),
                        json.dumps(data_json),
                        is_unknown,
                    ),
                )
            return True
        except Exception as e:
            print(f"Ошибка логирования события: {e}")
            return False

    def close_all_connections(self):
        with FRDatabase._pool_lock:
            if FRDatabase._connection_pool:
                FRDatabase._connection_pool.closeall()
                FRDatabase._connection_pool = None
                print("Все соединения PostgreSQL (FR) закрыты")

    def __del__(self):
        self.close_all_connections()
