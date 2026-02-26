# ver 0.05
import queue as Queue
import cv2, os
from threading import Lock, Thread
from time import sleep
from datetime import datetime as dt
from typing import Generator

os.environ['OPENCV_LOG_LEVEL'] = 'ERROR'  # или 'SILENT'
os.environ["OPENCV_FFMPEG_LOGLEVEL"] = "quiet"  # или "fatal", "error"


class VideoCapture:
    def __init__(self, name, namestream='NoName', resize=None, crop_params=None):
        # Инициализация всех атрибутов в начале
        self.thread = None
        self.stopped = False
        self.cap = None
        self.frame = None
        self.q2 = None
        self.initialized = False

        try:
            # Параметры для повторных попыток
            self.max_retries = 10
            self.retry_delay = 5
            self.retry_count = 0

            self.namestream = namestream
            self.resize = resize
            self.crop_params = crop_params
            self.name = name
            self.ok = True

            # Инициализация видеозахвата с проверкой
            self._init_video_capture()

            # Чтение и обработка первого кадра
            self._process_first_frame()

            # Инициализация структур данных
            self.q2 = Queue.Queue(maxsize=25)
            self.newframe = True
            self.lock = Lock()

            # Запуск потока чтения
            self.thread = Thread(
                target=self._reader,
                name=f"VideoReader_{namestream}",
                daemon=True
            )
            self.thread.start()

            self.initialized = True

        except Exception as e:
            print(f"Ошибка инициализации VideoCapture {namestream}: {str(e)}")
            self.stop()
            raise

    def _init_video_capture(self):
        """Версия с ограничением максимальной задержки (не более 60 сек)"""
        max_attempts = 10
        base_delay = 10
        max_delay = 60

        for attempt in range(1, max_attempts + 1):
            try:
                self.cap = cv2.VideoCapture(self.name, cv2.CAP_FFMPEG)
                self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)

                if self.cap.isOpened():
                    return

                self.cap.release()

            except Exception as e:
                print(f"Attempt {attempt} failed: {e}")

            sleep(min(base_delay * attempt, max_delay))

        raise RuntimeError(f"Failed to open video stream after {max_attempts} attempts")

    '''
    def _init_video_capture(self):
        """Отдельный метод для инициализации видеозахвата"""
        self.cap = cv2.VideoCapture(self.name)
        self.cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 15000)

        if not self.cap.isOpened():
            raise RuntimeError(f"Не удалось открыть видеопоток {self.name}")
    '''

    def _process_first_frame(self):
        """Обработка первого кадра"""
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Не удалось прочитать первый кадр")

        self.resizeY, self.resizeX, _ = frame.shape
        self.resizeYrun, self.resizeXrun = self.resizeY, self.resizeX
        print(f"Первый кадр успешно прочитан. {self.namestream} {self.resizeY}x{self.resizeX}")

        # Обработка параметров масштабирования и обрезки
        if self.resize is not None:
            if isinstance(self.resize, (list, tuple)):
                self.resizeX, self.resizeY = self.resize
                resize_factor = self.resizeY / frame.shape[0]
            else:
                resize_factor = self.resize
                self.resizeY = int(self.resizeY * resize_factor) if resize_factor > 0 else -int(self.resizeY / resize_factor)
                self.resizeX = int(self.resizeX * resize_factor) if resize_factor > 0 else -int(self.resizeX / resize_factor)

            if self.crop_params is not None:
                self.crop_params = [
                    int(p * resize_factor) if resize_factor > 0 else int(-p / resize_factor)
                    for p in self.crop_params
                ]

        # Применение изменений к кадру
        frame = self.reset(frame)
        self.frame = frame

    def reset(self, frame):
        """Метод для изменения масштаба и обрезки кадра"""
        if frame is None:
            return None

        if self.resize is not None:
            frame = cv2.resize(
                frame,
                (self.resizeX, self.resizeY),
                interpolation=cv2.INTER_AREA
            )

        if self.crop_params is not None:
            y1, y2, x1, x2 = self.crop_params
            frame = frame[y1:y2, x1:x2]

        return frame

    def _reader(self):
        """Основной цикл чтения кадров"""
        try:
            while not self.stopped:
                ret, frame = self.cap.read()

                if not ret:
                    self._handle_read_error()
                    continue

                self.retry_count = 0
                processed_frame = self.reset(frame)

                if processed_frame is not None:
                    with self.lock:
                        self.newframe = True
                        self.frame = processed_frame

        except Exception as e:
            print(f"Критическая ошибка в потоке {self.name} {self.namestream}: {str(e)}")
            self.stop()

    def _handle_read_error(self):
        """Обработка ошибок чтения кадра"""
        print(f"Ошибка чтения кадра. {self.name} Попытка {self.retry_count + 1}/{self.max_retries}, задержка = {self.retry_delay + self.retry_count*self.retry_delay*3}")
        self.cap.release()
        sleep(self.retry_delay + self.retry_count*self.retry_delay*3)

        try:
            self.cap = cv2.VideoCapture(self.name)
            if not self.cap.isOpened():
                raise RuntimeError(f"Не удалось переоткрыть видеопоток {self.name}")
        except Exception as e:
            print(f"Ошибка переоткрытия потока: {str(e)}")

        self.retry_count += 1
        if self.retry_count >= self.max_retries:
            self.stopped = True
            self.stop()

    def read(self):
        """Безопасное получение кадра"""
        if not hasattr(self, 'frame') or self.frame is None:
            return None

        with self.lock:
            if self.newframe:
                self.newframe = False
                return self.frame.copy()  # Возвращаем копию для безопасности
        return None

    def stop(self):
        """Безопасная остановка всех компонентов"""
        if self.stopped:
            return

        self.stopped = True

        # Остановка потока
        if hasattr(self, 'thread') and self.thread is not None:
            try:
                if self.thread.is_alive():
                    self.thread.join(timeout=1.0)
            except Exception as e:
                print(f"Ошибка остановки потока {self.name}: {str(e)}")

        # Освобождение видеозахвата
        if hasattr(self, 'cap') and self.cap is not None:
            try:
                self.cap.release()
            except:
                pass

        # Очистка очереди
        if hasattr(self, 'q2') and self.q2 is not None:
            try:
                with self.q2.mutex:
                    self.q2.queue.clear()
            except:
                pass

        print(f"Видеозахват {self.namestream} остановлен")

    # Остальные методы (add2, read2, size2) остаются без изменений
    def add2(self, count=1):
        """
        Метод для добавления кадров в очередь (в текущей реализации не используется).

        :param count: Количество кадров для добавления (по умолчанию 1).
        """
        if count==0:
            return

        if self.q2.full():
            return

        delay = 0.1  # Начальная задержка
        for _ in range(count):
            while True:  # Бесконечный цикл для ожидания нового кадра
                with self.lock:  # Используем контекстный менеджер для работы с блокировкой
                    if self.newframe:  # Проверяем, есть ли новый кадр
                        # Добавляем кадр и метку времени в очередь
                        try:
                            self.q2.put((self.frame, dt.now().strftime('%Y%m%d-%H%M%S-%f')), block=False)
                        except Queue.Full:
                            pass
                        self.newframe = False  # Сбрасываем флаг нового кадра
                        break  # Выходим из цикла ожидания
                sleep(delay)  # Задержка для уменьшения нагрузки на CPU
                delay = min(delay + 0.05, 0.5)  # Увеличение задержки, но не более 0.5 секунды

    def read2(self):
        """
        Метод для получения кадра из очереди.

        :return: Кадр из очереди.
        """
        return self.q2.get()

    def size2(self):
        """
        Метод для получения размера очереди.

        :return: Размер очереди.
        """
        return self.q2.qsize()
