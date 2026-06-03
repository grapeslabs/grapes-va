#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# ver 0.07 - 2026_05
"""
Модуль camstream.py – класс VideoCapture для асинхронного чтения видеопотоков.

Класс предназначен для:
- Захвата видео из файла, RTSP, IP-камеры или веб-камеры.
- Автоматического переподключения при обрыве потока.
- Масштабирования (resize) и обрезки (crop) кадров.
- Потокового чтения с буферизацией последнего кадра.
- Опциональной очереди для накопления кадров с временными метками.

Параметры конструктора __init__:
    name (str)                 : источник видео (путь к файлу, URL RTSP, номер камеры 0,1…)
    namestream (str)           : произвольное имя потока (для логов и идентификации)
    resize (None, int, float, tuple) :
        - None   : без изменения размера
        - int/float >0 : коэффициент масштабирования (0.5 – уменьшить вдвое)
        - int/float <0 : коэффициент увеличения (работает как -1/коэфф)
        - (width, height) : абсолютные размеры в пикселях
    crop_params (None, list, tuple, dict) :
        - None                   : без обрезки
        - (y1, y2, x1, x2)       : кортеж/список из 4 целых чисел (начало/конец по Y, X)
        - {'y1':y1,'y2':y2,'x1':x1,'x2':x2} : словарь с теми же ключами
        - Пустой список/словарь  : отключает обрезку
        Обрезка применяется ПОСЛЕ масштабирования (если resize задан).
    Порядок применения: resize → crop (можно изменить, переопределив reset).

Основные методы:
    read() -> frame (numpy.ndarray) или None
        Возвращает копию последнего обработанного кадра (не блокирует).
    stop() -> None
        Останавливает поток чтения и освобождает ресурсы.

Методы для работы с очередью кадров (не используются по умолчанию):
    add2(count=1) -> None
        Помещает в очередь count копий последнего кадра (если есть новый кадр).
    read2() -> (frame, timestamp_str)
        Извлекает из очереди кадр и строку времени (формат "YYYYMMDD-HHMMSS-ffffff").
    size2() -> int
        Возвращает текущий размер очереди.

Внутренние методы:
    _init_video_capture()      – открытие потока с повторными попытками.
    _process_first_frame()     – чтение первого кадра, инициализация размеров,
                                 пересчёт crop_params при resize.
    reset(frame)               – применяет resize и crop к кадру.
    _reader()                  – основной цикл потока чтения.
    _handle_read_error()       – обработка ошибок чтения, переподключение.
    _normalize_crop_params()   – приводит crop_params к единому формату (y1,y2,x1,x2).
"""

# ver 0.07 - 2026_05
import queue as Queue
...
import queue as Queue
import cv2, os
from threading import Lock, Thread
from time import sleep
from datetime import datetime as dt
from typing import Generator


class VideoCapture:
    """
    Асинхронный захват видео с автоматическим восстановлением соединения.

    Пример использования:
        cap = VideoCapture("rtsp://...", namestream="cam1", resize=0.5, crop_params=(100,300,50,250))
        while True:
            frame = cap.read()
            if frame is not None:
                cv2.imshow("frame", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        cap.stop()
    """

    def __init__(self, name, namestream="NoName", resize=None, crop_params=None):
        """
        Инициализация видеозахвата.

        :param name: источник видео (файл, RTSP, номер камеры)
        :param namestream: имя потока (для логов)
        :param resize: параметры масштабирования (None, коэффициент или (ширина, высота))
        :param crop_params: параметры обрезки (None, (y1,y2,x1,x2) или словарь)
        """
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
            self.crop_params = self._normalize_crop_params(crop_params)  # приведение к единому формату
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
                target=self._reader, name=f"VideoReader_{namestream}", daemon=True
            )
            self.thread.start()

            self.initialized = True

        except Exception as e:
            print(f"Ошибка инициализации VideoCapture {namestream}: {str(e)}")
            self.stop()
            raise

    def _init_video_capture(self):
        """
        Открытие видеопотока с повторными попытками и таймаутом.
        Используется cv2.CAP_FFMPEG для поддержки RTSP.
        """
        max_attempts = 30
        base_delay = 10
        max_delay = 180

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

    def _process_first_frame(self):
        """
        Чтение первого кадра для определения исходных размеров.
        Вычисляет параметры масштабирования и обрезки, затем применяет их к первому кадру.
        """
        ret, frame = self.cap.read()
        if not ret:
            raise RuntimeError("Не удалось прочитать первый кадр")

        self.resizeY, self.resizeX, _ = frame.shape
        self.resizeYrun, self.resizeXrun = self.resizeY, self.resizeX
        print(
            f"Первый кадр успешно прочитан. {self.namestream} {self.resizeY}x{self.resizeX}"
        )

        # Обработка параметров масштабирования и обрезки
        resize_factor = 1
        if self.resize is not None:
            # Если resize задан как (ширина, высота) – абсолютные значения
            if isinstance(self.resize, (list, tuple)):
                self.resizeX, self.resizeY = self.resize
                resize_factor = self.resizeY / frame.shape[0]
            else:
                # Если resize – число: коэффициент масштабирования
                resize_factor = self.resize
                self.resizeY = (
                    int(self.resizeY * resize_factor)
                    if resize_factor > 0
                    else -int(self.resizeY / resize_factor)
                )
                self.resizeX = (
                    int(self.resizeX * resize_factor)
                    if resize_factor > 0
                    else -int(self.resizeX / resize_factor)
                )

            # Если задана обрезка – пересчитываем координаты с учётом масштаба
            if self.crop_params is not None:
                self.crop_params = [
                    (
                        int(p * resize_factor)
                        if resize_factor > 0
                        else int(-p / resize_factor)
                    )
                    for p in self.crop_params
                ]

        # Применение изменений к кадру
        frame = self.reset(frame)
        self.height, self.width = frame.shape[:2]
        self.frame = frame

    def reset(self, frame):
        """
        Применяет масштабирование и обрезку к одному кадру.
        ПОРЯДОК: сначала resize, потом crop (если нужно изменить – поменяйте блоки местами).

        :param frame: исходный кадр (numpy.ndarray)
        :return: обработанный кадр или None
        """
        if frame is None:
            return None

        if self.resize is not None:
            frame = cv2.resize(
                frame, (self.resizeX, self.resizeY), interpolation=cv2.INTER_AREA
            )

        if self.crop_params is not None:
            y1, y2, x1, x2 = self.crop_params
            frame = frame[y1:y2, x1:x2]

        return frame

    def _reader(self):
        """
        Основной поток чтения кадров. Работает в бесконечном цикле, пока self.stopped == False.
        При ошибке чтения вызывает _handle_read_error().
        """
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
            print(
                f"Критическая ошибка в потоке {self.name} {self.namestream}: {str(e)}"
            )
            self.stop()

    def _handle_read_error(self):
        """
        Обработка ошибки чтения кадра: освобождает старый захват, делает паузу,
        пытается переоткрыть поток. При превышении max_retries останавливает всё.
        """
        print(
            f"Ошибка чтения кадра. {self.name} Попытка {self.retry_count + 1}/{self.max_retries}, задержка = {self.retry_delay + self.retry_count*self.retry_delay*3}"
        )
        self.cap.release()
        sleep(self.retry_delay + self.retry_count * self.retry_delay * 3)

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
        """
        Возвращает копию последнего обработанного кадра (не блокирует).
        Если новый кадр ещё не появился, возвращает None.

        :return: кадр (numpy.ndarray) или None
        """
        if not hasattr(self, "frame") or self.frame is None:
            return None

        with self.lock:
            if self.newframe:
                self.newframe = False
                return self.frame.copy()  # Возвращаем копию для безопасности
        return None

    def stop(self):
        """
        Останавливает поток чтения, освобождает видеозахват и очищает очередь.
        Безопасно вызывать несколько раз.
        """
        if self.stopped:
            return

        self.stopped = True

        # Остановка потока
        if hasattr(self, "thread") and self.thread is not None:
            try:
                if self.thread.is_alive():
                    self.thread.join(timeout=1.0)
            except Exception as e:
                print(f"Ошибка остановки потока {self.name}: {str(e)}")

        # Освобождение видеозахвата
        if hasattr(self, "cap") and self.cap is not None:
            try:
                self.cap.release()
            except:
                pass

        # Очистка очереди
        if hasattr(self, "q2") and self.q2 is not None:
            try:
                with self.q2.mutex:
                    self.q2.queue.clear()
            except:
                pass

        print(f"Видеозахват {self.namestream} остановлен")

    # ---------- Методы для работы с очередью (опционально) ----------
    def add2(self, count=1):
        """
        Добавляет в очередь count копий последнего кадра (если есть новый кадр).
        Каждый кадр сохраняется вместе с временной меткой создания.
        Очередь ограничена максимальным размером (maxsize=25).

        :param count: количество кадров для добавления (по умолчанию 1)
        """
        if count == 0:
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
                            self.q2.put(
                                (self.frame, dt.now().strftime("%Y%m%d-%H%M%S-%f")),
                                block=False,
                            )
                        except Queue.Full:
                            pass
                        self.newframe = False  # Сбрасываем флаг нового кадра
                        break  # Выходим из цикла ожидания
                sleep(delay)  # Задержка для уменьшения нагрузки на CPU
                delay = min(
                    delay + 0.05, 0.5
                )  # Увеличение задержки, но не более 0.5 секунды

    def read2(self):
        """
        Извлекает из очереди кадр и временную метку.
        Блокирует выполнение, если очередь пуста.

        :return: (кадр, строка времени) – строка формата "YYYYMMDD-HHMMSS-ffffff"
        """
        return self.q2.get()

    def size2(self):
        """
        Возвращает текущий размер очереди.

        :return: количество элементов в очереди
        """
        return self.q2.qsize()

    def _normalize_crop_params(self, crop_params):
        """
        Приводит crop_params к единому формату (y1, y2, x1, x2) или None.
        Поддерживаются:
        - None -> None
        - кортеж/список из 4 чисел (y1, y2, x1, x2)
        - словарь с ключами 'y1','y2','x1','x2' (или 'y1','y2','x1','x2')
        - пустой кортеж/список/словарь -> None (отключает обрезку)
        """
        if crop_params is None:
            return None

        # Если это кортеж или список
        if isinstance(crop_params, (tuple, list)):
            if len(crop_params) == 0:
                return None
            if len(crop_params) == 4:
                # Проверяем, что все элементы числа
                if all(isinstance(v, (int, float)) for v in crop_params):
                    # Приводим к int (координаты пикселей)
                    y1, y2, x1, x2 = (int(v) for v in crop_params)
                    return (y1, y2, x1, x2)
            raise ValueError(f"Некорректный формат crop_params (tuple/list): {crop_params}")

        # Если это словарь
        if isinstance(crop_params, dict):
            if len(crop_params) == 0:
                return None
            # Ищем ключи: могут быть 'y1','x1','y2','x2' или 'y1','y2','x1','x2'
            required = ['y1', 'y2', 'x1', 'x2']
            # Проверяем наличие всех ключей
            if all(k in crop_params for k in required):
                y1 = int(crop_params['y1'])
                y2 = int(crop_params['y2'])
                x1 = int(crop_params['x1'])
                x2 = int(crop_params['x2'])
                return (y1, y2, x1, x2)
            raise ValueError(f"Некорректный формат crop_params (dict): {crop_params}")

        raise TypeError(f"Неподдерживаемый тип crop_params: {type(crop_params)}")
