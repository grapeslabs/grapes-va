import cv2
import time
import numpy as np

class MotionDetector:
    def __init__(self, min_area=500, threshold=25, record_after_time=3):
        """
        Инициализация детектора движения

        Параметры:
        - min_area: минимальная площадь контура для детекции движения
        - threshold: порог бинаризации разницы кадров
        - record_after_time: время продолжения записи после последнего движения (в секундах)
        """
        self.min_area = min_area
        self.threshold = threshold
        self.record_after_time = record_after_time

        # Переменные состояния
        self.motion_detected = False
        self.recording_active = False
        self.last_motion_time = 0

        # Для обработки видео
        self.previous_frame = None

    def process_frame(self, frame, area_box_max_set = False):
        """
        Обработка кадра видеопотока

        Параметры:
        - frame: входной кадр (BGR)

        Возвращает:
        - motion_detected: обнаружено ли движение в текущем кадре
        - recording_active: активна ли запись (включая дополнительное время)
        """

        # Обработка изображения
        # 1. Конвертация в оттенки серого (уменьшение данных для обработки)
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        # 2. Размытие по Гауссу (ядро 21x21) для уменьшения шумов и мелких деталей
        gray = cv2.GaussianBlur(gray, (21, 21), 0)


        # Инициализируем предыдущий кадр при первом вызове
        if self.previous_frame is None:
            self.previous_frame = gray
            return False, False, 0, (0,0,0,0)

        # 3. Вычисление абсолютной разницы с предыдущим кадром
        frame_delta = cv2.absdiff(self.previous_frame, gray)
        # 4. Пороговая обработка (все что выше motion_threshold=25 становится 255, остальное 0)
        thresh = cv2.threshold(frame_delta, self.threshold, 255, cv2.THRESH_BINARY)[1]

        # Расширяем бинарное изображение
        # 5. Морфологическое расширение (заполнение дырок в контурах)
        thresh = cv2.dilate(thresh, None, iterations=2)

        # Находим контуры
        contours, _ = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # Сбрасываем флаг движения для текущего кадра
        current_motion = False

        # Проверяем контуры
        # Инициализация переменных
        max_area = 0
        best_box = (0, 0, 0, 0)  # (x, y, w, h)

        for contour in contours:
            current_area = cv2.contourArea(contour)

            # Пропускаем контуры меньше минимальной площади
            if current_area < self.min_area:
                continue

            # Движение обнаружено
            current_motion = True
            self.last_motion_time = time.time()

            # Получаем bounding box для текущего контура
            current_box = cv2.boundingRect(contour)

            # Обновляем максимальную площадь и соответствующий bbox
            if current_area > max_area:
                max_area = current_area
                best_box = current_box

            # Активируем запись при первом обнаружении движения
            if not self.recording_active:
                self.recording_active = True

            # Прерываем цикл если не требуется искать максимальный bbox
            if not area_box_max_set:
                break

        # После цикла:
        area = max_area
        area_box = best_box

        # Обновляем состояние
        self.motion_detected = current_motion

        # Проверяем, нужно ли продолжать запись
        if self.recording_active and not current_motion:
            if time.time() - self.last_motion_time > self.record_after_time:
                self.recording_active = False

        # Обновляем предыдущий кадр
        self.previous_frame = gray

        return self.motion_detected, self.recording_active, int(max_area), area_box

    def get_status(self):
        """
        Возвращает текущее состояние детектора

        Возвращает:
        - motion_detected: обнаружено ли движение
        - recording_active: активна ли запись
        """
        return self.motion_detected, self.recording_active

    def get_status_recording_active(self):
        """
        Возвращает текущее состояние детектора

        Возвращает:
        - recording_active: активна ли запись
        """
        if self.recording_active and time.time() - self.last_motion_time > self.record_after_time:
            self.recording_active = False
        return self.recording_active

if __name__ == '__main__':
    ## Пример использования
    # Инициализация видеопотока (0 - вебкамера, или путь к файлу)
    #cap = cv2.VideoCapture(0)
    # cam3
    cap = cv2.VideoCapture('rtsp://user:KKK123kkk@192.168.111.185:554/ISAPI/Streaming/Channels/101')

    # Создаем детектор движения
    detector = MotionDetector(min_area=2000, threshold=50, record_after_time=2)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Обрабатываем кадр
        motion, recording,  area, area_box  = detector.process_frame(frame)
        if motion : 
            i = area_box
            cv2.rectangle(
                    frame,
                    (int(i[0]), int(i[1])),
                    (int(i[0])+ int(i[2]), int(i[1]) + int(i[3])),
                    (0, 255, 0),
                    2,
                )


        status_info = [
    ("Motion: YES" if motion else "Motion: NO", (0, 0, 255) if motion else (0, 255, 0)),
    ("Recording: ON" if recording else "Recording: OFF", (0, 0, 255) if recording else (0, 255, 0)),
    (f"Area: {area}", (255, 255, 0))
]

# Отображаем статус на кадре
        for i, (text, color) in enumerate(status_info):
            cv2.putText(frame, text, (10, 30 + i * 30),cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)

        # Показываем кадр
        cv2.imshow("Motion Detection", frame)

        # Выход по нажатию 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


"""
## Описание работы

1. **Инициализация**:
   - `min_area`: минимальная площадь контура для детекции движения (игнорирует мелкие изменения)
   - `threshold`: порог для бинаризации разницы кадров
   - `record_after_time`: время продолжения записи после последнего обнаруженного движения

2. **Обработка кадров**:
   - Каждый кадр конвертируется в grayscale и размывается
   - Вычисляется разница с предыдущим кадром
   - Находятся контуры значительного размера
   - При обнаружении движения обновляется время последней активности

3. **Логика записи**:
   - Запись активируется при первом обнаружении движения
   - Продолжается в течение `record_after_time` секунд после последнего движения
   - Если движение продолжается, таймер сбрасывается

4. **Визуализация**:
   - На кадре отображается текущее состояние детектора
   - Можно добавить отрисовку контуров движения для наглядности
"""
