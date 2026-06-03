from ultralytics import YOLO
import cv2
from typing import List, Tuple, Dict, Union, Optional
import os
import numpy as np

class PersonDetector:
    """Класс для детекции людей на изображениях с поддержкой работы с файлами и фреймами"""

    def __init__(self, model_size: str = 'n'):
        """
        Инициализация детектора

        Args:
            model_size: размер модели YOLOv8 ('n', 's', 'm', 'l', 'x')
        """
        self.model = YOLO(f'detection_figure_model.pt')

    def detect_in_frame(self, frame: np.ndarray, conf_threshold: float = 0.5, colorrectangle = (255, 255, 0), thickness = 3, putTextinfo = True ) -> Tuple[List[List[int]], np.ndarray]:
        """
        Детекция людей во фрейме (изображении в виде numpy массива)

        Args:
            frame: изображение в формате numpy array (BGR)
            conf_threshold: порог уверенности для детекции

        Returns:
            Tuple[List[List[int]], np.ndarray]: список bounding boxes и фрейм с рамками
        """
        # Копируем фрейм для рисования рамок
        annotated_frame = frame.copy()

        # Детекция
        results = self.model(frame, classes=[0], conf=conf_threshold, verbose=False)

        boxes = []

        for result in results:
            for box in result.boxes:
                if box.cls == 0:  # класс 0 соответствует "person" в COCO
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    confidence = float(box.conf[0])
                    boxes.append([x1, y1, x2, y2, confidence])

                    # Рисование рамки
                    if thickness:
                        cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), colorrectangle, thickness)
                    if putTextinfo:
                        cv2.putText(annotated_frame, f'{confidence:.2f}',
                               (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX,
                               1, colorrectangle, 2)

        return boxes, annotated_frame

    def detect(self, input_data: Union[str, np.ndarray],
               save_picture: bool = False,
               conf_threshold: float = 0.5) -> Dict:
        """
        Универсальный метод для детекции людей

        Args:
            input_data: путь к файлу или фрейм (numpy array)
            save_picture: сохранять ли изображение с рамками (только для файлов)
            conf_threshold: порог уверенности для детекции

        Returns:
            Dict: словарь с результатами детекции
        """
        is_file = isinstance(input_data, str)

        # Загрузка изображения или использование фрейма
        if is_file:
            frame = cv2.imread(input_data)
            if frame is None:
                return {
                    'success': False,
                    'error': f"Не удалось загрузить изображение: {input_data}",
                    'input_type': 'file',
                    'boxes': []
                }
        else:
            frame = input_data.copy()

        # Детекция во фрейме
        boxes, annotated_frame = self.detect_in_frame(frame, conf_threshold)

        result = {
            'success': True,
            'person_count': len(boxes),
            'boxes': boxes,
            'annotated_frame': annotated_frame,
            'input_type': 'file' if is_file else 'frame'
        }

        # Сохранение изображения с рамками (только для файлов)
        if is_file and save_picture:
            try:
                base_name = os.path.splitext(input_data)[0]
                result_path = f"{base_name}_result.jpg"
                cv2.imwrite(result_path, annotated_frame)
                result['saved_path'] = result_path
            except Exception as e:
                result['save_error'] = str(e)

        return result

    def process_image(self, image_path: str, save_picture: bool = False) -> Dict:
        """
        Удобный метод для обработки изображения по пути (обратная совместимость)

        Args:
            image_path: путь к файлу изображения
            save_picture: сохранять ли изображение с рамками

        Returns:
            Dict: словарь с результатами детекции
        """
        return self.detect(image_path, save_picture)

    def process_frame(self, frame: np.ndarray) -> Dict:
        """
        Удобный метод для обработки фрейма

        Args:
            frame: изображение в формате numpy array

        Returns:
            Dict: словарь с результатами детекции
        """
        return self.detect(frame, save_picture=False)


# Пример использования с файлами
if __name__ == "__main__":
    # Создание детектора
    detector = PersonDetector('m')  # medium модель

    print("=== Обработка файлов ===")
    # Обработка списка файлов
    for nf in ['126-007_prn.jpg',
                '2018_10_02_10_59_MG_6007_DxO_cr_prn.jpg',
                '2019_03_07_DSCN0710!_prn.jpg',
                '2019_06_04_D81_4653!_prn.jpg']:

        # Проверяем существование файла
        if os.path.exists(nf):
            result = detector.process_image(nf, save_picture=True)
            print(f"\n{nf}:")
            print(f"  Найдено людей: {result['person_count']}")
            print(f"  Bounding boxes: {result['boxes']}")
            if result.get('saved_path'):
                print(f"  Результат сохранен: {result['saved_path']}")
            if 'error' in result:
                print(f"  Ошибка: {result['error']}")
        else:
            print(f"\nФайл не найден: {nf}")

    print("\n=== Обработка фреймов ===")
    # Пример работы с фреймами
    # Создаем тестовый фрейм или загружаем изображение как фрейм
    test_frame = cv2.imread('126-007_prn.jpg')

    if test_frame is not None:
        # Обработка фрейма
        result = detector.process_frame(test_frame)

        print(f"\nОбработка фрейма:")
        print(f"  Размер фрейма: {test_frame.shape}")
        print(f"  Найдено людей: {result['person_count']}")
        print(f"  Bounding boxes: {result['boxes']}")

        # Показать результат с рамками (если нужно)
        if result['success']:
            # Можем отобразить фрейм с рамками
            annotated_frame = result['annotated_frame']

            # Сохраняем результат во временный файл для демонстрации
            cv2.imwrite('frame_result_demo.jpg', annotated_frame)
            print(f"  Демо-результат сохранен в: frame_result_demo.jpg")

            # Если хотим показать в окне (закомментировано для скрипта)
            # cv2.imshow('Detected Persons', annotated_frame)
            # cv2.waitKey(0)
            # cv2.destroyAllWindows()

    print("\n=== Прямая работа с фреймами ===")
    # Пример прямого использования detect_in_frame для потоковой обработки

    # Симулируем получение фрейма от камеры или видеопотока
    cap = cv2.VideoCapture(0)  # 0 - индекс камеры, можно заменить на путь к видео

    if not cap.isOpened():
        print("Камера недоступна, используем тестовое изображение")
        test_image = cv2.imread('126-007_prn.jpg')
        if test_image is not None:
            # Детекция во фрейме без сохранения файла
            boxes, annotated_frame = detector.detect_in_frame(test_image)

            print(f"Найдено людей: {len(boxes)}")
            print(f"Bounding boxes: {boxes}")

            # Сохраняем для демонстрации
            cv2.imwrite('direct_frame_result.jpg', annotated_frame)
            print(f"Результат сохранен в: direct_frame_result.jpg")
    else:
        print("Камера доступна, захватываем один кадр...")
        ret, frame = cap.read()
        if ret:
            boxes, annotated_frame = detector.detect_in_frame(frame)
            print(f"Найдено людей в кадре с камеры: {len(boxes)}")

            # Сохраняем результат
            cv2.imwrite('camera_frame_result.jpg', annotated_frame)
            print(f"Результат сохранен в: camera_frame_result.jpg")

        cap.release()
