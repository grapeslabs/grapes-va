# built-in dependencies
model_path = "face_recognition_sface_2021dec.onnx"

from os.path import normpath, join, isfile, dirname, abspath
import cv2

# from numpy.linalg import norm as np_linalg_norm
import numpy as np

# ОПРЕДЕЛЯЕМ ТЕКУЩУЮ ПАПКУ СКРИПТА
current_dir = dirname(abspath(__file__))

for home in [
    current_dir,  # Сначала ищем в текущей папке (libs/pinfacekirjasto/face_recognition/sface/)
    "data/",
    "C:/Users/KKK/.deepface/weights/",
    "C:/Users/user/.deepface/weights/",
]:
    target_file = normpath(join(home, model_path))
    if isfile(target_file):
        break
    target_file = ""

if target_file == "":
    print("Not found ", model_path, " - stop")
    exit()

model_path = target_file
del target_file


class FaceRecognizerSface:
    def __init__(self, model_path=model_path):
        """
        Инициализация модели распознавания лиц.

        :param model_path: Путь к файлу модели (например, ONNX-файл).
        """
        self.recognizer = cv2.FaceRecognizerSF.create(
            model_path, config="", backend_id=0, target_id=0
        )

    def recognizer_(self, img_face):
        face_locations = self.recognizer.feature(img_face)
        face_locations = face_locations / np.linalg.norm(face_locations)
        return face_locations


def align_and_crop_face(
    image,
    right_eye,
    left_eye,
    top_left,
    bottom_right,
    desired_face_width=112,
    desired_face_height=None,
):
    """
    Выравнивает лицо на изображении так, чтобы глаза находились на одной горизонтальной линии,
    и вырезает область лица с учетом поворота, сохраняя пропорции от центра между глаз.

    Аргументы:
        image (np.ndarray): Исходное изображение (BGR).
        left_eye (tuple): Координаты левого глаза (x, y).
        right_eye (tuple): Координаты правого глаза (x, y).
        top_left (tuple): Координаты левого верхнего угла области лица (x, y).
        bottom_right (tuple): Координаты правого нижнего угла области лица (x, y).
        desired_face_width (int): Желаемая ширина вырезанного лица (по умолчанию 256).
        desired_face_height (int): Желаемая высота вырезанного лица. Если None, вычисляется автоматически.

    Возвращает:
        aligned_face (np.ndarray): Выровненное и вырезанное лицо.
    """
    # Если desired_face_height не задан, вычисляем его на основе desired_face_width
    if desired_face_height is None:
        desired_face_height = desired_face_width

    # Вычисляем угол наклона линии между глазами
    dy = right_eye[1] - left_eye[1]
    dx = right_eye[0] - left_eye[0]
    angle = np.degrees(np.arctan2(dy, dx))  # Угол в градусах

    # Вычисляем центр между глазами
    eyes_center = ((left_eye[0] + right_eye[0]) // 2, (left_eye[1] + right_eye[1]) // 2)

    # Получаем матрицу поворота
    rotation_matrix = cv2.getRotationMatrix2D(eyes_center, angle, scale=1.0)
    # rotation_matrix = cv2.getRotationMatrix2D(eyes_center, 30, scale=1.0)

    # Применяем аффинное преобразование (поворот)
    aligned_image = cv2.warpAffine(
        image,
        rotation_matrix,
        (image.shape[1], image.shape[0]),
        flags=cv2.INTER_CUBIC,
        borderMode=cv2.BORDER_REPLICATE,
    )
    # cv2.imwrite("aligned_image.jpeg", aligned_image)

    '''
    # Вычисляем новые координаты углов области лица после поворота
    def rotate_point(point, center, angle_rad):
        """Поворачивает точку вокруг центра на заданный угол."""
        x, y = point
        cx, cy = center
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        x_new = cos_a * (x - cx) - sin_a * (y - cy) + cx
        y_new = sin_a * (x - cx) + cos_a * (y - cy) + cy
        return int(x_new), int(y_new)

    angle_rad = np.radians(angle)
    rotated_top_left = rotate_point(top_left, eyes_center, angle_rad)
    rotated_bottom_right = rotate_point(bottom_right, eyes_center, angle_rad)

    # Вырезаем область лица
    x1, y1 = rotated_top_left
    x2, y2 = rotated_bottom_right
    '''
    x1, y1 = top_left
    x2, y2 = bottom_right

    # Убедимся, что координаты находятся в пределах изображения
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(aligned_image.shape[1], x2)
    y2 = min(aligned_image.shape[0], y2)

    # Вырезаем лицо
    aligned_face = aligned_image[y1:y2, x1:x2]
    # cv2.imwrite("aligned_face.jpeg", aligned_face)
    # Если вырезанная область меньше желаемого размера, добавляем черные границы
    if (
        aligned_face.shape[0] < desired_face_height
        or aligned_face.shape[1] < desired_face_width
    ):
        delta_w = desired_face_width - aligned_face.shape[1]
        delta_h = desired_face_height - aligned_face.shape[0]
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)
        aligned_face = cv2.copyMakeBorder(
            aligned_face, top, bottom, left, right, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )

    # Масштабируем вырезанное лицо до желаемого размера
    aligned_face = cv2.resize(aligned_face, (desired_face_width, desired_face_height))

    return aligned_face
