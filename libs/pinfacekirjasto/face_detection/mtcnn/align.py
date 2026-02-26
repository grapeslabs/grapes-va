#import sys
#import os

from libs.pinfacekirjasto.face_detection.mtcnn import mtcnn
#import argparse
from PIL import Image
#from tqdm import tqdm
#import random
#from datetime import datetime
#kkk
import numpy as np
import base64
import io
import os
mtcnn_model = mtcnn.MTCNN(device='cuda:0', crop_size=(112, 112))
#mtcnn_model = mtcnn.MTCNN(device='cpu', crop_size=(112, 112))

def add_padding(pil_img, top, right, bottom, left, color=(0,0,0)):
    width, height = pil_img.size
    new_width = width + right + left
    new_height = height + top + bottom
    result = Image.new(pil_img.mode, (new_width, new_height), color)
    result.paste(pil_img, (left, top))
    return result

def get_aligned_face(image_path, rgb_pil_image=None):
    if rgb_pil_image is None:
        img = Image.open(image_path).convert('RGB')
    else:
        assert isinstance(rgb_pil_image, Image.Image), 'Face alignment module requires PIL image or path to the image'
        img = rgb_pil_image
    # find face
    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=1)
        face = faces[0]
	#kkk
        bbox = bboxes[0]
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        face = None
    # kkk
    #return face
    return bbox, face

# kkk

def to_pil_image(input_data):
    """
    Приводит входные данные к объекту PIL.Image.
    Поддерживаемые форматы:
    - имя файла (str)
    - объект PIL.Image
    - массив np.array
    - строка base64
    """
    # Если входные данные — строка
    if isinstance(input_data, str):
        # Проверяем, является ли это именем файла
        if os.path.isfile(input_data):
            # Загружаем изображение из файла
            return Image.open(input_data)
        # Проверяем, является ли это base64
        try:
            # Декодируем base64
            image_bytes = base64.b64decode(input_data)
            # Преобразуем байты в изображение
            return Image.open(io.BytesIO(image_bytes))
        except (base64.binascii.Error, OSError):
            # Если это не base64 и не файл, выбрасываем исключение
            raise ValueError("Строка не является именем файла или корректной base64 строкой.")

    # Если входные данные — объект PIL.Image
    elif isinstance(input_data, Image.Image):
        return input_data

    # Если входные данные — массив np.array
    elif isinstance(input_data, np.ndarray):
   # Проверяем, является ли массив изображением в формате BGR
        if len(input_data.shape) == 3 and input_data.shape[2] == 3:  # 3 канала (BGR)
            # Преобразуем BGR в RGB
            input_data = input_data[:, :, ::-1]  # Инвертируем порядок каналов
        # Преобразуем массив в изображение
        return Image.fromarray(input_data)
    # Если входные данные — байты (например, из io.BytesIO)

    elif isinstance(input_data, bytes):
        try:
            # Пытаемся загрузить изображение из байтов
            return Image.open(io.BytesIO(input_data))
        except OSError:
            raise ValueError("Байты не содержат корректное изображение.")

    # Если тип не поддерживается
    else:
        raise TypeError("Неподдерживаемый тип данных. Ожидается: имя файла, PIL.Image, np.array или base64 строка.")


def get_aligned_faces(my_image, limit=None):

    img = to_pil_image(my_image)

    try:
        bboxes, faces = mtcnn_model.align_multi(img, limit=None)
    except Exception as e:
        print('Face detection Failed due to error.')
        print(e)
        faces = []
        bboxes = []
    return bboxes, faces
