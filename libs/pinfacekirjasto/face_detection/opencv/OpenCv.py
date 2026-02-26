import cv2

# Загрузка каскадного классификатора для обнаружения лиц и глаз
from os.path import normpath, join, isfile

cascadecv2 = 'haarcascade_frontalface_default.xml'
for home in ['pinfacekirjasto/face_detection/opencv/', 
              cv2.data.haarcascades]:
    target_file = normpath(join(home, cascadecv2))
    if isfile(target_file):
        break
    target_file = ''
if target_file == '':
    print('Not found ', cascadecv2, ' - stop')
    exit()

face_cascadecv2 = cv2.CascadeClassifier(target_file)

cascadecv2 = "haarcascade_eye.xml"
for home in ['pinfacekirjasto/face_detection/opencv/', 
              cv2.data.haarcascades]:
    target_file = normpath(join(home, cascadecv2))
    if isfile(target_file):
        break
    target_file = ''
if target_file == '':
    print('Not found ', cascadecv2, ' - stop')
    exit()
eye_cascadecv2 = cv2.CascadeClassifier(target_file)


def detectMultiScale3(frame, **ar):
    """
    Обнаружение лиц и глаз на изображении.

    :param frame: Входное изображение (кадр).
    :param ar: Дополнительные аргументы (не используются в текущей реализации).
    :return: Кортеж (faces, rejectLevels, levelWeights), где:
             - faces: Список прямоугольников, описывающих обнаруженные лица.
             - rejectLevels: Уровни уверенности для каждого обнаруженного лица.
             - levelWeights: Веса для каждого уровня уверенности.
    """
    # Преобразуем изображение в градации серого
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Обнаружение лиц с использованием detectMultiScale3
    faces, rejectLevels, levelWeights = face_cascadecv2.detectMultiScale3(
        gray,
        scaleFactor=1.3,
        minNeighbors=8,
        minSize=(50, 50),
        outputRejectLevels=True
    )

    # Если лица не обнаружены, возвращаем пустые списки
    if len(faces) == 0:
        return [], [], []


    return faces, rejectLevels, levelWeights


    # Отображение результатов
    for (x, y, w, h), level, weight in zip(faces, rejectLevels, levelWeights):
        #print(f"Обнаружено лицо: ({x}, {y}, {w}, {h}), уровень уверенности: {level}, вес: {weight}")

        # Область лица для поиска глаз
        roi_color = frame[y:y + h, x:x + w]
        roi_color = cv2.resize(roi_color, (112, 112), interpolation=cv2.INTER_AREA)
        roi_gray = cv2.cvtColor(roi_color, cv2.COLOR_BGR2GRAY)  # Преобразуем в градации серого

        # Обнаружение глаз в области лица
        eyes = eye_cascadecv2.detectMultiScale(
            roi_gray,
            scaleFactor=1.1,
            minNeighbors=10,
            # minSize=(20, 20)  # Закомментировано для гибкости
        )

        left_eye = None
        right_eye = None

        # Фильтрация ложных срабатываний
        valid_eyes = []
        for (ex, ey, ew, eh) in eyes:
            # Проверяем, что глаз находится в верхней половине лица
            if ey < h // 2 - 20:
                if ey + eh / 2 < h // 2 - 20:
                    valid_eyes.append((ex, ey, ew, eh))
        eyes = valid_eyes

        # Модуль обнаружения глаз в OpenCV не очень надежный. Он может найти больше двух глаз!
        # Кроме того, он возвращает глаза в разном порядке при каждом вызове (проблема 435).
        # Это важная проблема, потому что OpenCV используется по умолчанию, и SSD также использует этот метод.
        # Найдем два самых больших глаза. Спасибо @thelostpeace.

        # Сортируем глаза по площади (ширина * высота)
        eyes = sorted(eyes, key=lambda v: abs(v[2] * v[3]), reverse=True)

        # Если найдено хотя бы два глаза, определяем левый и правый
        if len(eyes) >= 2:
            eye_1 = eyes[0]
            eye_2 = eyes[1]

            # Определяем, какой глаз левый, а какой правый
            if eye_1[0] < eye_2[0]:
                right_eye = eye_1
                left_eye = eye_2
            else:
                right_eye = eye_2
                left_eye = eye_1

            # Находим центр глаз
            left_eye = (
                int(left_eye[0] + (left_eye[2] / 2)),
                int(left_eye[1] + (left_eye[3] / 2)),
            )
            right_eye = (
                int(right_eye[0] + (right_eye[2] / 2)),
                int(right_eye[1] + (right_eye[3] / 2)),
            )

        # Отображение глаз
        '''
        for (ex, ey, ew, eh) in eyes:
            cv2.rectangle(roi_color, (ex, ey), (ex + ew, ey + eh), (255, 0, 0), 2)
            print(f"Обнаружен глаз: ({x + ex}, {y + ey}, {ew}, {eh})")
        '''

	
        #if left_eye is not None:
        #    cv2.rectangle(roi_color, (left_eye[0], left_eye[1]), (right_eye[0], right_eye[1]), (255, 0, 0), 2)

        # Показываем область лица с глазами
        #cv2.imshow("Detected Faces and Eyes", roi_color)
	

    return faces, rejectLevels, levelWeights


'''
# Отображение результатов
for (x, y, w, h), level, weight in zip(faces, rejectLevels, levelWeights):
    print(f"Обнаружено лицо: ({x}, {y}, {w}, {h}), уровень уверенности: {level}, вес: {weight}")
    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
'''
