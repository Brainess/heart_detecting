import shutil
import sys
import cv2
import os
import numpy as np

np.set_printoptions(threshold=sys.maxsize)


# функция для просмотра изображения
def viewImage(image_for_view, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, image_for_view)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# функция для определения УФ/синего в паре кадров
def UF_or_blue(frame_1, frame_2):
    if np.sum(frame_2) / np.sum(frame_1) > 1:
        frame_uv = frame_2
        frame_blue = frame_1
        frame_1_UV = False
    else:
        frame_uv = frame_1
        frame_blue = frame_2
        frame_1_UV = True
    return frame_uv, frame_blue, frame_1_UV


def soska_detection(image_for_detection):
    # выделение порога
    retval, image_with_porog = cv2.threshold(image_for_detection, 254, 255, 3)
    # viewImage(image_with_porog, 'thresh')

    # параметры точности поиска кругов
    a = 100
    b = 100
    # максимально и минимально возможные радиусы искомого круга
    max_radius = 0
    min_radius = 0

    circles = cv2.HoughCircles(cv2.medianBlur(image_with_porog, 5), cv2.HOUGH_GRADIENT, 1, 1,
                               param1=a, param2=b, minRadius=min_radius, maxRadius=max_radius)
    # подбираем максимально возможные параметры точности
    while circles is None:
        circles = cv2.HoughCircles(cv2.medianBlur(image_with_porog, 5), cv2.HOUGH_GRADIENT, 1, 1,
                                   param1=a, param2=b, minRadius=min_radius, maxRadius=max_radius)
        a -= 1
        b -= 1
    # если нашли круги, то нам нужен только с максимальным радиусом
    if circles is not None:
        circles = np.uint16(np.around(circles))
        # Находим круг с максимальным радиусом
        largest_circle = max(circles[0, :], key=lambda x: x[2])
        # записываем его координаты и радиус
        center_of_largest_circle = (largest_circle[0], largest_circle[1])
        radius_of_largest_circle = largest_circle[2]
        # рисование круга на картинке
        # cv2.circle(image_for_detection, center_of_largest_circle, radius_of_largest_circle, (0, 0, 255), 3)
        # viewImage(image_for_detection, 'with circle')
        return center_of_largest_circle, radius_of_largest_circle
    else:
        print('Круг найти не удалось')


# определение контуров(в этом случае, кругов)
def detection_contours(image_for_contours):
    im_crop = image_for_contours
    # выставляем порог по интенсивности
    retval, im_porog = cv2.threshold(im_crop, 250, 255, 1)
    viewImage(im_porog, 'gray with porog')
    # Поиск контуров на изображении
    contours, hierarchy = cv2.findContours(im_porog, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        shape = cv2.approxPolyDP(contour, 0.0001 * cv2.arcLength(contour, True), True)
        x_cor = shape.ravel()[0]
        y_cor = shape.ravel()[1] - 15

        if len(shape) > 30:
            cv2.drawContours(im_crop, [shape], -1, (0, 255, 0), 3)
            cv2.putText(im_crop, "Prisoska", (x_cor, y_cor), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))


# функция для расчёта усреднённого ratio-сигнала внутри присоски
def ratio_of_mean_channels(images_directory):
    # переходим в мышиную папку
    os.chdir(images_directory)

    # получаем список кадров, полученных от мыши
    images = os.listdir()

    # создадим директории обработки
    os.makedirs('Blue', exist_ok=True)  # exist_ok=True предотвращает ошибку, если папка уже существует
    blue_dir = fr'{images_directory}\Blue'
    os.makedirs('UV', exist_ok=True)
    uv_dir = fr'{images_directory}\UV'
    os.makedirs('Mean_Ratio', exist_ok=True)
    ratio_dir = fr'{images_directory}\Mean_Ratio'
    os.makedirs('Raw', exist_ok=True)
    raw_dir = fr'{images_directory}\Raw'

    for image in images:
        shutil.move(image, raw_dir)

    # перейдём в папку с кадрами
    os.chdir(raw_dir)

    # найдём первый кадр в УФ (frame_1_UV_check проверяет, что первый кадр в массиве является УФ)
    frame_UV, frame_BLue, frame_1_UV_check = UF_or_blue(cv2.imread(images[0], cv2.IMREAD_GRAYSCALE),
                                                        cv2.imread(images[1], cv2.IMREAD_GRAYSCALE))

    # сделаем срезы кадров УФ и синего
    if frame_1_UV_check:
        slice_UV = images[::2]
        slice_blue = images[1::2]
    else:
        slice_UV = images[1::2]
        slice_blue = images[::2]

    # получим центр и радиус отверстия присоски
    center, radius = soska_detection(frame_UV)  # первая итерация
    # # cv2.circle(frame_UV, center, radius, (0, 0, 255), 3)
    # # viewImage(frame_UV, 'with circle')
    # n = 1.5   # множитель для радиуса
    # frame_UV = frame_UV[center1[0] - int(n * radius1):center1[0] + int(n * radius1),
    #                     center1[1] - int(n * radius1):center1[1] + int(n * radius1)]
    # center, radius = soska_detection(frame_UV)  # вторая итерация

    n = 0.5  # множитель радиуса

    for i in range(0, int(len(images) / 2), 5):
        images_uv = []
        images_blue = []

        # берём m-е усреднение
        for m in range(i, i + 5):
            #   UV
            shutil.copy(slice_UV[m], uv_dir)
            images_uv.append(cv2.imread(slice_UV[m], cv2.IMREAD_GRAYSCALE)
                             [center[0] - int(n * radius):center[0] + int(n * radius),
                             center[1] - int(n * radius):center[1] + int(n * radius)])
            #   Blue
            shutil.copy(slice_blue[m], blue_dir)
            images_blue.append(cv2.imread(slice_blue[m], cv2.IMREAD_GRAYSCALE)
                               [center[0] - int(n * radius):center[0] + int(n * radius),
                               center[1] - int(n * radius):center[1] + int(n * radius)])

        mean_uv = np.mean(images_uv, axis=0).astype(np.float32)
        mean_blue = np.mean(images_blue, axis=0).astype(np.float32)

        frame_ratio = np.clip(mean_blue / mean_uv).astype(np.float32)

        os.chdir(ratio_dir)
        try:
            check_write = cv2.imwrite(f'{i + 5}.tif', frame_ratio)
        except:
            print('Проблемы с записью:')
            raise
        os.chdir(raw_dir)


# определяем, где находится эпизод
episode_directory = os.getcwd()

# получаем список файлов в эпизоде
mouses = os.listdir()

mouse_directories = []

# создаём список мышиных папок
for mouse in mouses:
    full_path = os.path.join(episode_directory, mouse)
    if os.path.isdir(full_path):
        mouse_directories.append(mouse)

# парсим все изображения в папках
for mouse_directory in mouse_directories:
    ratio_of_mean_channels(fr'{episode_directory}\{mouse_directory}')
