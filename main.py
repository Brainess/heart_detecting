import re
import shutil
import sys
import os

import numpy as np
from scipy import stats
from scipy.signal import medfilt
import cv2

import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import inquirer
from tqdm import tqdm

import win32file
import win32con
import pywintypes
import datetime

np.set_printoptions(threshold=sys.maxsize)


def get_creation_time_windows_local(filepath):
    """Получает время создания файла на Windows в локальном часовом поясе.

       Args:
        filepath: Путь к файлу.

       Returns:
        Время создания файла в локальном часовом поясе как объект datetime,
        или None, если произошла ошибка.
    """
    try:
        file_handle = win32file.CreateFile(
            filepath,
            win32con.GENERIC_READ,
            win32con.FILE_SHARE_READ,
            None,
            win32con.OPEN_EXISTING,
            win32con.FILE_ATTRIBUTE_NORMAL,
            None
        )

        creation_time_filetime = win32file.GetFileTime(file_handle)[0]  # Получаем FILETIME структуру
        win32file.CloseHandle(file_handle)

        # Преобразование FILETIME в datetime
        creation_time_seconds = creation_time_filetime.timestamp()  # timestamp() преобразует в секунды

        timestamp_utc = datetime.datetime.utcfromtimestamp(creation_time_seconds)

        # Преобразование из UTC в локальное время
        local_tz = datetime.datetime.now(datetime.timezone.utc).astimezone().tzinfo
        local_creation_time = timestamp_utc.replace(tzinfo=datetime.timezone.utc).astimezone(local_tz)

        return local_creation_time.strftime("%H:%M:%S:%f")

    except FileNotFoundError:
        print(f"Файл не найден: {filepath}")
        return None
    except pywintypes.error as e:
        print(f"Ошибка при получении времени создания: {e}")
        return None
    except Exception as e:  # Добавили общий обработчик исключений
        print(f"Произошла непредвиденная ошибка: {e}")
        return None


# функция для просмотра изображения
def viewImage(image_for_view, name_of_window):
    height, width = image_for_view.shape[:2]
    new_width = 640
    new_height = int(height * (new_width / width))
    resized_img = cv2.resize(image_for_view, (new_width, new_height))
    cv2.namedWindow(name_of_window, cv2.WINDOW_AUTOSIZE)
    cv2.imshow(name_of_window, resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# функция для определения УФ/синего в паре кадров
def UV_or_blue(frame_1, frame_2):
    if np.sum(frame_2) / np.sum(frame_1) > 1:
        # frame_uv = frame_2
        # frame_blue = frame_1
        frame_1_UV = False
    else:
        # frame_uv = frame_1
        # frame_blue = frame_2
        frame_1_UV = True
    return frame_1_UV


def soska_detection(image_for_detection, binning: int = 1):
    # выделение порога
    retval, image_with_porog = cv2.threshold(image_for_detection, 254, 255, 3)
    # viewImage(image_with_porog, 'thresh')

    # параметры точности поиска кругов
    a = 100
    b = 100
    # максимально и минимально возможные радиусы искомого круга
    height, width = image_for_detection.shape[:2]
    max_radius = int(max(height, width) * 0.25)
    min_radius = int(min(height, width) * 0.24)

    circles = cv2.HoughCircles(cv2.medianBlur(image_with_porog, 5), cv2.HOUGH_GRADIENT, 2, 100,
                               param1=a, param2=b, minRadius=min_radius, maxRadius=max_radius)
    # подбираем максимально возможные параметры точности
    while circles is None:
        circles = cv2.HoughCircles(cv2.medianBlur(image_with_porog, 5), cv2.HOUGH_GRADIENT, 2, 100,
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
        # img_copy = image_for_detection.copy()
        # cv2.circle(img_copy, center_of_largest_circle, radius_of_largest_circle, (0, 0, 255), 3)
        # viewImage(img_copy, 'with circle')
        return center_of_largest_circle, radius_of_largest_circle
    else:
        print('Круг найти не удалось')


# функция для записи в txt-файл
def write_to_txt(string: str, file_directory: str = os.getcwd(), name: str = 'noname'):
    with open(fr'{file_directory}\{name}.txt', 'w', encoding="UTF-8") as file_txt:
        file_txt.write(f'{string}\n')


# функция для получения цвета графика по длине волны
def wavelength_to_rgb(wavelength):
    """Converts a wavelength in nanometers to an approximate RGB color."""
    wavelength = float(wavelength)
    if 380 <= wavelength <= 440:
        R = -(wavelength - 440) / (440 - 380)
        G = 0.0
        B = 1.0
    elif 440 <= wavelength <= 490:
        R = 0.0
        G = (wavelength - 440) / (490 - 440)
        B = 1.0
    elif 490 <= wavelength <= 510:
        R = 0.0
        G = 1.0
        B = -(wavelength - 510) / (510 - 490)
    elif 510 <= wavelength <= 580:
        R = (wavelength - 510) / (580 - 510)
        G = 1.0
        B = 0.0
    elif 580 <= wavelength <= 645:
        R = 1.0
        G = -(wavelength - 645) / (645 - 580)
        B = 0.0
    elif 645 <= wavelength <= 780:
        R = 1.0
        G = 0.0
        B = 0.0
    else:
        R = 0.0
        G = 0.0
        B = 0.0
    return R, G, B


# функция для построения и сохранения графика
def graph_for_3(x1: list, y1: list, err1: list, x2: list, y2: list, err2: list, x3: list, y3: list,
                graph_dir: os.getcwd(), name_of_graph: str = 'noname_graph'):
    # сглаживание
    # y1_smoothed = medfilt(y1, kernel_size=window_size)  # применение медианного фильтра
    # y2_smoothed = medfilt(y2, kernel_size=window_size)
    # y3_smoothed = medfilt(y3, kernel_size=window_size)

    fig, ax1 = plt.subplots(figsize=(10, 6))

    wavelength_1 = 405
    wavelength_1_str = '{405}'
    R1, G1, B1 = wavelength_to_rgb(wavelength_1)

    wavelength_2 = 470
    wavelength_2_str = '{470}'
    R2, G2, B2 = wavelength_to_rgb(wavelength_2)

    wavelength_3 = 525
    R3, G3, B3 = wavelength_to_rgb(wavelength_3)

    ax1.errorbar(x1, y1, yerr=err1, fmt='o-', ecolor=(R1, G1, B1), elinewidth=0.5, capsize=5, capthick=2,
                 label=f'$F_{wavelength_1_str}$', color=(R1, G1, B1))
    ax1.errorbar(x2, y2, yerr=err2, fmt='o-', ecolor=(R2, G2, B2), elinewidth=0.5, capsize=5, capthick=2,
                 label=f'$F_{wavelength_2_str}$', color=(R2, G2, B2))
    ax1.set_ylabel('Fluorescence signal')
    ax1.set_xlabel('Time, s')
    # ax1.xaxis.set_major_locator(MaxNLocator(5))
    ax1.grid(which='major', axis='y', linestyle='')

    ax2 = ax1.twinx()
    ax2.plot(x3, y3, label=f'Ratio $F_{wavelength_2_str}/F_{wavelength_1_str}$', color=(R3, G3, B3))
    ax2.set_ylabel('Ratio signal')

    fig.suptitle(f'{name_of_graph}')  # Заголовок для всей фигуры
    fig.legend(fontsize=14, loc="upper right", bbox_to_anchor=(1, 1))  # легенда для всей фигуры

    # Настройка общего вида
    plt.tight_layout()
    plt.savefig(os.path.join(graph_dir, f'{name_of_graph}.png'))


# функция для создания директорий
def folders_creation(folder_directory):
    # создадим директории обработки
    blue_dir = fr'{folder_directory}\Blue'
    if os.path.exists(blue_dir):
        shutil.rmtree(blue_dir)
        os.makedirs('Blue', exist_ok=True)  # exist_ok=True предотвращает ошибку, если папка уже существует
    else:
        os.makedirs('Blue', exist_ok=True)

    uv_dir = fr'{folder_directory}\UV'
    if os.path.exists(uv_dir):
        shutil.rmtree(uv_dir)
        os.makedirs('UV', exist_ok=True)
    else:
        os.makedirs('UV', exist_ok=True)

    ratio_dir = fr'{folder_directory}\Mean_Ratio'
    if os.path.exists(ratio_dir):
        shutil.rmtree(ratio_dir)
        os.makedirs('Mean_Ratio', exist_ok=True)
    else:
        os.makedirs('Mean_Ratio', exist_ok=True)

    media_dir = fr'{folder_directory}\Media'
    if os.path.exists(media_dir):
        shutil.rmtree(media_dir)
        os.makedirs('Media', exist_ok=True)
    else:
        os.makedirs('Media', exist_ok=True)

    os.makedirs('Raw', exist_ok=True)
    raw_dir = fr'{folder_directory}\Raw'
    return raw_dir, blue_dir, uv_dir, ratio_dir, media_dir


# определение контуров(в этом случае, кругов)
# def detection_contours(image_for_contours):
#     im_crop = image_for_contours
#     # выставляем порог по интенсивности
#     retval, im_porog = cv2.threshold(im_crop, 250, 255, 1)
#     viewImage(im_porog, 'gray with porog')
#     # Поиск контуров на изображении
#     contours, hierarchy = cv2.findContours(im_porog, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     for contour in contours:
#         shape = cv2.approxPolyDP(contour, 0.0001 * cv2.arcLength(contour, True), True)
#         x_cor = shape.ravel()[0]
#         y_cor = shape.ravel()[1] - 15
#
#         if len(shape) > 30:
#             cv2.drawContours(im_crop, [shape], -1, (0, 255, 0), 3)
#             cv2.putText(im_crop, "Prisoska", (x_cor, y_cor), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))


# функция для расчёта усреднённого ratio-сигнала внутри присоски
def ratio_of_mean_channels(images_directory, averaging: int = 5, fps: int = 10, search_sucker: int = 0):
    # переходим в мышиную папку
    os.chdir(images_directory)

    # получаем содержимое папки мыши
    files = os.listdir()

    # создадим директории обработки
    raw_dir, blue_dir, uv_dir, ratio_dir, media_dir = folders_creation(images_directory)

    for file in files:
        image_path = os.path.join(images_directory, file)
        if not os.path.isdir(image_path):
            shutil.move(file, raw_dir)

    # перейдём в папку с кадрами
    os.chdir(raw_dir)

    # получаем список кадров, полученных от мыши
    images = os.listdir()

    # сортировка
    mouse_number = re.search(r'AM\d+-', images[0]).group()
    numbers = []
    for file in images:
        number = re.search(r'-\d+', file).group()
        number = int(number.replace('-', ''))
        numbers.append(number)
    numbers.sort()
    images = []
    for number in numbers:
        images.append(f'{mouse_number}{"{:03d}".format(number)}.tif')

    # найдём первый кадр в УФ (frame_1_UV_check проверяет, что первый кадр в массиве является УФ)
    image0 = cv2.imread(images[0], cv2.IMREAD_GRAYSCALE)
    image1 = cv2.imread(images[1], cv2.IMREAD_GRAYSCALE)
    frame_1_UV_check = UV_or_blue(image0, image1)
    frame_uv_1 = image0 if frame_1_UV_check else image1

    # сделаем срезы кадров УФ и синего
    if frame_1_UV_check:
        slice_UV = images[::2]
        slice_blue = images[1::2]
    else:
        slice_UV = images[1::2]
        slice_blue = images[::2]

    # получим центр и радиус отверстия присоски
    if search_sucker:
        center, radius = soska_detection(frame_uv_1)  # первая итерация
    else:
        height, width = frame_uv_1.shape[:2]  # Получаем высоту и ширину изображения
        center_x = width // 2  # Целочисленное деление, чтобы получить целое число
        center_y = height // 2
        center = (np.uint16(center_x), np.uint16(center_y))
        radius = min(height, width) / 2

    n = 0.5  # множитель радиуса

    # для отчёта посмотрим, какую область вырезаем
    shutil.copy(slice_UV[0], os.path.join(media_dir, 'prisoska_UV.tif'))
    shutil.copy(slice_blue[0], os.path.join(media_dir, 'prisoska_Blue.tif'))
    os.chdir(media_dir)
    try:
        uv_frame = cv2.imread('prisoska_UV.tif', cv2.IMREAD_GRAYSCALE)
        blue_frame = cv2.imread('prisoska_Blue.tif', cv2.IMREAD_GRAYSCALE)

        if search_sucker:
            cv2.imwrite('prisoska_UV.png', cv2.circle(uv_frame, center, radius, (255, 255, 255), 10))
            cv2.imwrite('prisoska_Blue.png', cv2.circle(blue_frame, center, radius, (255, 255, 255), 10))

        cv2.imwrite(f'crop_UV.png', cv2.rectangle(uv_frame,
                                                  pt1=(center[0] - int(n * radius), center[1] - int(n * radius)),
                                                  pt2=(center[0] + int(n * radius), center[1] + int(n * radius)),
                                                  color=(255, 255, 255), thickness=10))

        cv2.imwrite(f'crop_Blue.png', cv2.rectangle(blue_frame,
                                                    pt1=(center[0] - int(n * radius), center[1] - int(n * radius)),
                                                    pt2=(center[0] + int(n * radius), center[1] + int(n * radius)),
                                                    color=(255, 255, 255), thickness=10))
    except:
        print('Проблемы с записью:')
        raise
    os.chdir(raw_dir)

    # даннные для графиков
    intense_uv = []
    intense_blue = []
    intense_ratio = []
    err_uv = []
    err_blue = []

    timeline_uv = []
    tau = 1 / fps  # длительность кадра(по умолчанию примерно 100 мс)
    t_uv = (tau * 2 * averaging - tau) if frame_1_UV_check else (tau * 2 * averaging)
    timeline_blue = []
    t_blue = (tau * 2 * averaging) if frame_1_UV_check else (tau * 2 * averaging - tau)
    timeline_ratio = []
    t_ratio = (1 / fps) * 2 * averaging  # длительность кадра * 2 канала * количество кадров для усреднения

    # электронный шум(вычисляется для каждого нового режима и экспонирования)
    electronic_noise = 739

    counts = int(len(images) / 2)

    multiplier = 1

    for i in tqdm(range(0, counts, averaging)):
        images_uv = []
        images_blue = []

        if i + averaging > len(slice_UV):
            break
        # берём m-е усреднение
        for m in range(i, i + averaging):
            image_uv = cv2.imread(slice_UV[m], cv2.IMREAD_UNCHANGED) - electronic_noise

            image_blue = cv2.imread(slice_blue[m], cv2.IMREAD_UNCHANGED) - electronic_noise

            # frame_m_UV_check = UV_or_blue(image_uv, image_blue)
            # if not frame_m_UV_check == frame_1_UV_check:  # проверка, что порядок кадров не сбивается
            #     print(f'Я перепутал кадры на {slice_UV[m]}')
            #     exit(1)

            # UV
            shutil.copy(slice_UV[m], uv_dir)
            frame_uv_cropped = image_uv[center[0] - int(n * radius):center[0] + int(n * radius),
                               center[1] - int(n * radius):center[1] + int(n * radius)]
            images_uv.append(frame_uv_cropped)

            # creation_time_uv = f'УФ кадр {get_creation_time_windows_local(slice_UV[m])}'
            # # пишем таймстемп УФ кадра
            # write_to_txt(creation_time_uv, images_directory, 'UV_timestamps')
            #   Blue

            # Blue
            shutil.copy(slice_blue[m], blue_dir)
            frame_blue_cropped = image_blue[center[0] - int(n * radius):center[0] + int(n * radius),
                                 center[1] - int(n * radius):center[1] + int(n * radius)]
            images_blue.append(frame_blue_cropped)

            # creation_time_blue = f'Синий кадр {get_creation_time_windows_local(slice_blue[m])}'
            # # пишем таймстемп синего кадра
            # write_to_txt(creation_time_blue, images_directory, 'Blue_timestamps')

            # # запишем оба таймстемпа в общий файл
            # write_to_txt(creation_time_uv if frame_1_UV_check else creation_time_blue,
            #              images_directory, 'All_timestamps')
            # write_to_txt(creation_time_blue if frame_1_UV_check else creation_time_uv,
            #              images_directory, 'All_timestamps')

        # uv
        mean_uv = np.mean(images_uv, axis=0).astype(np.float32)
        intense_uv.append(np.mean(mean_uv))

        stacked_uv = np.stack(images_uv)
        std_uv = np.std(stacked_uv, axis=0)
        err_uv.append(np.mean(std_uv))

        timeline_uv.append(t_uv)
        t_uv += (1 / fps) * 2 * averaging

        # blue
        mean_blue = np.mean(images_blue, axis=0).astype(np.float32)
        if i == 0:
            multiplier = mean_uv/mean_blue
        intense_blue.append(np.mean(mean_blue*multiplier))

        stacked_blue = np.stack(images_blue)
        std_blue = np.std(stacked_blue, axis=0)
        err_blue.append(np.mean(std_blue))

        timeline_blue.append(t_blue)
        t_blue += (1 / fps) * 2 * averaging

        # ratio
        frame_ratio = (mean_blue / mean_uv).astype(np.float32)
        intense_ratio.append(frame_ratio.mean(axis=0).mean())

        # std_ratio = np.std(frame_ratio.ravel())
        # err_ratio.append(std_ratio)

        timeline_ratio.append(t_ratio)
        t_ratio += (1 / fps) * 2 * averaging

        os.chdir(ratio_dir)
        try:
            cv2.imwrite(f'{i + averaging}.tif', frame_ratio)
        except:
            print('Проблемы с записью:')
            raise
        os.chdir(raw_dir)

    graph_for_3(timeline_uv, intense_uv, err_uv, timeline_blue, intense_blue, err_blue, timeline_ratio,
                intense_ratio, media_dir, 'SypHer3s')


# определяем, где находится эпизод
episode_directory = os.getcwd()

# получаем список файлов в эпизоде
mouses = os.listdir()

mouse_directories = []

# создаём список мышиных папок
for mouse in mouses:
    full_path = os.path.join(episode_directory, mouse)
    if os.path.isdir(full_path) and mouse != 'ffc':
        mouse_directories.append(mouse)

# определяем, по какому количеству кадров будем усреднять
average_count = int(input('Введите количество кадров, по которым происходит усреднение:\n'))

# хочет ли он искать присоску
question = [
    inquirer.List(
        'soska_detect',
        message="Ищем присоску или парсим весь кадр?",
        choices=['Весь кадр', 'Ищем присоску'],
    )
]

soska_detect = inquirer.prompt(question)
match soska_detect.get('soska_detect'):
    case 'Ищем присоску':
        search = 1
    case 'Весь кадр':
        search = 0

# спросим пользователя, хочет он запарсить конкретную папку или все
question = [
    inquirer.List(
        'dir_count',
        message="Сколько папок обрабатываем?",
        choices=['All', 'Select folder'],
    )
]

dir_count = inquirer.prompt(question)
match dir_count.get('dir_count'):
    case 'All':
        # парсим все изображения во всех папках
        for mouse_directory in mouse_directories:
            ratio_of_mean_channels(fr'{episode_directory}\{mouse_directory}', average_count, 10, search)
            print(f'Обработал папку {mouse_directory}')
        print(f'\nОбработка всех папок завершена')
    case 'Select folder':
        question = [
            inquirer.List(
                'dir',
                message="Какую папку хотим обработать?",
                choices=mouse_directories
            )
        ]
        mouse_directory = inquirer.prompt(question).get('dir')
        ratio_of_mean_channels(fr'{episode_directory}\{mouse_directory}', average_count, 10, search)
        print(f'Обработал папку {mouse_directory}')
