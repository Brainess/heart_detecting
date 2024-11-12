import sys
import cv2
import os
import numpy as np
np.set_printoptions(threshold=sys.maxsize)


# функция для просмотра изображения
def viewImage(imagi, name_of_window):
    cv2.namedWindow(name_of_window, cv2.WINDOW_NORMAL)
    cv2.imshow(name_of_window, imagi)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


# переходим в рабочую директорию
os.chdir(f'C:/Users/ifade/Desktop/Image/halogen')

# получаем массив изображений в папке
images = os.listdir()

# парсим картинки
for image in images:
    # получаем изображение в виде оттенков серого
    im = cv2.imread(image, cv2.IMREAD_GRAYSCALE)
    # выставляем порог по интенсивности
    retval, im_porog = cv2.threshold(im, 100, 170, 1)
    viewImage(im_porog, 'gray with porog')
    # Поиск контуров на изображении
    contours, hierarchy = cv2.findContours(im_porog, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    for contour in contours:
        shape = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True)
        x_cor = shape.ravel()[0]
        y_cor = shape.ravel()[1] - 15

        if len(shape) > 12:
            cv2.drawContours(im, [shape], -1, (0, 255, 0), 3)
            cv2.putText(im, "Prisoska", (x_cor, y_cor), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0, 0, 255))
    cv2.imwrite(f'C:/Users/ifade/Desktop/Image/halogen/test.png', im)
    viewImage(im, 'With prisoska')
    break
    #cv2.waitKey(0)
