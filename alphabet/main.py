import numpy as np
from scipy.signal import convolve2d


def skeletonize(image):
    # Перевод изображения в бинарный формат (0 - фон, 1 - объект)
    image = (image > 0).astype(np.uint8)

    # Создание структурного элемента для скелетонизации
    skeleton_element = np.array([[0, 0, 0],
                                 [1, 1, 1],
                                 [0, 1, 0]], dtype=np.uint8)

    # Инициализация пустого скелета
    skeleton = np.zeros_like(image)

    # Процесс скелетонизации
    while True:
        # Шаг 1: Получение обводки (неотсортированных пикселей)
        border_pixels = np.zeros_like(image)
        border_pixels[1:-1, 1:-1] = np.logical_and.reduce([
            image[1:-1, 1:-1] == 1,  # Пиксели объекта
            np.sum(image[0:-2, 0:-2], axis=(0, 1)) > 0,
            np.sum(image[0:-2, 1:-1], axis=(0, 1)) > 0,
            np.sum(image[0:-2, 2:], axis=(0, 1)) > 0,
            np.sum(image[1:-1, 0:-2], axis=(0, 1)) > 0,
            np.sum(image[1:-1, 2:], axis=(0, 1)) > 0,
            np.sum(image[2:, 0:-2], axis=(0, 1)) > 0,
            np.sum(image[2:, 1:-1], axis=(0, 1)) > 0,
            np.sum(image[2:, 2:], axis=(0, 1)) > 0
        ])

        # Шаг 2: Удаление внутренних пикселей (кроме границы)
        border_image = convolve2d(image, skeleton_element, mode='same', boundary='fill')
        border_image = (border_image > 0).astype(np.uint8)
        border_image = border_image * image  # Пиксели, которые были нулевыми до операции свертки, останутся нулевыми
        border_image[1:-1, 1:-1] = 0  # Обнуление внутренних пикселей

        # Шаг 3: Добавление скелета
        skeleton += border_pixels

        # Шаг 4: Проверка на окончание
        if np.sum(border_image) == 0:
            break

        # Обновление изображения для следующей итерации
        image = border_image

    return skeleton


# Пример использования
image = np.array([[0, 0, 0, 0, 0, 0, 0],
                  [0, 1, 1, 1, 1, 1, 0],
                  [0, 0, 0, 0, 0, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 1, 1, 1, 0, 0],
                  [0, 0, 0, 0, 0, 0, 0]], dtype=np.uint8)

skeleton = skeletonize(image)
print(skeleton)
