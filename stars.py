import numpy as np
from scipy import ndimage

# Загрузка бинарного изображения
binary_image = np.loadtxt('/Users/nikitastepanov/Downloads/ps.npy.txt')

# Применение операций морфологического анализа (например, дилатации)
structuring_element = np.array([[1, 1, 1],
                                [1, 1, 1],
                                [1, 1, 1]])  # пример структурирующего элемента
dilated_image = ndimage.binary_dilation(binary_image, structure=structuring_element)

# Подсчёт связных компонент
labeled_array, num_features = ndimage.label(dilated_image)

# Общее количество объектов
total_objects = num_features

# Подсчёт количества объектов для каждого вида
# (в этом месте вам нужно будет предоставить информацию о структурирующих элементах
# и способе классификации объектов по видам)

# Вывод результатов
print("Общее количество объектов:", total_objects)
# Вывод количества объектов для каждого вида
# (ваш код для определения количества объектов каждого вида здесь)
