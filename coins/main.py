from skimage.measure import label
from skimage.morphology import binary_closing, binary_dilation, binary_opening, binary_erosion
import matplotlib.pyplot as plt
import numpy as np
import os


def area(LB, label=1):
    return np.sum(LB == label)


def uinq_areas(LB):
    arr = {}
    for i in np.unique(labeled):
        if i != 0:
            ar = area(labeled == i)
            if ar not in arr:
                arr[ar] = 0
    return arr


image = np.load("coins.npy.txt")
labeled = label(image)
result = uinq_areas(labeled)
arr = [1, 2, 5, 10]

for i in np.unique(labeled):
    if i != 0:
        result[area(labeled == i)] += 1

result = dict(sorted(result.items()))

sum = 0
for i, area in enumerate(result):
    sum += result[area] * arr[i]

print(sum)
plt.show()