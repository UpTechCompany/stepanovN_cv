import numpy as np
#from scipy.
import matplotlib.pyplot as plt

def translate(image, vector):
    translated = np.zeros_like(image)

    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            ni = i - vector[0]
            nj = j - vector[1]
            if ni < 0 or nj < 0:
                continue
            if ni > image.shape[0] or nj > image.shape[1]:
                continue
            translated[ni, nj] = image[i, j]

    return translated

struct = np.ones((3, 3))

def dilation(arr):
    result = np.zeros_like(arr)
    for y in range(1, arr.shape[0] - 1):
        for x in range(1, arr.shape[1] - 1):
            if arr[y, x] == 1:
                result[y - 1: y + 2, x - 1: x + 2] = struct

    return result

def erosion(arr):
    result = np.zeros_like(arr)
    for y in range(1, arr.shape[0] - 1):
        for x in range(1, arr.shape[1] - 1):
            if np.all(arr[y - 1: y + 2, x - 1: x + 2] == struct):
                result[y, x] = 1

    return result

def closing(arr):
    return erosion(dilation(arr))

def opening(arr):
    return dilation(erosion(arr))

#image = face(True)

image = arr = np.array([[0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,1,1,1,1,1,1,1,0,0],
                [0,0,0,0,1,1,1,1,0,0],
                [0,0,0,0,1,1,1,1,0,0],
                [0,0,0,1,1,1,1,1,0,0],
                [0,0,0,0,1,1,1,1,0,0],
                [0,0,0,1,1,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0],
                [0,0,0,0,0,0,0,0,0,0]])

result = opening(image)

plt.imshow(result)
plt.show()