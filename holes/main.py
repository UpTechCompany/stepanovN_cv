import numpy as np
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from collections import defaultdict
crooss_masks = np.array([
    [[0,0], [0,1]],
    [[0,1], [1,1]],])

def match(a, mask):
    if np.all(a == mask):
        return True
    return False

def add_frame(array):
    rows, cols = array.shape
    new_array = np.zeros((rows + 2, cols + 2), dtype=array.dtype)
    new_array[1:-1, 1:-1] = array
    return new_array

def eleur(B, masks):
    X = 0
    V = 0
    for y in range(0, B.shape[0]-1):
        for x in range(0, B.shape[1]-1):
            sub = B[y:y+2, x:x+2]

            if match(sub, masks[0]):
                X += 1
            if match(sub, masks[1]):
                V += 1
    return X - V

image = np.load('/content/holes.npy')
regions = regionprops(label(image))
result = defaultdict(lambda:0)

for i in regions:
    if i != 0:
        # print(eleur(add_frame(i.image), crooss_masks))
        # plt.subplot(121)
        # plt.imshow(i.image)
        # plt.show()
        result[str(eleur(add_frame(i.image), crooss_masks))] += 1

print("Два отверстия:", result["-1"], "\n", "Одно отверстие:", result["0"], "\n", "Нет отверстий:", result["1"])