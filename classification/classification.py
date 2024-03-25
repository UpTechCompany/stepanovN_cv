# import matplotlib.pyplot as plt
# import numpy as np
# from skimage import draw
# from skimage.filters import threshold_otsu
# def hist(arr):
#     result = np.zeros(256)
#
#     for y in range(arr.shape[0]):
#         for x in range(arr.shape[1]):
#             result[arr[y, x]] += 1
#
#     return result
#
# image = plt.imread("/Users/rayhil/Desktop/University/1 course/Python/CV/cv/gradient/coins.jpg")
# image = np.mean(image, 2).astype("uint8")
#
# h = hist(image)
#
# thresh = threshold_otsu(image) * .9
#
# image[image > thresh] = 0
# image[image > 0] = 1
#
# plt.subplot(121)
# plt.imshow(image)
# plt.subplot(122)
# plt.plot(h)
# plt.show()

import matplotlib.pyplot as plt
import numpy as np
from skimage import draw
from skimage.measure import label
from skimage.filters import threshold_otsu

image = plt.imread("/Users/nikitastepanov/PycharmProjects/stepanovN_cv/alphabet.png")
image[image > 0] = 1
print(len(np.unique(label(image))) - 1)

# plt.subplot(121)
# plt.imshow(image)
# plt.subplot(122)
# plt.imshow(image2)
# plt.show()