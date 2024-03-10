from skimage.measure import label
from skimage.morphology import binary_closing, binary_dilation, binary_opening, binary_erosion
import matplotlib.pyplot as plt
import numpy as np

image = np.load("/Users/nikitastepanov/Downloads/wires1npy.txt")

plt.imshow(image)
plt.show()