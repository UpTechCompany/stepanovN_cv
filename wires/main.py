from skimage.measure import label
from skimage.morphology import binary_closing, binary_dilation, binary_opening, binary_erosion
import matplotlib.pyplot as plt
import numpy as np
import os

for img_path in os.listdir("wires_image"):
    print("Следующая картинка: ")
    image = np.load("wires_images/" + img_path)
    # plt.imshow(image)
    # plt.show()

    res = label(image)

    for i in np.unique(res):
        if i != 0:
            trans = res == i
            trans = binary_erosion(trans)
            # plt.imshow(trans)
            # plt.show()
            l = len(np.unique(label(trans))) - 1

            match l:
                case 1:
                    print(i, "Провод целый")
                case 0:
                    print(i, "Провод АННИГИЛИРОВАН")
                case _:
                    print(i, l)



