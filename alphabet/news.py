import cv2
import matplotlib.pyplot as plt
import numpy as np

bg = cv2.imread("bb.jpg")

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Camera not Working!")
# fg = cv2.imread("cheburashka.jpg")
_, fg = camera.read()

rows, cols, _ = fg.shape

pts_fg = np.float32([[0, 0], [0, rows], [cols, 0], [cols, rows]])
pts_bg = np.float32([[490, 146], [496, 303], [802, 136], [808, 292]])

M = cv2.getPerspectiveTransform(pts_fg, pts_bg)

while True:
    _, fg = camera.read()

    persp = cv2.warpPerspective(fg, M, (bg.shape[1], bg.shape[0]))

    pos = np.where(persp > 0)
    bg[pos] = persp[pos]

    cv2.imshow("Image", bg)

    key = cv2.waitKey(10)
    if key == ord('g'):
        break

camera.release()
cv2.destroyAllWindows()