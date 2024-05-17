import cv2
import matplotlib.pyplot as plt
import numpy as np

cv2.namedWindow("Image")
cv2.namedWindow("Background")

roi = None

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Camera not Working!")
# fg = cv2.imread("cheburashka.jpg")
_, image = camera.read()

while True:
    _, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    key = cv2.waitKey(10)
    if key == ord('q'):
        break

    if key == ord('f'):
        x, y, w, h = cv2.selectROI("ROISelection", gray)
        roi = gray[y: y + h, x: x + w]
        cv2.imshow("POI", roi)
        cv2.destroyWindow("ROISelection")
    cv2.imshow("Image", image)


camera.release()
cv2.destroyAllWindows()