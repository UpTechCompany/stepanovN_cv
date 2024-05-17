import cv2
import matplotlib.pyplot as plt
import numpy as np
from time import perf_counter
cv2.namedWindow("Image")
cv2.namedWindow("Background")

background = None

camera = cv2.VideoCapture(0)
if not camera.isOpened():
    raise RuntimeError("Camera not Working!")
# fg = cv2.imread("cheburashka.jpg")
_, image = camera.read()

min_area = 1000
prev_time = perf_counter()
prev_gray = None
while True:
    _, image = camera.read()



    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (21, 21), 0)

    if prev_gray is None:
        prev_gray = gray.copy()

    diff_gray = cv2.absdiff(gray, prev_gray)
    diff_thresh = cv2.threshold(diff_gray, 25, 255, cv2.THRESH_BINARY)[1]
    diff_percent = np.sum(diff_thresh) / diff_thresh.size
    pref_gray = gray.copy()
    key = cv2.waitKey(10)
    if key == ord('q'):
        break



    curr_time = perf_counter()
    if key == ord('b') or diff_percent < 0.1:# or (curr_time - prev_time) >= 1:
        background = gray.copy()
        prev_time = curr_time

    if background is not None:
        delta = cv2.absdiff(background, gray)
        thresh = cv2.threshold(delta, 25, 255, cv2.THRESH_BINARY)[1]
        thresh = cv2.dilate(thresh, None, iterations=2)
        contors, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL,
                                   cv2.CHAIN_APPROX_SIMPLE)

        for c in contors:
            area = cv2.contourArea(c)
            if area > min_area:
                (x, y, w, h) = cv2.boundingRect(c)
                cv2.rectangle(image, (x, y), (x + w, y + h),
                              (0, 255, 0), 2)
        cv2.imshow("Background", thresh)
    cv2.imshow("Image", image)


camera.release()
cv2.destroyAllWindows()