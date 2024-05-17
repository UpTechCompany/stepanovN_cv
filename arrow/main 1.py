import cv2
import numpy as np
import matplotlib.pyplot as plt
camera = cv2.VideoCapture(0)

# cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
while camera.isOpened():
    _, image = camera.read()
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = image[:, :, 1]
    image = cv2.threshold(image, 80, 255, cv2.THRESH_BINARY_INV)[1]
    image = cv2.erode(image, None, iterations=10)
    image = cv2.dilate(image, None, iterations=25)

    cnts = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                    cv2.CHAIN_APPROX_SIMPLE)[0]
    mx = 0
    c = max(cnts, key = cv2.contourArea)
    print(c)

    cv2.drawContours(image, [c], 0, (255, 0, 0), 2)

    ellipse = cv2.fitEllipse(c)

    cv2.ellipse(image, ellipse, (0, 255, 0), 5)

    cont = cv2.findContours(image, cv2.RETR_EXTERNAL,
                            cv2.CHAIN_APPROX_SIMPLE)[0]

    print(len(cont) - 2)

    cv2.imshow("Image", image)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("Image", image)

camera.release()
cv2.destroyAllWindows()