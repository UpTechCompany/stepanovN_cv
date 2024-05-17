import cv2
import numpy as np
camera = cv2.VideoCapture(0)

cv2.namedWindow("Image")

colors = {
    "yellow": [25, 30],
    "red": [178, 190],
    "green": [60, 70]
}

position = []

def on_mouse_click(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        global position
        position = [y, x]
        print(f"clicked at {position}")


cv2.setMouseCallback("Image", on_mouse_click)


while camera.isOpened():
    _, image = camera.read()
    if position:
        bgr = image[position[0], position[1]]
        hsv = cv2.cvtColor(np.uint8([[bgr]]), cv2.COLOR_BGR2HSV)
        cv2.circle(image, (position[1], position[0]), 10,
                   (255, 0, 0))

        cv2.putText(image, f"BGR = {bgr}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(map(int, bgr)))

        cv2.putText(image, f"HSV = {hsv}", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, tuple(map(int, bgr)))
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("Image", image)

camera.release()
cv2.destroyAllWindows()