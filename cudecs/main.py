import cv2
import numpy as np
import zmq


def count_objects(image):
    circle = 0
    quadro = 0
    cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)
    image = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
    image = image[:, :, 1]
    image = cv2.threshold(image, 55, 255, cv2.THRESH_BINARY)[1]
    image = cv2.erode(image, None, iterations=5)
    image = cv2.dilate(image, None, iterations=3)

    counters = cv2.findContours(image, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

    for i in counters:
        (x, y), r = cv2.minEnclosingCircle(i)
        if (3.1415 * r ** 2) > cv2.contourArea(i) * 1.2:
            circle += 1
        else:
            quadro += 1
        print((3.1415 * r ** 2), cv2.contourArea(i))


    cv2.waitKey(ord("w"))
    cv2.putText(image, f"Quadro = {quadro}, Circle = {circle}", (30,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (127, 255, 255))
    cv2.imshow("Mask", image)

#
# # Пример использования
# count_objects('out.png')


cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
port = 5055
socket.connect(f"tcp://192.168.0.113:{port}")
n = 0
while True:
    bts = socket.recv()
    n += 1
    arr = np.frombuffer(bts, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    key = cv2.waitKey(10)
    count_objects(image)
    if key == ord("q"):
        break
    cv2.putText(image, f"Image = {n}", (30,30), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (127, 255, 255))
    cv2.imshow("Image", image)

cv2.destroyAllWindows()

