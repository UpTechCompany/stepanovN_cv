import cv2
import numpy as np
import zmq

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)
context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")
port = 5055
socket.connect(f"tcp://192.168.0.113:{port}")

lower = 100
upper = 200

# Функции обновления пороговых значений
def lower_update(value):
    global lower
    lower = value

def upper_update(value):
    global upper
    upper = value

cv2.createTrackbar("Lower", "Mask", lower, 255, lower_update)
cv2.createTrackbar("Upper", "Mask", upper, 255, upper_update)

# Функция для добавления текста на изображение с учетом перспективы
def add_text_perspective(image, text, contour):
    # Определение прямоугольника, описывающего контур
    rect = cv2.minAreaRect(contour)
    box = np.int0(cv2.boxPoints(rect))

    # Определение центра контура
    cx = int(rect[0][0])
    cy = int(rect[0][1])

    # Расчет угла наклона контура
    angle = rect[2]

    # Преобразование матрицы перспективы
    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))

    # Добавление текста на изображение с учетом перспективы
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 255)
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int(cx - text_size[0] / 2)
    text_y = int(cy + text_size[1] / 2)
    cv2.putText(rotated, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    rotated = cv2.warpAffine(rotated, M, (-cols, -rows))

    return rotated

while True:
    bts = socket.recv()
    arr = np.frombuffer(bts, np.uint8)
    image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    mask = cv2.Canny(image, lower, upper)
    cnts, _ = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(image, cnts, -1, (0, 0, 255), 5)

    image1 = add_text_perspective(cnts[1], "Hello world", cnts[1])

    cv2.imshow("Image", image)
    cv2.imshow("Mask", mask)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
