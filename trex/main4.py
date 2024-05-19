import cv2 as cv
from mss import mss
import numpy as np
from time import sleep, time
import pyautogui as inp

# Загрузка и подготовка изображения динозавра
dino_img = cv.imread("t-rex.png", cv.IMREAD_GRAYSCALE)
sleep(3)

# Параметры захвата экрана
frame_rate = 60
frame_duration = 1.0 / frame_rate


# Функция для поиска динозавра на экране
def find_dino(img, template):
    result = cv.matchTemplate(img, template, cv.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv.minMaxLoc(result)
    return max_loc


def process_frame(img):
    img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    _, img = cv.threshold(img, 127, 255, cv.THRESH_BINARY_INV)

    enemy = img[:, int(img.shape[1] * 0.3):]
    enemy = enemy[:int(enemy.shape[0] * 0.855), :]

    kernel = cv.getStructuringElement(cv.MORPH_ELLIPSE, (5, 5))
    enemy = cv.morphologyEx(enemy, cv.MORPH_CLOSE, kernel)

    return enemy


def detect_and_act(contours, start, dino_pos):
    for contour in contours:
        x, y, w, h = cv.boundingRect(contour)

        agr = 45
        elapsed_time = time() - start
        if elapsed_time < 50:
            agr = 16
        if elapsed_time > 330:
            elapsed_time = 330

        if x < agr:
            sleeper = (w) * 20 / (1000 + elapsed_time * 40)
            if y + h - dino_pos[1] - 20 > 0:
                if y - dino_pos[1] >= -10:
                    sleeper += 0.11
                    inp.press("up")
                    sleep(sleeper / 4.6)
                    inp.keyDown("down")
                    sleep(0.03)
                    inp.keyUp("down")
            else:
                inp.keyDown("down")
                sleep(abs(sleeper - 0.04))
                inp.keyUp("down")


with mss() as sct:
    monitor = {"top": 315, "left": 400, "width": 600, "height": 43}
    start = time()

    # Определение положения динозавра
    img = np.array(sct.grab(monitor))
    gray_img = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
    dino_pos = find_dino(gray_img, dino_img)

    while True:
        frame_start_time = time()

        try:
            img = np.array(sct.grab(monitor))
            enemy = process_frame(img)

            contours, _ = cv.findContours(enemy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE)
            detect_and_act(contours, start, dino_pos)

            cv.imshow('Image', enemy)

            if cv.waitKey(1) == ord("q"):
                break

            frame_end_time = time()
            elapsed_time = frame_end_time - frame_start_time
            if elapsed_time < frame_duration:
                sleep(frame_duration - elapsed_time)

        except Exception as e:
            print(f"An error occurred: {e}")
            break

cv.destroyAllWindows()
