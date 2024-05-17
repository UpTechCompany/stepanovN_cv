import pyautogui as gui
from pynput.keyboard import Key, Controller
import cv2
import numpy as np
import time
import math

keyboard = Controller()

def get_pixel(image, x, y):
    if 0 <= x < image.shape[1] and 0 <= y < image.shape[0]:
        return image[y, x]
    else:
        return None

def start():
    # Координаты захвата экрана для окна игры на MacBook 14"
    x, y, width, height = 570, 380, 860, 300
    jumping_time = 0
    last_jumping_time = 0
    last_interval_time = 0

    x_start, x_end = 76, 85
    y_search1, y_search2 = 115, 135
    y_search_for_bird = 99

    # Загрузка изображения динозавра
    dino_template = cv2.imread('dino.png', cv2.IMREAD_GRAYSCALE)
    dino_w, dino_h = dino_template.shape[::-1]

    dino_template = cv2.imread('DinoRun1.png', cv2.IMREAD_GRAYSCALE)
    dino_w, dino_h = dino_template.shape[::-1]

    dino_template = cv2.imread('DinoRun2.png', cv2.IMREAD_GRAYSCALE)
    dino_w, dino_h = dino_template.shape[::-1]

    time.sleep(3)
    keyboard.press(Key.up)
    time.sleep(0.1)
    keyboard.release(Key.up)

    while True:
        screenshot = gui.screenshot(region=(x, y, width, height))
        screenshot = cv2.cvtColor(np.array(screenshot), cv2.COLOR_RGB2BGR)
        screenshot_gray = cv2.cvtColor(screenshot, cv2.COLOR_BGR2GRAY)

        res = cv2.matchTemplate(screenshot_gray, dino_template, cv2.TM_CCOEFF_NORMED)
        min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)

        threshold = 0.8  # Уровень совпадения
        if max_val >= threshold:
            dino_x, dino_y = max_loc

            # Область перед динозавром
            for i in range(dino_x + dino_w, width):
                if np.any(screenshot[dino_y + dino_h // 2, i] != get_pixel(screenshot, dino_x, dino_y)):
                    keyboard.press(Key.up)
                    time.sleep(0.1)
                    keyboard.release(Key.up)
                    break

            # Область над динозавром
            for i in range(dino_x + dino_w, width):
                if np.any(screenshot[dino_y - 20, i] != get_pixel(screenshot, dino_x, dino_y)):
                    keyboard.press(Key.down)
                    time.sleep(0.5)
                    keyboard.release(Key.down)
                    break

        # Add a short sleep to prevent excessive CPU usage
        time.sleep(0.01)

start()
