import cv2
import numpy as np
import pyautogui
import time
from mss import mss
from skimage.metrics import structural_similarity as ssim


def capture_screen(region):
    with mss() as sct:
        screen = sct.grab(region)
        return np.array(screen)


def preprocess(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, threshold = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
    return threshold


def find_dinosaur(image):
    template = cv2.imread('dinosaur_template.png', cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(result)
    return max_loc, max_val


def find_game_over(image):
    template = cv2.imread('game_over_template.png', cv2.IMREAD_GRAYSCALE)
    result = cv2.matchTemplate(image, template, cv2.TM_CCOEFF_NORMED)
    _, max_val, _, _ = cv2.minMaxLoc(result)
    return max_val


def jump():
    pyautogui.press('space')


def main():
    game_region = {'top': 280, 'left': 0, 'width': 800, 'height': 200}
    previous_score = -1
    score = 0

    while score < 10000:
        screen = capture_screen(game_region)
        processed_screen = preprocess(screen)

        dinosaur_loc, dinosaur_confidence = find_dinosaur(processed_screen)
        game_over_confidence = find_game_over(processed_screen)

        if dinosaur_confidence > 0.8 and game_over_confidence < 0.8:
            if previous_score != score:
                print(f'Score: {score}')
                previous_score = score

            jump()
            score += 1
            time.sleep(0.08)  # Adjust delay for better performance

    print("Reached 10000 points!")


if __name__ == "__main__":
    main()
