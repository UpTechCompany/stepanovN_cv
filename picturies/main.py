import cv2
import numpy as np

def find_image_in_video(image_path, video_path, threshold=0.7):
    # Загрузка изображения
    image = cv2.imread(image_path)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = image_gray.shape

    # Определение цветовых областей (оранжевая рамка и синий фон)
    lower_orange = np.array([5, 100, 100])
    upper_orange = np.array([15, 255, 255])
    orange_mask = cv2.inRange(image_hsv, lower_orange, upper_orange)

    lower_blue = np.array([100, 50, 50])
    upper_blue = np.array([130, 255, 255])
    blue_mask = cv2.inRange(image_hsv, lower_blue, upper_blue)

    # Объединение масок
    combined_mask = cv2.bitwise_or(orange_mask, blue_mask)

    # Загрузка видео
    cap = cv2.VideoCapture(video_path)

    # Создание объекта FlannBasedMatcher
    flann = cv2.FlannBasedMatcher()
    orb = cv2.ORB_create()
    kp1, des1 = orb.detectAndCompute(image_gray, None)

    matches_count = 0

    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            break

        # Поиск особых точек и их дескрипторов в текущем кадре видео
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        kp2, des2 = orb.detectAndCompute(gray, None)

        # Сопоставление дескрипторов особых точек между изображением и кадром видео
        matches = flann.knnMatch(des1, des2, k=2)

        # Применение ratio test, чтобы найти лучшие соответствия
        good_matches = []
        for m, n in matches:
            if m.distance < threshold * n.distance:
                good_matches.append(m)

        # Проверка цвета окрестности особых точек
        for match in good_matches:
            x, y = kp2[match.trainIdx].pt
            x, y = int(x), int(y)
            if combined_mask[y, x] > 0:
                matches_count += 1
                break

    cap.release()
    cv2.destroyAllWindows()

    return matches_count

# Пример использования
image_path = "Stepanov.png"
video_path = "pictures.avi"
matches_count = find_image_in_video(image_path, video_path)
print("Количество совпадений изображения в видео:", matches_count)
