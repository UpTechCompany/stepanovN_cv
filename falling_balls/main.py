import cv2
import numpy as np
import pymunk
import random

# Установки экрана
screen_width = 800
screen_height = 600

# Инициализация Pymunk
space = pymunk.Space()
space.gravity = (0, 981)  # гравитация, 981 пиксель/сек² по оси Y

# Цвета
blue = (255, 0, 0)  # OpenCV использует BGR формат
green = (0, 255, 0)
white = (255, 255, 255)

# Класс для создания шариков
class Ball:
    def __init__(self, x, y, radius):
        self.radius = radius
        self.body = pymunk.Body(1, pymunk.moment_for_circle(1, 0, radius))
        self.body.position = x, y
        self.shape = pymunk.Circle(self.body, radius)
        self.shape.elasticity = 0.8
        self.shape.collision_type = 1  # Устанавливаем тип коллизии для шариков
        space.add(self.body, self.shape)

    def draw(self, img):
        pos = int(self.body.position.x), int(self.body.position.y)
        cv2.circle(img, pos, self.radius, blue, -1)

# Создание статичной поверхности для "пола"
static_body = space.static_body
floor = pymunk.Segment(static_body, (0, screen_height), (screen_width, screen_height), 1)

# Функция для создания препятствий из контуров
def create_obstacles_from_contours(contours):
    for contour in contours:
        contour = contour.reshape(-1, 2)
        contour = [(int(point[0]), int(point[1])) for point in contour]
        if len(contour) < 3:
            continue
        obstacle_body = pymunk.Body(body_type=pymunk.Body.STATIC)
        obstacle_shape = pymunk.Poly(obstacle_body, contour)
        space.add(obstacle_body, obstacle_shape)

# Функция для обработки видео с камеры и нахождения контуров
def get_obstacle_contours(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blurred, 100, 255)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

# Инициализация видеопотока с камеры
cap = cv2.VideoCapture(0)

# Список шариков
balls = []

# Основной игровой цикл
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Получение контуров препятствий с камеры
    contours = get_obstacle_contours(frame)

    # Очистка старых препятствий
    for shape in space.shapes:
        if shape.body.body_type == pymunk.Body.STATIC and shape != floor:
            space.remove(shape.body, shape)

    # Создание новых препятствий из контуров
    create_obstacles_from_contours(contours)

    # Добавление нового шарика в случайной позиции сверху
    if random.random() < 0.05:  # Вероятность появления шарика
        x = random.randint(20, screen_width - 20)
        y = 0
        radius = random.randint(10, 20)
        balls.append(Ball(x, y, radius))

    # Обновление физики
    space.step(1 / 100.0)

    # Удаление шариков, которые ушли за границу пола
    balls = [ball for ball in balls if ball.body.position.y < screen_height + ball.radius]

    # Создание пустого изображения с белым фоном
    img = np.ones((screen_height, screen_width, 3), dtype=np.uint8) * 255

    # Отрисовка шариков
    for ball in balls:
        ball.draw(img)

    # Отрисовка контуров препятствий
    for contour in contours:
        cv2.drawContours(img, [contour], -1, green, 2)

    # Показ изображения
    cv2.imshow("UpTech", img)

    # Выход по нажатию клавиши 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
