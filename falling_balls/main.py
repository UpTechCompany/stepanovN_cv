import cv2
import numpy as np
import pymunk
import random

# Установки экрана
screen_width = 1280
screen_height = 800

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
space.add(floor)

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
    dilated = cv2.dilate(edges, None, iterations=2)  # Добавление функции dilate
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    return contours

pp_rect = np.zeros((4, 2), dtype="float32")
pp_dst = np.zeros((4, 2), dtype="float32")
maxW = 0
maxH = 0

def order_points(pts):
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    return rect

def find_paper_and_crop(image):
    global pp_rect, pp_dst, maxW, maxH
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 70, 255, type=cv2.THRESH_BINARY)[1]
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        largest_contour = max(contours, key=cv2.contourArea)
        epsilon = 0.02 * cv2.arcLength(largest_contour, True)
        approx = cv2.approxPolyDP(largest_contour, epsilon, True)
        if len(approx) == 4:
            pts = approx.reshape(4, 2)
            rect = order_points(pts)
            widthA = np.linalg.norm(rect[2] - rect[3])
            widthB = np.linalg.norm(rect[1] - rect[0])
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.linalg.norm(rect[1] - rect[2])
            heightB = np.linalg.norm(rect[0] - rect[3])
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([[0, 0], [maxWidth, 0], [maxWidth, maxHeight], [0, maxHeight]], dtype="float32")
            pp_rect = rect
            pp_dst = dst
            maxW = maxWidth
            maxH = maxHeight
            warped = apply_perspective(thresh, rect, dst, maxW, maxH)
            return warped
        else:
            raise Exception("Paper not detected or the paper does not have four corners")
    return 0

def apply_perspective(image, rect, ppd, maxW, maxH):
    if rect.shape == (4, 2) and ppd.shape == (4, 2):
        rect = np.array(rect, dtype="float32")
        ppd = np.array(ppd, dtype="float32")
        M = cv2.getPerspectiveTransform(rect, ppd)
        return cv2.warpPerspective(image, M, (maxW, maxH))
    else:
        raise ValueError("Invalid shape for rect or ppd in apply_perspective")

def process_image(img):
    global pp_rect, pp_dst, maxW, maxH
    if pp_rect.shape == (4, 2) and pp_dst.shape == (4, 2) and maxW > 0 and maxH > 0:
        img = apply_perspective(img, pp_rect, pp_dst, maxW, maxH)
        img = cv2.resize(img, (640, 480), interpolation=cv2.INTER_NEAREST)
    return img

lower = 0
upper = 55

def detect_rects(image):
    processed_img = process_image(image)
    gray = processed_img.copy()
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    mask = cv2.Canny(gray, lower, upper)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    rects = []
    for cnt in contours:
        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        if cv2.arcLength(box, True) > 200:
            continue
        rects.append(box)
    return rects

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
    # contours = detect_rects(frame)
    processed_img = process_image(frame)
    gray = processed_img.copy()
    gray = cv2.GaussianBlur(gray, (7, 7), 0)
    mask = cv2.Canny(gray, lower, upper)
    dilated = cv2.dilate(mask, None, iterations=6)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = [cnt for cnt in contours if cv2.contourArea(cnt) >= 300]

    # Очистка старых препятствий
    for shape in space.shapes:
        if shape.body.body_type == pymunk.Body.STATIC and shape != floor:
            space.remove(shape.body, shape)

    # Создание новых препятствий из контуров
    create_obstacles_from_contours(contours)

    # Добавление нового шарика в случайной позиции сверху
    if random.random() < 0.1:  # Вероятность появления шарика
        x = random.randint(20, screen_width - 20)
        y = 0
        radius = random.randint(10, 20)
        balls.append(Ball(x, y, radius))

    # Обновление физики
    space.step(1 / 10.0)

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
