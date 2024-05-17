import random

import cv2
from time import perf_counter
from operator import itemgetter
from random import choice

camera = cv2.VideoCapture(0)

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)

# yellow
y_lower = (22, 150, 150)
y_upper = (35, 255, 255)

# green
g_lower = (50, 120, 120)
g_upper = (75, 255, 255)

# red
r_lower = (2, 120, 120)
r_upper = (5, 255, 255)

#blue
b_lower = (89,100,100)
b_upper = (109,255,255)

upper = {
    "y": y_upper,
    "g": g_upper,
    "r": r_upper,
    "b": b_upper
}

lower = {
    "y": y_lower,
    "g": g_lower,
    "r": r_lower,
    "b": b_lower
}


points = []

prev_time = perf_counter()
curr_time = perf_counter()

prev_x = 0
prev_y = 0

arr = []

curr_x = 0
curr_y = 0

line = {"g": (0,0),
        "r": (0,0),
        "y": (0,0),
        "b": (0,0)}

res = ["g", 'r', "y", "b"]
random.shuffle(res)

curr_dist = 0

d = 7.37  # cm
r = 1
while camera.isOpened():
    _, image = camera.read()
    curr_time = perf_counter()
    blurred = cv2.GaussianBlur(image, (11, 11), 0)
    hsv = cv2.cvtColor(blurred, cv2.COLOR_BGR2HSV)

    for l, u in list(zip(lower.keys(), upper.keys())):
        mask = cv2.inRange(hsv, lower[l], upper[u])
        mask = cv2.erode(mask, None, iterations=2)
        mask = cv2.dilate(mask, None, iterations=2)

        cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL,
                                cv2.CHAIN_APPROX_SIMPLE)[0]

        if len(cnts) > 0:
            c = max(cnts, key=cv2.contourArea)
            (curr_x, curr_y), r = cv2.minEnclosingCircle(c)
            line[u] = (curr_x, curr_y)
            if r > 10:
                cv2.circle(image, (int(curr_x), int(curr_y)), int(r), (0, 255, 255), 3)

    print(res, list(dict(sorted(line.items(), key=itemgetter(1))).keys()))
    if res == list(dict(sorted(line.items(), key=itemgetter(1))).keys()):
        print("yes")

    # delta = curr_time - prev_time
    # dist = ((curr_x - curr_y)**2 + (prev_x - prev_y)**2) ** 0.5
    #
    # pxl_per_m = (d / 100) / (2 * r)
    #
    # cv2.putText(image, f"Speed = {dist * pxl_per_m / delta:.4f}m/s",
    #             (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 3,
    #             (127, 255, 255))
    # arr.append(dist * pxl_per_m / delta)
    prev_x = curr_x
    prev_y = curr_y
    prev_time = curr_time
    line["g"] = (0, 0)
    line["r"] = (0, 0)
    line["y"] = (0, 0)
    line["b"] = (0, 0)

    # points.append((curr_x, curr_y))

    # if len(points) > 12:
    #     points.pop(0)
    #
    # if len(points) >= 2:
    #     for n, i in enumerate(range(len(points) - 1)):
    #         p1 = points[i]
    #         p2 = points[i+1]
    #         cv2.line(image, tuple(map(int, p1)), tuple(map(int, p2)), (255, 0, 0), n+1)
    key = cv2.waitKey(1)
    if key == ord('q'):
        break
    cv2.imshow("Image", image)
    cv2.imshow("Mask", mask)
# print(max(arr))
camera.release()
cv2.destroyAllWindows()
