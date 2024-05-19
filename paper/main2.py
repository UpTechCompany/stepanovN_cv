import cv2
import numpy as np
import zmq

cv2.namedWindow("Image", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Mask", cv2.WINDOW_GUI_NORMAL)
cv2.namedWindow("Image1", cv2.WINDOW_GUI_NORMAL)

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.setsockopt(zmq.SUBSCRIBE, b"")

cam = cv2.VideoCapture(0)

lower_threshold = 200
upper_threshold = 255

# Update threshold parameters
def lower_threshold_update(value):
    global lower_threshold
    lower_threshold = value

def upper_threshold_update(value):
    global upper_threshold
    upper_threshold = value

cv2.createTrackbar("Lower Threshold", "Mask", lower_threshold, 255, lower_threshold_update)
cv2.createTrackbar("Upper Threshold", "Mask", upper_threshold, 255, upper_threshold_update)

# Function to add text to the image with perspective correction
def add_text_perspective(image, text, contour):
    rect = cv2.minAreaRect(contour)
    box = np.intp(cv2.boxPoints(rect))

    cx = int(rect[0][0])
    cy = int(rect[0][1])
    angle = rect[2]

    rows, cols = image.shape[:2]
    M = cv2.getRotationMatrix2D((cx, cy), angle, 1)
    rotated = cv2.warpAffine(image, M, (cols, rows))

    # Create a white background image
    white_paper = np.ones_like(rotated) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1
    font_color = (0, 0, 0)  # Black color for text
    thickness = 2
    text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
    text_x = int((cols - text_size[0]) / 2)
    text_y = int((rows + text_size[1]) / 2)
    cv2.putText(white_paper, text, (text_x, text_y), font, font_scale, font_color, thickness, cv2.LINE_AA)

    # Overlay the text on the rotated image
    result = cv2.addWeighted(rotated, 1, white_paper, 0.5, 0)

    return result

while cam.isOpened():
    _, image = cam.read()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)

    # Thresholding with adjustable parameters
    _, threshold = cv2.threshold(blurred, lower_threshold, upper_threshold, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(threshold, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filter contours based on area
    filtered_cnts = [cnt for cnt in cnts if cv2.contourArea(cnt) > 5000]

    # Draw contours
    cv2.drawContours(image, filtered_cnts, -1, (0, 0, 255), 5)
    if filtered_cnts:
        image1 = add_text_perspective(image, "Hello world", filtered_cnts[0])
        cv2.imshow("Image1", image1)
    cv2.imshow("Image", image)
    cv2.imshow("Mask", threshold)

    key = cv2.waitKey(10)
    if key == ord("q"):
        break

cv2.destroyAllWindows()
