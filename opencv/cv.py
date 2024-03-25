import cv2

cam = cv2.VideoCapture(0)
if cam.isOpened():
    ret, frame = cam.read()
    print(frame.shape)
cam.release()