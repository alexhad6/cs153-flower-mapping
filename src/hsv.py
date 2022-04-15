import cv2

def hsv_mask(image, lower_thresh, upper_thresh):
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    return cv2.inRange(image_hsv, lower_thresh, upper_thresh)
