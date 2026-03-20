import cv2
import numpy as np

def detect_colour_in_frame(frame):
    """
    Detects the number of pixels for red, yellow, and green in a video frame.
    Returns a dict: {'red': int, 'yellow': int, 'green': int}
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_ranges = {
        'red': ((0, 70, 50), (10, 255, 255)),
        'yellow': ((15, 150, 150), (35, 255, 255)),
        'green': ((40, 70, 50), (90, 255, 255))
    }
    color_detection = {'red': 0, 'yellow': 0, 'green': 0}
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        color_detection[color] = cv2.countNonZero(mask)
    return color_detection
