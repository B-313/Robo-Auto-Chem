import cv2
import pandas as pd
import time

# Function to detect color in a frame

def detect_colour_in_frame(frame):
    # Convert frame to HSV color space
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    # Define color ranges for red, yellow, and green
    color_ranges = {
        'red': ((0, 70, 50), (10, 255, 255)),
        'yellow': ((15, 150, 150), (35, 255, 255)),
        'green': ((40, 70, 50), (90, 255, 255))
    }
    color_detection = {'red': 0, 'yellow': 0, 'green': 0}
    for color, (lower, upper) in color_ranges.items():
        mask = cv2.inRange(hsv, lower, upper)
        # Count the number of non-zero pixels in the mask
        color_detection[color] = cv2.countNonZero(mask)
    return color_detection

# Integration into basic_recorder function

def basic_recorder():
    # Initialize video capture
    cap = cv2.VideoCapture(0)
    detected_colors = []
    # Init DataFrame for logging
    color_data = []

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        # Call the color detection function
        color_count = detect_colour_in_frame(frame)
        detected_colors.append(color_count)
        color_data.append(time.strftime('%Y-%m-%d %H:%M:%S') + ',' + ','.join(str(color_count[color]) for color in ['red', 'yellow', 'green'])));
        # Existing operations in basic_recorder continue...

    # Save the color data to CSV
    df = pd.DataFrame(color_data, columns=['Timestamp', 'Red', 'Yellow', 'Green'])
    df.to_csv('colour_log.csv', index=False)

    # Create a summary file
    summary = f"Total Red: {sum(item['red'] for item in detected_colors)}, Total Yellow: {sum(item['yellow'] for item in detected_colors)}, Total Green: {sum(item['green'] for item in detected_colors)}"
    with open('colour_summary.txt', 'w') as f:
        f.write(summary)

    cap.release()  
    cv2.destroyAllWindows()