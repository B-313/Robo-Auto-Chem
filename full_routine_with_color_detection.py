import cv2
import numpy as np
import csv
import time

# Initialization of the robot control
# (assumed imports and definitions remain unchanged)

# Camera function to record video and perform color detection

def camera_function():
    # Start video capture
    cap = cv2.VideoCapture(0)
    # Create a CSV file to log color changes
    with open('color_log.csv', mode='w', newline='') as color_log:
        color_writer = csv.writer(color_log)
        color_writer.writerow(['Time', 'Detected Color'])

        while True:
            ret, frame = cap.read()  # Capture frame
            if not ret:
                break
            # Convert frame to HSV color space
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

            # Color detection ranges
            red_mask = cv2.inRange(hsv, (0, 100, 100), (10, 255, 255)) + cv2.inRange(hsv, (160, 100, 100), (180, 255, 255))
            yellow_mask = cv2.inRange(hsv, (20, 100, 100), (30, 255, 255))
            green_mask = cv2.inRange(hsv, (40, 100, 100), (70, 255, 255))

            # Check for colors and log them
            detected_colors = []
            if cv2.countNonZero(red_mask) > 0:
                detected_colors.append('red')
            if cv2.countNonZero(yellow_mask) > 0:
                detected_colors.append('yellow')
            if cv2.countNonZero(green_mask) > 0:
                detected_colors.append('green')

            if detected_colors:
                timestamp = time.strftime('%Y-%m-%d %H:%M:%S')
                for color in detected_colors:
                    color_writer.writerow([timestamp, color])
                    print(f'Detected color: {color}')  # Optional: Print detected colors

            # Display the resulting frame (optional)
            cv2.imshow('Color Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        # Release the capture
    cap.release()
    cv2.destroyAllWindows()


# Main loop (remains unchanged)

# camera_function()  # Uncomment to start camera function

