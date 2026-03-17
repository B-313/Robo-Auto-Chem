import cv2
import csv
import datetime

# Initialize the camera
cap = cv2.VideoCapture(0)

# Define the color ranges
color_ranges = {
    "red": ((0, 100, 100), (10, 255, 255)),
    "yellow": ((20, 100, 100), (30, 255, 255)),
    "green": ((50, 100, 100), (70, 255, 255))
}

# Open a CSV file to log the detected colors
with open('color_log.csv', mode='a', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Timestamp', 'Detected Color'])

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        # Convert the frame to HSV color space
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        detected_color = None

        # Check for colors
        for color, (lower, upper) in color_ranges.items():
            mask = cv2.inRange(hsv_frame, lower, upper)
            if cv2.countNonZero(mask) > 0:
                detected_color = color
                break

        # Log detected color with timestamp
        if detected_color:
            timestamp = datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            writer.writerow([timestamp, detected_color])
            print(f'Detected Color: {detected_color} at {timestamp}')

        # Display the resulting frame
        cv2.imshow('Color Detection', frame)

        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

# When everything done, release the capture
cap.release()
cv2.destroyAllWindows()