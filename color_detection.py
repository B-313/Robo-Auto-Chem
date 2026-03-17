import cv2 as cv
import os
import time
import csv
from datetime import datetime

def color_detection_recorder(vial_number, camera_index=0, width=1280, height=720, fps=30, record_seconds=10):
    """
    Records video from camera while detecting RED, YELLOW, and GREEN colors.
    Logs color detection events to a CSV file with timestamps and hue values.
    
    Args:
        vial_number: Vial number (0-3)
        camera_index: Camera device index (default 0)
        width: Video width (default 1280)
        height: Video height (default 720)
        fps: Frames per second (default 30)
        record_seconds: Recording duration in seconds (default 10)
    """
    
    # Path for saving the video
    output_video_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/{vial_number+1}.mp4"
    
    # Path for saving the CSV log
    output_csv_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/color_detection_log.csv"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_video_path), exist_ok=True)
    
    # Initialize CSV file if it doesn't exist
    if not os.path.exists(output_csv_path):
        with open(output_csv_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['vial_number', 'timestamp', 'color_detected', 'hue_value', 'frame_count'])
    
    # Open the camera
    cap = cv.VideoCapture(camera_index, cv.CAP_V4L2)
    
    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Cannot open camera")
        return
    
    # Set video capture resolution
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    
    # Verify if the camera supports the desired resolution
    actual_width = cap.get(cv.CAP_PROP_FRAME_WIDTH)
    actual_height = cap.get(cv.CAP_PROP_FRAME_HEIGHT)
    
    if actual_width != width or actual_height != height:
        print(f"Warning: Camera resolution set to {actual_width}x{actual_height} instead of {width}x{height}.")
    
    # Set the FPS (frame rate)
    cap.set(cv.CAP_PROP_FPS, fps)
    
    # Create VideoWriter object with desired codec
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(output_video_path, fourcc, fps, (int(actual_width), int(actual_height)))
    
    # Define color ranges in HSV
    # RED: 0-10 and 170-180 (wraps around)
    # YELLOW: 20-30
    # GREEN: 40-80
    
    color_ranges = {
        'RED': [(0, 100, 100, 10, 255, 255), (170, 100, 100, 180, 255, 255)],
        'YELLOW': [(20, 100, 100, 30, 255, 255)],
        'GREEN': [(40, 100, 100, 80, 255, 255)]
    }
    
    # Start recording and color detection
    print(f"Recording for {record_seconds} seconds with color detection... Vial {vial_number + 1}")
    
    start_time = time.time()
    frame_count = 0
    last_detected_color = None
    
    while True:
        ret, frame = cap.read()
        
        # If frame is not successfully captured, exit
        if not ret:
            print("Can't receive frame. Exiting...")
            break
        
        # Convert frame to HSV
        hsv_frame = cv.cvtColor(frame, cv.COLOR_BGR2HSV)
        
        # Detect colors
        detected_color = detect_color(hsv_frame, color_ranges)
        
        # Log color detection only when it changes
        if detected_color and detected_color != last_detected_color:
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
            hue_value = get_dominant_hue(hsv_frame, color_ranges, detected_color)
            
            # Write to CSV
            with open(output_csv_path, 'a', newline='') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow([vial_number + 1, timestamp, detected_color, hue_value, frame_count])
            
            print(f"[Vial {vial_number + 1}] Color detected: {detected_color} | Hue: {hue_value} | Frame: {frame_count}")
            last_detected_color = detected_color
        
        # Write the frame to the video file
        out.write(frame)
        frame_count += 1
        
        # Stop after the specified recording duration
        if time.time() - start_time > record_seconds:
            break
    
    # Release everything once the recording is done
    cap.release()
    out.release()
    cv.destroyAllWindows()
    
    print(f"Video saved to: {output_video_path}")
    print(f"Color log saved to: {output_csv_path}")


def detect_color(hsv_frame, color_ranges):
    """
    Detects which color is dominant in the frame.
    
    Args:
        hsv_frame: Frame in HSV color space
        color_ranges: Dictionary with color ranges
    
    Returns:
        String with detected color name or None
    """
    
    color_pixels = {}
    
    for color_name, ranges in color_ranges.items():
        total_pixels = 0
        
        for h_min, s_min, v_min, h_max, s_max, v_max in ranges:
            lower = (h_min, s_min, v_min)
            upper = (h_max, s_max, v_max)
            mask = cv.inRange(hsv_frame, lower, upper)
            total_pixels += cv.countNonZero(mask)
        
        color_pixels[color_name] = total_pixels
    
    # Return the color with the most pixels (if any detected)
    if max(color_pixels.values()) > 0:
        return max(color_pixels, key=color_pixels.get)
    
    return None


def get_dominant_hue(hsv_frame, color_ranges, detected_color):
    """
    Gets the average hue value for the detected color.
    
    Args:
        hsv_frame: Frame in HSV color space
        color_ranges: Dictionary with color ranges
        detected_color: The color to analyze
    
    Returns:
        Average hue value (0-180)
    """
    
    if detected_color not in color_ranges:
        return 0
    
    ranges = color_ranges[detected_color]
    combined_mask = None
    
    for h_min, s_min, v_min, h_max, s_max, v_max in ranges:
        lower = (h_min, s_min, v_min)
        upper = (h_max, s_max, v_max)
        mask = cv.inRange(hsv_frame, lower, upper)
        
        if combined_mask is None:
            combined_mask = mask
        else:
            combined_mask = cv.bitwise_or(combined_mask, mask)
    
    # Calculate average hue of detected pixels
    hue_channel = hsv_frame[:, :, 0]
    detected_hues = hue_channel[combined_mask > 0]
    
    if len(detected_hues) > 0:
        return int(detected_hues.mean())
    
    return 0