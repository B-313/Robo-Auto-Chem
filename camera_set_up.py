# Camera 2 test - GROUP B

'''
import cv2 as cv
from datetime import datetime
import os
import time

############
# SETTINGS #
############

video_name = 'z_test_30fps_30_delay.mp4'

output_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/{video_name}" # the path needs to end with the name of the file
fps = 30.0
delay = 30.0 / fps
width = 1280
height = 720
camera_index = 0 # port of the camera
record_seconds = 10   # set RECORDING DURATION here

############
############

os.makedirs(os.path.dirname(output_path), exist_ok=True) # don't change this!

cap = cv.VideoCapture(camera_index)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, fps, (width, height))

print(f"Recording for {record_seconds} seconds...")

start_time = time.time()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Can't receive frame. Exiting...")
        break

    # Timestamp overlay
    
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv.putText(frame, timestamp,
               (20, height - 20),
               cv.FONT_HERSHEY_SIMPLEX,
               0.7,
               (0, 255, 0),
               2,
               cv.LINE_AA)

    out.write(frame)import cv2 as cv
from datetime import datetime
import os
import time

 

############
# SETTINGS #
############
video_name = 'z_test_fixed.mp4'
output_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/{video_name}"
TARGET_FPS = 5.0  # SLOWER = normal timestamps
width, height = 1280, 720
camera_index = 0
record_seconds = 10

 

os.makedirs(os.path.dirname(output_path), exist_ok=True)

 

cap = cv.VideoCapture(camera_index)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Flush old frames

 

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, TARGET_FPS, (width, height))

 

print(f"Recording {TARGET_FPS} FPS for {record_seconds}s...")

 

start_time = time.time()
frame_count = 0

 

while time.time() - start_time < record_seconds:
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv.putText(frame, timestamp, (20, height - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, 
               (0, 255, 0), 2, cv.LINE_AA)

    out.write(frame)
    frame_count += 1

    # SPEED FIX: Control ACTUAL capture rate
    time.sleep(1.0 / TARGET_FPS)

 

print(f"Recorded {frame_count} frames @ {TARGET_FPS} FPS")
cap.release()
out.release()
cv.destroyAllWindows()
print(f"Saved: {output_path}")

    # Stop after desired time
    if time.time() - start_time >= record_seconds:
        break

# dont deletes the following lines, or we will have to restart!
cap.release()
out.release()
cv.destroyAllWindows()

print(f"Finished. Video saved to: {output_path}")

'''
import cv2 as cv
from datetime import datetime
import os
import time
 
############
# SETTINGS #
############
video_name = 'z_test_fixed.mp4'
output_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/{video_name}"
TARGET_FPS = 5.0  # SLOWER = normal timestamps
width, height = 1280, 720
camera_index = 0
record_seconds = 10
 
os.makedirs(os.path.dirname(output_path), exist_ok=True)
 
cap = cv.VideoCapture(camera_index)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
cap.set(cv.CAP_PROP_BUFFERSIZE, 1)  # Flush old frames
 
fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(output_path, fourcc, TARGET_FPS, (width, height))
 
print(f"Recording {TARGET_FPS} FPS for {record_seconds}s...")
 
start_time = time.time()
frame_count = 0
 
while time.time() - start_time < record_seconds:
    ret, frame = cap.read()
    if not ret:
        break
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cv.putText(frame, timestamp, (20, height - 20), cv.FONT_HERSHEY_SIMPLEX, 0.7, 
               (0, 255, 0), 2, cv.LINE_AA)
    out.write(frame)
    frame_count += 1
    # SPEED FIX: Control ACTUAL capture rate
    time.sleep(1.0 / TARGET_FPS)
 
print(f"Recorded {frame_count} frames @ {TARGET_FPS} FPS")
cap.release()
out.release()
cv.destroyAllWindows()
print(f"Saved: {output_path}")