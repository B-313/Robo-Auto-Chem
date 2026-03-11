import cv2 as cv
import numpy as np
from datetime import datetime
import os
import time
import csv

base_name = 'green_fixed_changes'
video_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/{base_name}.mp4"
csv_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/{base_name}_changes.csv"
TARGET_FPS = 10.0
width, height = 1280, 720
camera_index = 0

# YOUR TUNED ROI
ROI = [450, 250, 750, 450]  # Update from tuner!

record_seconds = 60

os.makedirs(os.path.dirname(video_path), exist_ok=True)

cap = cv.VideoCapture(camera_index)
cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)

fourcc = cv.VideoWriter_fourcc(*'mp4v')
out = cv.VideoWriter(video_path, fourcc, TARGET_FPS, (width, height))

change_log = []
csv_headers = ['timestamp', 'frame', 'time_s', 'from_state', 'to_state', 'BGR']

print("🟢 GREEN BOOSTED - Colour Changes Only...")

current_state = "NONE"
start_time = time.time()
frame_count = 0

while time.time() - start_time < record_seconds:
    ret, frame = cap.read()
    if not ret:
        break

    # ROI COLOUR
    x1, y1, x2, y2 = ROI
    roi = frame[y1:y2, x1:x2]

    # BGR MEANS
    b, g, r = np.mean(roi, axis=(0,1)).astype(int)

    # RELATIVE COLOUR (GREEN BOOSTED)
    new_state = "NONE"
    if g > r + 25 and g > b + 25: new_state = "GREEN"     # STRONGER GREEN!
    elif r > g + 25 and r > b + 25: new_state = "RED"
    elif r > b + 15 and g > b + 15: new_state = "YELLOW"

    # CHANGE DETECTED → LOG
    if new_state != current_state and new_state != "NONE":
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        change_log.append([timestamp, frame_count, round(time.time() - start_time, 1), 
                          current_state, new_state, f"B{r}G{g}R{b}"])
        print(f"🔄 {timestamp}: {current_state} → {new_state} (B{r}G{g}R{b})")
        current_state = new_state

    # SIMPLE VISUAL
    cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
    state_emoji = {"GREEN": "🟢", "RED": "🔴", "YELLOW": "🟡"}.get(current_state, "⚪")
    cv.putText(frame, f"{state_emoji} {current_state} (B{r}G{g}R{b})", 
               (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    out.write(frame)
    frame_count += 1

    cv.imshow('Green Fixed Logger', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()

# CSV (changes only)
with open(csv_path, 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(csv_headers)
    writer.writerows(change_log)

print(f"\n✅ {len(change_log)} changes logged!")
print(f"📹 {video_path} | 📝 {csv_path}")
if change_log:
    print("\nRecent:")
    for row in change_log[-3:]:
        print(f"  {row[0]} | {row[3]} → {row[4]}")
 