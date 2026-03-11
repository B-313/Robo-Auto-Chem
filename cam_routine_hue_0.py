import cv2 as cv
import numpy as np
from datetime import datetime
import os, sys, time, csv, threading

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.UR_Functions import URfunctions as URControl
from utils.robotiq_gripper import RobotiqGripper

# =============================================================================
# SETTINGS  <-- the only things you should need to change
# =============================================================================

ROBOT_IP              = "192.168.0.2"
CAMERA_INDEX          = 0
ROI                   = [450, 250, 750, 450]   # [x1, y1, x2, y2] from tuner
EXPERIMENT_DURATION_S = 180                    # seconds to wait on stirrer
VIDEO_PATH            = "/home/robot/group_B/robo_chem_504/group_B_videos/experiment.mp4"
CSV_PATH              = "/home/robot/group_B/robo_chem_504/group_B_videos/experiment_log.csv"

# HSV hue ranges for each colour (H is 0-179 in OpenCV)
# If a colour is missed, change the numbers here
GREEN_HUE  = (60, 95)   # teal-green
YELLOW_HUE = (15, 40)   # amber-yellow
RED_HUE    = (0,  15)   # red (also catches H 160-179, handled in code)

# Set to True if you want to see live H/S/V numbers in the terminal
DEBUG = False

# =============================================================================
# SHARED FLAG  (tells camera when to start and stop)
# =============================================================================
camera_should_record = False
camera_should_stop   = False


# =============================================================================
# COLOUR DETECTION
# =============================================================================
def get_colour(roi_bgr):
    hsv  = cv.cvtColor(roi_bgr, cv.COLOR_BGR2HSV)
    h, s, v = np.mean(hsv, axis=(0, 1)).astype(int)

    if DEBUG:
        print(f"  HSV -> H:{h}  S:{s}  V:{v}")

    total = roi_bgr.shape[0] * roi_bgr.shape[1]

    def pixels_in_range(h_low, h_high):
        mask = cv.inRange(hsv, (h_low, 50, 50), (h_high, 255, 255))
        return cv.countNonZero(mask) / total  # fraction of ROI

    green_frac  = pixels_in_range(*GREEN_HUE)
    yellow_frac = pixels_in_range(*YELLOW_HUE)
    red_frac    = pixels_in_range(0, 15) + pixels_in_range(160, 179)  # red wraps in HSV

    best = max([("GREEN", green_frac), ("YELLOW", yellow_frac), ("RED", red_frac)],
               key=lambda x: x[1])

    # Only count it if at least 20% of the ROI matches
    if best[1] < 0.20:
        return "NONE", h, s, v

    return best[0], h, s, v


# =============================================================================
# CAMERA  (runs in background while robot moves)
# =============================================================================
def run_camera():
    global camera_should_record, camera_should_stop

    os.makedirs(os.path.dirname(VIDEO_PATH), exist_ok=True)

    cap = cv.VideoCapture(CAMERA_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  1280)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, 720)

    out = cv.VideoWriter(VIDEO_PATH,
                         cv.VideoWriter_fourcc(*'mp4v'),
                         10.0, (1280, 720))

    log          = []
    current      = "NONE"
    start_time   = None
    frame_count  = 0

    print("[CAMERA] Ready -- waiting for vial to be placed on stirrer...")

    while True:

        # Wait for robot to place vial
        if not camera_should_record:
            time.sleep(0.05)
            continue

        # Start timer on first recorded frame
        if start_time is None:
            start_time = time.time()
            print("[CAMERA] Recording started!")
            print("[CAMERA] !! Turn the stirrer ON now !!")

        # Stop when robot picks vial back up
        if camera_should_stop:
            print("[CAMERA] Stopping recording.")
            break

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # Check colour in ROI
        x1, y1, x2, y2 = ROI
        colour, h, s, v = get_colour(frame[y1:y2, x1:x2])

        # Log when colour changes
        if colour != current and colour != "NONE":
            t = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            elapsed = round(time.time() - start_time, 1)
            log.append([t, elapsed, frame_count, current, colour, h, s, v])
            print(f"[CAMERA] {t}  (+{elapsed}s)  {current} -> {colour}")
            current = colour

        # Draw on frame
        box_colours = {"GREEN": (0,200,0), "RED": (0,0,220), "YELLOW": (0,200,220)}
        box_col = box_colours.get(current, (200,200,200))
        elapsed_now = round(time.time() - start_time, 1) if start_time else 0

        cv.rectangle(frame, (x1,y1), (x2,y2), box_col, 3)
        cv.putText(frame, f"{current}  H:{h} S:{s} V:{v}",
                   (20, 40), cv.FONT_HERSHEY_SIMPLEX, 0.8, box_col, 2)
        cv.putText(frame, f"t={elapsed_now}s  changes={len(log)}",
                   (20, 75), cv.FONT_HERSHEY_SIMPLEX, 0.6, (200,200,200), 2)

        out.write(frame)
        frame_count += 1

        cv.imshow("Traffic Light Experiment", frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # Save everything
    cap.release()
    out.release()
    cv.destroyAllWindows()

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['time', 'elapsed_s', 'frame', 'from', 'to', 'H', 'S', 'V'])
        writer.writerows(log)

    print(f"\n[CAMERA] Done! {len(log)} colour changes saved.")
    print(f"[CAMERA] Video -> {VIDEO_PATH}")
    print(f"[CAMERA] CSV   -> {CSV_PATH}")


# =============================================================================
# ROBOT SEQUENCE
# =============================================================================
def run_robot():
    global camera_should_record, camera_should_stop

    robot   = URControl(ip=ROBOT_IP, port=30003)
    gripper = RobotiqGripper()
    gripper.connect(ROBOT_IP, 63352)

    # Shortcut so each move is one readable line
    def move(joints, label):
        print(f"[ROBOT] {label}")
        robot.move_joint_list(joints, 0.5, 0.5, 0.02)

    try:
        gripper.move(255, 125, 125)

        move([1.447051763534546,  -1.5378911060145875, 1.5790107885943812,
              -1.6676823101439417, -1.5373862425433558, -0.32640201250185186], "Home")
        gripper.move(0, 255, 255)

        move([1.167259693145752,  -1.0337765973857422, 1.4777286688434046,
              -1.9788419208922328, -1.4967520872699183, -0.5067313353167933],  "Above vial")

        move([1.1672273874282837, -1.003362850551941,  1.5086296240436,
              -2.04012455562734,   -1.4970105330096644, -0.5064757505999964],  "On vial")
        gripper.move(140, 255, 255)  # grip

        move([1.1668643951416016, -1.1354245704463501, 1.1316335836993616,
              -1.5328094039908429, -1.4989655653582972, -0.5087154547320765],  "Raise vial")

        move([1.118363857269287,  -1.286565975551941,  1.377596680318014,
              -1.6786891422667445, -1.544044319783346,  -0.48208934465517217], "Above stirrer")

        move([1.118239164352417,  -1.210919664507248,  1.617370907460348,
              -1.9941722355284632, -1.5449407736407679, -0.48071128526796514], "Place on stirrer")
        gripper.move(0, 255, 255)  # release

        move([1.118363857269287,  -1.286565975551941,  1.377596680318014,
              -1.6786891422667445, -1.544044319783346,  -0.48208934465517217], "Retreat from stirrer")

        # Tell camera to start
        camera_should_record = True
        print(f"\n[ROBOT] Waiting {EXPERIMENT_DURATION_S}s for experiment...\n")
        time.sleep(EXPERIMENT_DURATION_S)

        # Tell camera to stop, then pick vial back up
        camera_should_stop = True

        move([1.118363857269287,  -1.286565975551941,  1.377596680318014,
              -1.6786891422667445, -1.544044319783346,  -0.48208934465517217], "Above stirrer")

        move([1.118239164352417,  -1.210919664507248,  1.617370907460348,
              -1.9941722355284632, -1.5449407736407679, -0.48071128526796514], "Grip from stirrer")
        gripper.move(140, 255, 255)  # grip

        move([1.118363857269287,  -1.286565975551941,  1.377596680318014,
              -1.6786891422667445, -1.544044319783346,  -0.48208934465517217], "Raise from stirrer")

        move([1.1668643951416016, -1.1354245704463501, 1.1316335836993616,
              -1.5328094039908429, -1.4989655653582972, -0.5087154547320765],  "Above rack")

        move([1.167259693145752,  -1.0337765973857422, 1.4777286688434046,
              -1.9788419208922328, -1.4967520872699183, -0.5067313353167933],  "Lower into rack")
        gripper.move(0, 255, 255)  # release

        move([1.447051763534546,  -1.5378911060145875, 1.5790107885943812,
              -1.6676823101439417, -1.5373862425433558, -0.32640201250185186], "Home")

    except Exception as e:
        print(f"[ROBOT] Error: {e}")
        camera_should_stop = True  # always save camera data even if robot crashes


# =============================================================================
# RUN EVERYTHING
# =============================================================================
print("Starting experiment...")
print(f"  Duration  : {EXPERIMENT_DURATION_S}s")
print(f"  Video     : {VIDEO_PATH}")
print(f"  CSV       : {CSV_PATH}")
print()

# Camera runs in background, robot runs in main thread
cam_thread = threading.Thread(target=run_camera, daemon=True)
cam_thread.start()

time.sleep(2)   # give camera time to open before robot starts moving
run_robot()

cam_thread.join(timeout=10)
print("\nAll done!")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT ENDS
# ═════════════════════════════════════════════════════════════════════════════