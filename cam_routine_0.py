"""
robo_chem_combined.py
=====================
Robot arm + colour-change camera logger.

Workflow:
  1. Robot picks vial from rack
  2. Robot places vial on stirrer  ← camera starts recording HERE
  3. Human turns stirrer on manually
  4. Camera watches GREEN → RED → YELLOW cycles
  5. Robot picks vial up from stirrer  ← camera stops recording HERE
  6. Robot returns vial to rack
  7. Video + CSV saved

Usage:
    python3 robo_chem_combined.py
"""

import cv2 as cv
import numpy as np
from datetime import datetime
import os
import sys
import time
import csv
import threading

current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

from utils.UR_Functions import URfunctions as URControl
from utils.robotiq_gripper import RobotiqGripper


# ═════════════════════════════════════════════════════════════════════════════
# SHARED STATE  (thread-safe)
# ═════════════════════════════════════════════════════════════════════════════
state_lock = threading.Lock()
shared = {
    "camera_active": False,   # robot sets True when vial is on stirrer
    "stop_camera":   False,   # robot sets True when picking vial back up
    "robot_step":    "IDLE",
}

def set_robot_step(label: str):
    with state_lock:
        shared["robot_step"] = label
    print(f"[ROBOT] ── {label}")


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA CONFIG
# ═════════════════════════════════════════════════════════════════════════════
BASE_NAME     = 'traffic_light_experiment'
VIDEO_DIR     = "/home/robot/group_B/robo_chem_504/group_B_videos"
VIDEO_PATH    = os.path.join(VIDEO_DIR, f"{BASE_NAME}.mp4")
CSV_PATH      = os.path.join(VIDEO_DIR, f"{BASE_NAME}_changes.csv")
TARGET_FPS    = 10.0
WIDTH, HEIGHT = 1280, 720
CAMERA_INDEX  = 0
ROI           = [450, 250, 750, 450]   # [x1, y1, x2, y2] — update from tuner!


# ═════════════════════════════════════════════════════════════════════════════
# CAMERA THREAD
# ═════════════════════════════════════════════════════════════════════════════
def camera_thread_fn():
    os.makedirs(VIDEO_DIR, exist_ok=True)

    cap = cv.VideoCapture(CAMERA_INDEX)
    cap.set(cv.CAP_PROP_FRAME_WIDTH,  WIDTH)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, HEIGHT)

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out    = cv.VideoWriter(VIDEO_PATH, fourcc, TARGET_FPS, (WIDTH, HEIGHT))

    change_log  = []
    csv_headers = ['timestamp', 'elapsed_s', 'frame',
                   'from_state', 'to_state', 'B', 'G', 'R']

    current_state = "NONE"
    recording     = False
    start_time    = None
    frame_count   = 0

    print("[CAM]  Camera ready — waiting for vial to be placed on stirrer...")

    while True:
        with state_lock:
            should_start = shared["camera_active"]
            should_stop  = shared["stop_camera"]

        # ── Wait for robot to place vial on stirrer ──────────────────────────
        if not recording:
            if should_start:
                recording  = True
                start_time = time.time()
                print("[CAM]  🟢 Vial on stirrer — recording started.")
                print("[CAM]  ⚠️  Please turn the stirrer ON now.")
            else:
                time.sleep(0.05)
                continue

        # ── Robot picked vial back up — stop recording ───────────────────────
        if should_stop:
            print("[CAM]  🔴 Vial removed from stirrer — stopping recording.")
            break

        ret, frame = cap.read()
        if not ret:
            time.sleep(0.05)
            continue

        # ── ROI colour detection ─────────────────────────────────────────────
        x1, y1, x2, y2 = ROI
        roi = frame[y1:y2, x1:x2]
        b, g, r = np.mean(roi, axis=(0, 1)).astype(int)

        # Indigo carmine: green is teal so blue channel is also elevated
        new_state = "NONE"
        if   g > r + 15 and (g + b) > (r * 2): new_state = "GREEN"
        elif r > g + 25 and r > b + 25:         new_state = "RED"
        elif r > b + 15 and g > b + 15:         new_state = "YELLOW"

        # ── Log on change ────────────────────────────────────────────────────
        if new_state != current_state and new_state != "NONE":
            timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
            elapsed   = round(time.time() - start_time, 1)
            change_log.append([
                timestamp, elapsed, frame_count,
                current_state, new_state,
                int(b), int(g), int(r)
            ])
            icon = {"GREEN": "🟢", "RED": "🔴", "YELLOW": "🟡"}.get(new_state, "⚪")
            print(f"[CAM]  {icon} {timestamp} (+{elapsed}s): "
                  f"{current_state} → {new_state}  (B{b} G{g} R{r})")
            current_state = new_state

        # ── Overlay on frame ─────────────────────────────────────────────────
        elapsed_disp = round(time.time() - start_time, 1)
        icon_disp = {"GREEN": "GRN", "RED": "RED", "YELLOW": "YEL"}.get(current_state, "---")

        cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
        cv.putText(frame, f"State: {icon_disp}  B{b} G{g} R{r}",
                   (20, 40),  cv.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
        cv.putText(frame, f"Time: {elapsed_disp}s  Changes: {len(change_log)}",
                   (20, 75),  cv.FONT_HERSHEY_SIMPLEX, 0.6, (0, 200, 255), 2)

        out.write(frame)
        frame_count += 1

        cv.imshow('Traffic Light Logger', frame)
        if cv.waitKey(1) & 0xFF == ord('q'):
            break

    # ── Save outputs ─────────────────────────────────────────────────────────
    cap.release()
    out.release()
    cv.destroyAllWindows()

    with open(CSV_PATH, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_headers)
        writer.writerows(change_log)

    print(f"\n[CAM]  ✅ Complete. {len(change_log)} colour changes recorded.")
    print(f"[CAM]  📹 {VIDEO_PATH}")
    print(f"[CAM]  📝 {CSV_PATH}")


# ═════════════════════════════════════════════════════════════════════════════
# ROBOT POSITIONS
# ═════════════════════════════════════════════════════════════════════════════
ROBOT_IP   = "192.168.0.2"
ROBOT_PORT = 30003

stirring_position_on    = [1.118239164352417,  -1.210919664507248,  1.617370907460348,
                            -1.9941722355284632, -1.5449407736407679, -0.48071128526796514]
stirring_position_above = [1.118363857269287,  -1.286565975551941,  1.377596680318014,
                            -1.6786891422667445, -1.544044319783346,  -0.48208934465517217]


# ═════════════════════════════════════════════════════════════════════════════
# ROBOT THREAD
# ═════════════════════════════════════════════════════════════════════════════
def robot_thread_fn():
    try:
        robot   = URControl(ip=ROBOT_IP, port=ROBOT_PORT)
        gripper = RobotiqGripper()
        gripper.connect(ROBOT_IP, 63352)

        def move(js, label, v=0.5, a=0.5, r=0.02):
            set_robot_step(label)
            robot.move_joint_list(js, v, a, r)

        # ── 1. Initialise gripper ─────────────────────────────────────────────
        gripper.move(255, 125, 125)

        # ── 2. Home ───────────────────────────────────────────────────────────
        move([1.447051763534546,  -1.5378911060145875, 1.5790107885943812,
              -1.6676823101439417, -1.5373862425433558, -0.32640201250185186],
             "HOME")
        gripper.move(0, 255, 255)

        # ── 3. Pick vial from rack ────────────────────────────────────────────
        move([1.167259693145752,  -1.0337765973857422, 1.4777286688434046,
              -1.9788419208922328, -1.4967520872699183, -0.5067313353167933],
             "ABOVE_VIAL")
        gripper.move(0, 255, 255)

        move([1.1672273874282837, -1.003362850551941,  1.5086296240436,
              -2.04012455562734,   -1.4970105330096644, -0.5064757505999964],
             "ON_VIAL")
        gripper.move(140, 255, 255)   # grip vial

        move([1.1668643951416016, -1.1354245704463501, 1.1316335836993616,
              -1.5328094039908429, -1.4989655653582972, -0.5087154547320765],
             "RAISE_VIAL")

        # ── 4. Place vial on stirrer ──────────────────────────────────────────
        move(stirring_position_above, "ABOVE_STIRRER")
        move(stirring_position_on,    "PLACE_ON_STIRRER")
        gripper.move(0, 255, 255)     # release vial

        # ── 5. Retreat and signal camera to START ─────────────────────────────
        move(stirring_position_above, "RETREAT_FROM_STIRRER")

        with state_lock:
            shared["camera_active"] = True
            shared["robot_step"]    = "WAITING — STIRRER RUNNING"

        print("[ROBOT] Vial placed. Camera is now recording.")
        print("[ROBOT] ⚠️  Turn the stirrer ON now.")
        print(f"[ROBOT] Waiting {EXPERIMENT_DURATION_S}s for experiment...")

        # ── 6. Hold while experiment runs ─────────────────────────────────────
        #       Adjust EXPERIMENT_DURATION_S below to match your experiment.
        time.sleep(EXPERIMENT_DURATION_S)

        # ── 7. Signal camera to STOP, then pick vial back up ─────────────────
        with state_lock:
            shared["stop_camera"] = True

        move(stirring_position_above, "ABOVE_STIRRER_PICKUP")
        move(stirring_position_on,    "GRIP_FROM_STIRRER")
        gripper.move(140, 255, 255)   # grip vial

        # ── 8. Return vial to rack ────────────────────────────────────────────
        move(stirring_position_above, "RAISE_FROM_STIRRER")

        move([1.1668643951416016, -1.1354245704463501, 1.1316335836993616,
              -1.5328094039908429, -1.4989655653582972, -0.5087154547320765],
             "ABOVE_RACK")

        move([1.167259693145752,  -1.0337765973857422, 1.4777286688434046,
              -1.9788419208922328, -1.4967520872699183, -0.5067313353167933],
             "LOWER_INTO_RACK")
        gripper.move(0, 255, 255)     # release vial into rack

        # ── 9. Home ───────────────────────────────────────────────────────────
        move([1.447051763534546,  -1.5378911060145875, 1.5790107885943812,
              -1.6676823101439417, -1.5373862425433558, -0.32640201250185186],
             "HOME")

    except Exception as e:
        print(f"[ROBOT] ❌ Error: {e}")
        with state_lock:
            shared["stop_camera"] = True   # always save camera data on crash

    finally:
        print("[ROBOT] ✅ Sequence complete.")


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT DURATION — adjust to match your experiment
# ═════════════════════════════════════════════════════════════════════════════
EXPERIMENT_DURATION_S = 180   # seconds (3 min). Robot waits this long on stirrer.

# ═════════════════════════════════════════════════════════════════════════════
# MAIN
# ═════════════════════════════════════════════════════════════════════════════
def main():
    print("=" * 55)
    print("  Traffic Light Experiment — Robo Chem 504")
    print("=" * 55)
    print(f"  Video  → {VIDEO_PATH}")
    print(f"  CSV    → {CSV_PATH}")
    print(f"  Duration → {EXPERIMENT_DURATION_S}s on stirrer")
    print("=" * 55)
    print()

    cam_thread   = threading.Thread(target=camera_thread_fn, daemon=True)
    robot_thread = threading.Thread(target=robot_thread_fn,  daemon=False)

    cam_thread.start()
    time.sleep(2)          # camera initialises before robot moves
    robot_thread.start()
    robot_thread.join()    # wait for full robot sequence to finish

    cam_thread.join(timeout=10)
    print("\n[MAIN] All done.")


if __name__ == "__main__":
    main()


# ═════════════════════════════════════════════════════════════════════════════
# EXPERIMENT ENDS — CSV Logs and Video Saved
# ═════════════════════════════════════════════════════════════════════════════