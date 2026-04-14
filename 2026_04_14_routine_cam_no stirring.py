
################
# FULL ROUTINE #
################


######################
# IMPORTS & SETTINGS 
######################

from PIL import ImageTk, Image
import numpy as np
import math
import os
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
from utils.UR_Functions import URfunctions as URControl
# sys.path.append(os.path.join(current_dir, 'robotiq'))
# current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(f"current_dir : {current_dir}")
# sys.path.append(current_dir)
from utils.robotiq_gripper import RobotiqGripper

import threading
import cv2 as cv
from datetime import datetime
import csv

from color_detection_module import detect_colour_in_frame
from roi_color_detection_module import detect_colour_in_roi, load_roi_from_json
from stir_session_module import connect_plate, run_stir_session

##################
# VIDEO SETTINGS #
##################

video_name = '14 Apr vials.mp4'

fps = 30.0
delay = 30.0 / fps
width = 1280
height = 720
camera_index = 0
record_seconds = 120   # recording duration

##################
# STIRRING PLATE SETTINGS #
##################

STIR_PORT = "/dev/ttyACM0"
STIR_RPM = 700
STIR_TEMP = 25    # degrees Celsius; set to None to disable heating
STIR_SECONDS = 30

# 1280x720 is HD

def basic_recorder(vial_number, camera_index=0, width=640, height=480, fps=30, record_seconds=10):
    # Path for saving the video
    output_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/{vial_number+1}.mp4"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # Open the camera
    cap = cv.VideoCapture(camera_index, cv.CAP_V4L2)  # Try V4L2 if GStreamer doesn't work well

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
    fourcc = cv.VideoWriter_fourcc(*'mp4v')  # Or use 'XVID'/'MJPG' if necessary
    out = cv.VideoWriter(output_path, fourcc, fps, (int(actual_width), int(actual_height)))

    # Start recording
    print(f"Recording for {record_seconds} seconds... Vial {vial_number + 1}")

    start_time = time.time()

    while True:
        ret, frame = cap.read()

        # If frame is not successfully captured, exit
        if not ret:
            print("Can't receive frame. Exiting...")
            break

        # Write the frame to the video file
        out.write(frame)

        # Stop after the specified recording duration
        if time.time() - start_time > record_seconds:
            break

    # Release everything once the recording is done
    cap.release()
    out.release()
    cv.destroyAllWindows()

    print(f"Video saved to: {output_path}")


def color_detection_recorder(
    vial_number,
    camera_index=0,
    width=640,
    height=480,
    fps=30,
    record_seconds=600,   # hard cap safety limit in seconds
    min_pixels=800,
    stop_event=None,      # set() this externally to stop recording
):
    output_dir = "/home/robot/group_B/robo_chem_504/group_B_videos"
    video_path = f"{output_dir}/{vial_number+1}.mp4"
    csv_path = f"{output_dir}/{vial_number+1}_colour_changes.csv"

    os.makedirs(output_dir, exist_ok=True)

    cap = cv.VideoCapture(camera_index, cv.CAP_V4L2)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    out = cv.VideoWriter(video_path, fourcc, fps, (actual_width, actual_height))

    print(f"Recording with colour detection for {record_seconds} seconds... Vial {vial_number + 1}")

    ##########################
    # ROI DETECTION - START
    ##########################
    roi = None
    try:
        roi = load_roi_from_json("roi_config.json")
        print(f"ROI loaded: {roi}")
    except Exception as e:
        print(f"ROI config not loaded, using full frame detection: {e}")
    ########################
    # ROI DETECTION - END
    ########################

    current_state = "none"
    start_time = time.time()
    frame_index = 0

    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["timestamp", "frame", "elapsed_s", "from_state", "to_state", "red", "yellow", "green"])

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame. Exiting...")
                break

            out.write(frame)

            ##########################
            # ROI DETECTION - START
            ##########################
            if roi is not None:
                detection = detect_colour_in_roi(frame, roi, min_pixels=min_pixels)
                counts = detection["counts"]
                new_state = detection["state"]
                x1, y1, x2, y2 = detection["roi"]
                cv.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), 2)
            else:
                counts = detect_colour_in_frame(frame)
                dominant_colour = max(counts, key=counts.get)
                dominant_pixels = counts[dominant_colour]
                new_state = dominant_colour if dominant_pixels >= min_pixels else "none"
            ########################
            # ROI DETECTION - END
            ########################

            if new_state != current_state:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                elapsed = round(time.time() - start_time, 3)
                writer.writerow([
                    timestamp,
                    frame_index,
                    elapsed,
                    current_state,
                    new_state,
                    counts["red"],
                    counts["yellow"],
                    counts["green"],
                ])
                current_state = new_state

            frame_index += 1
            if stop_event is not None and stop_event.is_set():
                break
            if time.time() - start_time > record_seconds:
                break

    cap.release()
    out.release()
    cv.destroyAllWindows()

    print(f"Video saved to: {video_path}")
    print(f"Colour change log saved to: {csv_path}")

#if __name__ == "__main__":
#    basic_recorder()

#############
# Positions #
#############

home_position = [1.3193929195404053, -1.6615268192686976, 1.6442220846759241, -1.57421936611318, -1.547626797352926, -0.282325569783346]

# Before reacting

unreacted_approach_high = [
    [1.702911138534546, -1.2955393356135865, 1.7377761046039026, -2.0385571918883265, -1.5581796805011194, 0.10387864708900452],
    [1.695432424545288, -1.175900177364685, 1.5821941534625452, -2.002573629418844, -1.5579708258258265, 0.0965086817741394],
    [1.771836519241333, -1.2054326397231598, 1.6055048147784632, -1.996965070764059, -1.5599125067340296, 0.17283010482788086],
    [1.7598854303359985, -1.0811243814281006, 1.4432695547686976, -1.958919187585348, -1.5596111456500452, 0.16096806526184082]
]

unreacted_approach_low = [
    [1.7027943134307861, -1.219853715305664, 1.8187602202044886, -2.195282121697897, -1.5588277021991175, 0.10447625815868378],
    [1.6953520774841309, -1.1113875967315217, 1.6564257780658167, -2.1413375339903773, -1.5585158506976526, 0.09700823575258255],
    [1.7717715501785278, -1.1369521182826539, 1.6863048712359827, -2.1462160549559535, -1.5605314413653772, 0.17337778210639954],
    [1.759840965270996, -1.023309664135315, 1.5164054075824183, -2.089883943597311, -1.5601356665240687, 0.16147497296333313]
]

unreacted_insert = [
    [1.7027702331542969, -1.1989212197116395, 1.8344619909869593, -2.2318717441954554, -1.5589988867389124, 0.10454279184341430],
    [1.6953003406524658, -1.0928057891181489, 1.6714962164508265, -2.1750599346556605, -1.5586908499347132, 0.09715914726257324],
    [1.7716856002807617, -1.1152427357486268, 1.703991715108053, -2.185610910455221, -1.5607073942767542, 0.17350609600543976],
    [1.7597777843475342, -1.005433515911438, 1.5321853796588343, -2.123598714868063, -1.5602858702289026, 0.161591038107872]
]

# After reacting

reacted_insert = [
    [1.7736680507659912, -1.7024322948851527, 2.415126625691549, -2.309547563592428, -1.5609906355487269, 0.17467042803764343],
    [1.7508153915405273, -1.570087658449058, 2.3082061449633997, -2.3348666630186976, -1.5605104605304163, 0.15212920308113098],
    [1.8768088817596436, -1.5697949928329145, 2.307462278996603, -2.3350564442076625, -1.5638979117022913, 0.2781129479408264],
    [1.8435537815093994, -1.460801677112915, 2.194772545491354, -2.3312183819212855, -1.5629757086383265, 0.24502253532409668]
]

reacted_approach_low = [
    [1.7736990451812744, -1.716959138909811, 2.4054392019854944, -2.2853337726988734, -1.5608862082110804, 0.1745983064174652],
    [1.7508206367492676, -1.5886441669859828, 2.296417538319723, -2.3044830761351527, -1.5603516737567347, 0.1520262062549591],
    [1.8768328428268433, -1.5930525265135707, 2.2924214045154017, -2.296755929986471, -1.563714329396383, 0.2780013084411621],
    [1.8435415029525757, -1.483963669543602, 2.179882828389303, -2.293112417260641, -1.5628193060504358, 0.2449340671300888]
]

reacted_approach_high = [
    [1.7739837169647217, -1.8434292278685511, 2.2804930845843714, -2.033940454522604, -1.5599435011493128, 0.1738280951976776],
    [1.751039981842041, -1.6982251606383265, 2.1973212401019495, -2.095750471154684, -1.5595305601703089, 0.15137901902198792],
    [1.877051830291748, -1.7067915401854457, 2.1848610083209437, -2.075393339196676, -1.562838379536764, 0.2773132920265198],
    [1.8437175750732422, -1.5870100460448207, 2.0843218008624476, -2.0945011578001917, -1.5619915167437952, 0.24426542222499847]
]

# stir plate
stirring_position_on = [1.118239164352417, -1.210919664507248, 1.617370907460348, -1.9941722355284632, -1.5449407736407679, -0.48071128526796514]
stirring_position_above = [1.118363857269287, -1.286565975551941, 1.377596680318014, -1.6786891422667445, -1.544044319783346, -0.48208934465517217]

#########
# START #
#########

HOST = "192.168.0.2"
PORT = 30003
# from robotiq.robotiq_gripper import RobotiqGripper
def main():
    robot = URControl(ip="192.168.0.2", port=30003)
    HOST = "192.168.0.2"
    PORT = 30003
    gripper=RobotiqGripper()
    gripper.connect("192.168.0.2", 63352)

    # Stirring plate init (one-time); if unavailable, routine continues without stirring control.
    plate = connect_plate(STIR_PORT)
#   gripper.move(255,125,125)
#     joint_state=degreestorad([93.77,-89.07,89.97,-90.01,-90.04,0.0])
#     robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
    
       
    # Gipper open/close
    # self.gripper.move(0,255,255) ## open
    # self.gripper.move(225,255,255) ## close

    
    
    #home
    
    # Go to the home position before the loop
    joint_state=home_position
    robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
    gripper.move(0,255,255) ## open
    
    # Pick vials, put on stirr and place back to the plate
    for i in range(4):
        
        # Reaches for the vial i
        joint_state=unreacted_approach_high[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        joint_state=unreacted_insert[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
    
        gripper.move(170,255,255) # close grip
        
        # Moves to the stirrer
        
        joint_state=unreacted_approach_low[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
    
        joint_state=unreacted_approach_high[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        joint_state=stirring_position_above
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        joint_state=stirring_position_on
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        ############################
        # STIRRING + CAMERA - START
        ############################
        # Camera starts when vial is placed. Stirrer runs then stops.
        # Recording continues until gripper picks vial back up.
        stop_recording = threading.Event()

        recorder_thread = threading.Thread(
            target=color_detection_recorder,
            kwargs=dict(
                vial_number=i,
                camera_index=camera_index,
                width=width,
                height=height,
                fps=fps,
                stop_event=stop_recording,
            ),
            daemon=True,
        )
        recorder_thread.start()

        # Stirrer heats and stirs for STIR_SECONDS then stops.
        run_stir_session(plate, STIR_RPM, STIR_SECONDS, temp=STIR_TEMP)
        ############################
        # STIRRING + CAMERA - END
        ############################

        # Goes home and then returns the vial

        joint_state=home_position
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)

        joint_state=unreacted_approach_high[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)

        joint_state=unreacted_approach_low[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)

        joint_state=unreacted_insert[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)

        gripper.move(0,255,255)   # vial released back in rack
        stop_recording.set()       # vial removed - stop camera
        recorder_thread.join(timeout=10)

        joint_state=unreacted_approach_high[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
   
    
    
    
#######
# END #
#######




############
############


def degreestorad(list):
     for i in range(6):
          list[i]=list[i]*(math.pi/180)
     return(list)    
 

if __name__=="__main__":
     main()