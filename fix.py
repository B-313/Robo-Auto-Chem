
#############
# 13th March
'''
FLOW:
for i in range(3):  # Vial 1,2,3
  Home → grab → stirrer
  sleep(180s)      # ← Reaction
  Record 190s + color track  # ← Per-vial MP4 + CSV
  Home → return vial

Vial 1: 180s react + 190s record = ~6.5min
Vial 2: same  
Vial 3: same  
Total: ~20min runtime
'''

#############
from PIL import ImageTk, Image
import numpy as np
import math
import os
import cv2 as cv
import csv
import sys
import time
current_dir = os.path.dirname(os.path.abspath(__file__))
from utils.UR_Functions import URfunctions as URControl
# sys.path.append(os.path.join(current_dir, 'robotiq'))
# current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(f"current_dir : {current_dir}")
# sys.path.append(current_dir)
from utils.robotiq_gripper import RobotiqGripper


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

####################
# START EXPERIMENT #
####################


#############
# Main loop #
#############

def main():
    robot = URControl(ip="192.168.0.2", port=30003)
    HOST = "192.168.0.2"
    PORT = 30003
    gripper=RobotiqGripper()
    gripper.connect("192.168.0.2", 63352)
#   gripper.move(255,125,125)
#   joint_state=degreestorad([93.77,-89.07,89.97,-90.01,-90.04,0.0])
#   robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
    # Gipper open/close
    # self.gripper.move(0,255,255) ## open
    # self.gripper.move(225,255,255) ## close
    
#############
# Camera setup #
#############

    base_name = '2026_03_13_traffic_light_experiment_trial'
    video_dir = "/home/robot/group_B/robo_chem_504/group_B_videos"
    color_names = ['None', 'Red', 'Yellow', 'Green']
    os.makedirs(video_dir, exist_ok=True)
    video_path = os.path.join(video_dir, f"{base_name}.mp4")

    #############
    # Per-vial camera setup
    #############
    cam = cv.VideoCapture(0)  # Your index
    if not cam.isOpened():
        print("Camera fail")
        return
    
    h, w = 1280, 720  # Or get from cam.read()
    target_fps = 60.0
    fourcc = cv.VideoWriter_fourcc(*'mp4v')
    
    # Reset for each vial
    prev_color = 0
    change_log = []

    '''
    prev_color = 0  # Start 'none'
    change_log = []  # List of changes
    start_time = time.time()
    
    cam = cv.VideoCapture(0)  # Camera index
    ret, frame = cam.read()
    if not ret:
        print("Could not read from camera")
        cam = None
    else:
        h, w = frame.shape[:2]
        fourcc = cv.VideoWriter_fourcc(*'mp4v')
        vid = cv.VideoWriter(video_path, fourcc, target_fps, (w, h))
    '''
    
    #############
    # Main loop #
    #############
    
    for i in range(1):
        
        # Goes to the home position every iteration
        joint_state=home_position
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        gripper.move(0,255,255) ## open
        
        # Reaches for the vial i
        joint_state=unreacted_approach_high[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        joint_state=unreacted_insert[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
    
        gripper.move(255,255,255) # close grip
        
        # Moves to the stirrer
        
        joint_state=unreacted_approach_low[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
    
        joint_state=unreacted_approach_high[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        joint_state=stirring_position_above
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        joint_state=stirring_position_on
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        ###############################
        # ROBOT STOPS MOVING - SLEEP 
        ###############################
        # time.sleep(10.00)      
        ###############################
        # CAMERA RECORDS HERE 
        ###############################
        
        if cam is not None:
            # NEW VIDEO WRITER PER VIAL
            video_path = os.path.join(video_dir, f"{base_name}_vial{i+1}.mp4")
            csv_path = os.path.join(video_dir, f"{base_name}_vial{i+1}_changes.csv")
            
            vid = cv.VideoWriter(video_path, fourcc, target_fps, (w, h))
            
            record_seconds = 1
            num_frames = int(target_fps * record_seconds)
            
            vial_start_time = time.time()  # Vial-specific timing
            prev_color = 0  # Reset per vial
            
            '''
            for frame_idx in range(num_frames):
                ret, frame = cam.read()  # ← READ FIRST!
                if not ret:
                    break
            '''
                
                vid.write(frame)
                
                ##################
                # Color detection 
                ##################
                hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)  # ← Now frame exists
                frame_time = time.time() - vial_start_time  # Use vial_start_time!

                # Check each color threshold (adjust these!)
                color_pixels = {
                    'red':    cv.countNonZero(cv.inRange(hsv, np.array([0,50,50]), np.array([10,255,255]))),
                    'yellow': cv.countNonZero(cv.inRange(hsv, np.array([20,50,50]), np.array([35,255,255]))),
                    'green':  cv.countNonZero(cv.inRange(hsv, np.array([40,50,50]), np.array([90,255,255])))
                }
                
                pixel_threshold = 5000  # Min pixels to call "present"
                current_color = None
                for color, pixels in color_pixels.items():
                    if pixels > pixel_threshold:
                        current_color = color
                        break  # First color that meets threshold
                
                # Log ONLY if color changes (to/from)
                if current_color and current_color != color_names[prev_color]:
                    change_log.append({
                        'vial': i+1,
                        'time_s': round(frame_time, 1),
                        'from_color': color_names[prev_color],
                        'to_color': current_color.capitalize(),  # 'yellow' → 'Yellow'
                        'pixels': color_pixels[current_color]
                    })
                    # Simple ID: 0=None,1=Red,2=Yellow,3=Green
                    prev_color = {'red':1, 'yellow':2, 'green':3}.get(current_color, 0)

                
                # Annotate frame
                cv.putText(frame, f"{current_color or 'none'}: {color_pixels.get(current_color,0)}px", 
                          (10, 30), cv.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)
            
            # Close THIS vial's video
            vid.release()
             
            # Save THIS vial's CSV
            if change_log:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=['vial','time_s','from_color','to_color','pixels'])
                    writer.writeheader()
                    writer.writerows(change_log)
                print(f"Vial {i+1} saved: {video_path}, {csv_path}")


        '''
        
        if cam is not None:
            record_seconds = 190
            num_frames = int(target_fps * record_seconds)
            
            for frame_idx in range(num_frames):
                ret, frame = cam.read()
                if not ret:
                    break
                
                vid.write(frame)
        

            ##################
            # Save CSV after each vial (or once at end)
            ##################
            if change_log:
                with open(csv_path, 'w', newline='') as f:
                    writer = csv.DictWriter(f, fieldnames=change_log[0].keys())
                    writer.writeheader()
                    writer.writerows(change_log)
                print(f"Changes logged: {csv_path}")

        '''
        
        ################
        # ROBOT RESUMES 
        ################
        
        # Goes home and then returns the vial
        
        joint_state=home_position
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        joint_state=unreacted_approach_high[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        joint_state=unreacted_approach_low[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        joint_state=unreacted_insert[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
        
        gripper.move(0,255,255)
        
        joint_state=unreacted_approach_high[i]
        robot.move_joint_list(joint_state, 0.5, 0.5, 0.02)
   
    #################
    # CAMERA CLEANUP #
    #################
    if cam is not None:
        cam.release()
        vid.release()
        cv.destroyAllWindows()
        print(f"Video saved: {video_path}")
    
    
####################
# END EXPERIMENT #
####################

 
def degreestorad(list):
     for i in range(6):
          list[i]=list[i]*(math.pi/180)
     return(list)    
 

if __name__=="__main__":
     main()
