################
# FULL ROUTINE #
################

# Current Routine: 
#   Pick up vial
#   Place vial on stirrer 
#   Camera takes one photo 
#   Sleep 1s
#   Record 150s + Stir 60s
#   Pick up Vial
#
# Loop 4 times for 4 vials

##########
# CONFIG #
##########
ROBOT_IP       = "192.168.0.2" #HOST
GRIPPER_PORT   = 63352 #PORT
PLATE_PORT     = "/dev/ttyACM0" #STIR PLATE PORT
VIDEO_DIR      = "/home/robot/group_B/robo_chem_504/group_B_videos"
RECORD_SECONDS = 190
STIR_SECONDS   = 15
STIR_RPM       = 1300
NUM_VIALS      = 4

###########
# IMPORTS #
###########
import cv2 as cv
import os, sys, time, math, threading
from utils.UR_Functions import URfunctions as URControl
from utils.robotiq_gripper import RobotiqGripper
current_dir = os.path.dirname(os.path.abspath(__file__))

# sys.path.append(os.path.join(current_dir, 'robotiq'))
# current_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# print(f"current_dir : {current_dir}")
# sys.path.append(current_dir)

# Stirring plate libraries
from stirring_plate import IKADriver

#############
# POSITIONS #
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

reacted_insert = [ # updated
    [1.755118489265442, -1.622995515862936, 2.3268495241748255, -2.300579687158102, -1.5604632536517542, 0.15620261430740356],
    [1.7368957996368408, -1.5046419662288208, 2.2058165709124964, -2.29784693340444, -1.5600035826312464, 0.1383223533630371],
    [1.8499835729599, -1.508009986286499, 2.2097938696490687, -2.29905428508901, -1.5630114714251917, 0.2513377368450165],
    [1.8210809230804443, -1.3954226535609742, 2.0836508909808558, -2.285426279107565, -1.5622661749469202, 0.22260428965091705]
]

reacted_approach = [1.757645845413208, -1.6042686901488246, 2.0646565596209925, -2.0572530231871546, -1.559568230305807, 0.15809589624404907]

#reacted_approach_low = [
#    [1.7736990451812744, -1.716959138909811, 2.4054392019854944, -2.2853337726988734, -1.5608862082110804, 0.1745983064174652],
#    [1.7508206367492676, -1.5886441669859828, 2.296417538319723, -2.3044830761351527, -1.5603516737567347, 0.1520262062549591],
#    [1.8768328428268433, -1.5930525265135707, 2.2924214045154017, -2.296755929986471, -1.563714329396383, 0.2780013084411621],
#    [1.8435415029525757, -1.483963669543602, 2.179882828389303, -2.293112417260641, -1.5628193060504358, 0.2449340671300888]
#]
#
#reacted_approach_high = [
#    [1.7739837169647217, -1.8434292278685511, 2.2804930845843714, -2.033940454522604, -1.5599435011493128, 0.1738280951976776],
#    [1.751039981842041, -1.6982251606383265, 2.1973212401019495, -2.095750471154684, -1.5595305601703089, 0.15137901902198792],
#    [1.877051830291748, -1.7067915401854457, 2.1848610083209437, -2.075393339196676, -1.562838379536764, 0.2773132920265198],
#    [1.8437175750732422, -1.5870100460448207, 2.0843218008624476, -2.0945011578001917, -1.5619915167437952, 0.24426542222499847]
#]

# stir plate
stirring_position_on = [1.118239164352417, -1.210919664507248, 1.617370907460348, -1.9941722355284632, -1.5449407736407679, -0.48071128526796514]
stirring_position_above = [1.118363857269287, -1.286565975551941, 1.377596680318014, -1.6786891422667445, -1.544044319783346, -0.48208934465517217]


##################
# CAMERA / PHOTO #
##################
def take_photo(vial_number, camera_index=0):
    output_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/batch1_reaction_1_vial_{vial_number+1}_photo.jpg"
    cap = cv.VideoCapture(camera_index, cv.CAP_V4L2)
    ret, frame = cap.read()
    if ret:
        cv.imwrite(output_path, frame)
        print(f"Photo saved: {output_path}")
    cap.release()

def basic_recorder(vial_number, camera_index=0, width=640, height=480, fps=30, record_seconds=180):
    output_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/batch1_reaction_1_vial_{vial_number+1}.mp4"
    os.makedirs(os.path.dirname(output_path), exist_ok=True)  # ensure dir exists
    cap = cv.VideoCapture(camera_index, cv.CAP_V4L2)
    cap.set(cv.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv.CAP_PROP_FPS, fps)

    actual_width = int(cap.get(cv.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv.CAP_PROP_FRAME_HEIGHT))
    print(f"Recording vial {vial_number+1} at {actual_width}x{actual_height} for {record_seconds}s → {output_path}")

    out = cv.VideoWriter(output_path, cv.VideoWriter_fourcc(*'mp4v'), fps, (actual_width, actual_height))

    start_time = time.time()
    while time.time() - start_time < record_seconds:
        ret, frame = cap.read()
        if not ret:
            print("Frame capture failed.")
            break
        out.write(frame)

    cap.release()
    out.release()
    print(f"Saved: {output_path}")
    
###########
# STIRRER #
###########

def connect_plate(port):
    try:
        plate = IKADriver(port)
        print(f"Stirring plate connected on {port}")
        return plate
    except Exception as e:
        print(f"Stirring plate init failed: {e}")
        return None
    
def stir_then_stop(plate, rpm, stir_seconds):
    # Runs in background thread — stirs for stir_seconds then stops
    if plate is None:
        return
    plate.setStir(rpm)
    plate.startStir()
    time.sleep(stir_seconds)
    plate.stopStir()

##################
# ORCHESTRATION  #
##################

def record_with_stir(plate, vial_number, rpm, total_seconds=180, stir_seconds=60):
    # Start stir in background
    stir_thread = threading.Thread(target=stir_then_stop, args=(plate, rpm, stir_seconds))
    stir_thread.start()
    
    # Record full duration in main thread
    basic_recorder(vial_number, record_seconds=total_seconds)
    
    stir_thread.join()


def run_vial(i, robot, gripper, plate):
    # Pick vial
    robot.move_joint_list(unreacted_approach_high[i], 0.5, 0.5, 0.02)
    robot.move_joint_list(unreacted_insert[i], 0.5, 0.5, 0.02)
    gripper.move(170, 255, 255) # close gripper

    # Move to stirrer
    robot.move_joint_list(unreacted_approach_low[i], 0.5, 0.5, 0.02)
    robot.move_joint_list(unreacted_approach_high[i], 0.5, 0.5, 0.02)
    robot.move_joint_list(stirring_position_above, 0.5, 0.5, 0.02)
    robot.move_joint_list(stirring_position_on, 0.5, 0.5, 0.02)

    # Photo + record
    #take_photo(i)
    #record_with_stir(plate, i, rpm=STIR_RPM, total_seconds=RECORD_SECONDS, stir_seconds=STIR_SECONDS)

    # Return vial
    robot.move_joint_list(home_position, 0.5, 0.5, 0.02)
    #robot.move_joint_list(unreacted_approach_high[i], 0.5, 0.5, 0.02)
    #robot.move_joint_list(unreacted_approach_low[i], 0.5, 0.5, 0.02)
    robot.move_joint_list(reacted_approach, 0.5, 0.5, 0.02)
    robot.move_joint_list(reacted_insert[i], 0.5, 0.5, 0.02)
    gripper.move(0, 255, 255) # open gripper
    robot.move_joint_list(unreacted_approach_high[i], 0.5, 0.5, 0.02)

manual_vial = 3

########
# MAIN #
########
def main():
    robot  = URControl(ip=ROBOT_IP)
    gripper = RobotiqGripper(); gripper.connect(ROBOT_IP, GRIPPER_PORT)
    plate  = connect_plate(PLATE_PORT)

    for i in range(NUM_VIALS):
        run_vial(i, robot, gripper, plate)
    #robot.move_joint_list(unreacted_approach_high[manual_vial], 0.5, 0.5, 0.02)
    #robot.move_joint_list(unreacted_approach_low[manual_vial], 0.5, 0.5, 0.02)
    #robot.move_joint_list(unreacted_insert[manual_vial], 0.5, 0.5, 0.02) 
    #gripper.move(170, 255, 255) # close gripper
    #gripper.move(0,255,255) ## open

    robot.move_joint_list(home_position, 0.5, 0.5, 0.02)

if __name__ == "__main__":
    main()
