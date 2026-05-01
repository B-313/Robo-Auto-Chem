################
# FULL ROUTINE #
################

# Current Routine: Place vial on stirrer → Camera takes one photo → Record 150s → Stir 60s → Pick up Vial → Loop 4 times for 4 vials

##########
# CONFIG #
##########
ROBOT_IP       = "192.168.0.2" #HOST
GRIPPER_PORT   = 63352 #PORT
PLATE_PORT     = "/dev/ttyACM0" #STIR PLATE PORT
VIDEO_DIR      = "/home/robot/group_B/robo_chem_504/group_B_videos"
RECORD_SECONDS = 190
STIR_SECONDS   = 20
STIR_RPM       = 1500
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

exhauted_approach_high = [[1.4711607694625854, -1.729419847527975, 2.3005879561053675, -2.1649967632689417, -1.5527351538287562, -0.12832719484438115],
                          [1.461242914199829, -1.826648851434225, 2.3842182795154017, -2.1513544521727503, -1.5524557272540491, -0.13839942613710576],
                          [1.886932611465454, -1.6541978321471156, 2.006794277821676, -1.949963232079977, -1.5626638571368616, 0.28678539395332336],
                          [1.9119048118591309, -1.7283650837340296, 2.0814393202411097, -1.9505115948119105, -1.5633629004107874, 0.3116168975830078]
    
]

exhauted_approach_low = [[1.471045732498169, -1.6876474819579066, 2.336992327366964, -2.2431241474547328, -1.5530813376056116, -0.12810308138002569],
                         [1.4611554145812988, -1.782081743279928, 2.4229098002063196, -2.2346278629698695, -1.5528138319598597, -0.1381438414203089],
                         [1.8866541385650635, -1.5190337349525471, 2.175055805836813, -2.253425260583395, -1.5637877623187464, 0.2878590226173401],
                         [1.9116365909576416, -1.5874926052489222, 2.2529290358172815, -2.262961050073141, -1.5645421187030237, 0.3126955032348633]
                         
]

exhauted_insert = [
    [1.470982313156128, -1.6536113224425257, 2.361356560383932, -2.3016063175597132, -1.5533021132098597, -0.12793237367738897],
    [1.4610836505889893, -1.7513462505736292, 2.4449361006366175, -2.2873412571349085, -1.5529988447772425, -0.13799745241274053],
    [1.8866300582885742, -1.4920480784824868, 2.1945555845843714, -2.2998458347716273, -1.564014736806051, 0.2880030870437622],
    [1.9115538597106934, -1.5526053880206128, 2.277067009602682, -2.321963449517721, -1.5647676626788538, 0.31282931566238403]                        
    
]

reacted_insert = [ # Deprecated - we are not inserting after reaction in this routine, but keeping for reference
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


##################
# CAMERA / PHOTO #
##################
def take_photo(vial_number, camera_index=0):
    output_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/superdemo_snapp_of_vial_{vial_number+1}_photo.jpg"
    cap = cv.VideoCapture(camera_index, cv.CAP_V4L2)
    ret, frame = cap.read()
    if ret:
        cv.imwrite(output_path, frame)
        print(f"Photo saved: {output_path}")
    cap.release()

def basic_recorder(vial_number, camera_index=0, width=640, height=480, fps=30, record_seconds=180):
    output_path = f"/home/robot/group_B/robo_chem_504/group_B_videos/superdemo_recipe_7_{vial_number+1}.mp4"
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
    time.sleep(1) # slight delay to ensure stir starts before recording
    stir_thread.start()
    
    # Record full duration in main thread
    basic_recorder(vial_number, record_seconds=total_seconds)
    
    stir_thread.join()


def run_vial(i, robot, gripper, plate):
    
    gripper.move(0,255,255) ## opens gripper at the start, make sure there is no vial in the gripper at the start of the routine
    
    # Pick vial
    robot.move_joint_list(unreacted_approach_high[i], 0.5, 0.5, 0.02)
    robot.move_joint_list(unreacted_insert[i], 0.5, 0.5, 0.02)
    gripper.move(170, 255, 255)

    # Move to stirrer
    robot.move_joint_list(unreacted_approach_low[i], 0.5, 0.5, 0.02)
    robot.move_joint_list(unreacted_approach_high[i], 0.5, 0.5, 0.02)
    robot.move_joint_list(stirring_position_above, 0.5, 0.5, 0.02)
    robot.move_joint_list(stirring_position_on, 0.5, 0.5, 0.02)

    # Photo + record
    take_photo(i)
    record_with_stir(plate, i, rpm=STIR_RPM, total_seconds=RECORD_SECONDS, stir_seconds=STIR_SECONDS)

    # Return vial
    robot.move_joint_list(home_position, 0.5, 0.5, 0.02)
    robot.move_joint_list(exhauted_approach_high[i], 0.5, 0.5, 0.02)
    robot.move_joint_list(exhauted_approach_low[i], 0.5, 0.5, 0.02)
    robot.move_joint_list(exhauted_insert[i], 0.5, 0.5, 0.02)
    gripper.move(0, 255, 255)
    robot.move_joint_list(unreacted_approach_high[i], 0.5, 0.5, 0.02)

########
# MAIN #
########
def main():
    robot  = URControl(ip=ROBOT_IP)
    gripper = RobotiqGripper(); gripper.connect(ROBOT_IP, GRIPPER_PORT)
    plate  = connect_plate(PLATE_PORT)

    for i in range(NUM_VIALS):
        run_vial(i, robot, gripper, plate)

if __name__ == "__main__":
    main()
