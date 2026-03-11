import sys
import os
import time
import argparse
import math 
# Add the directory containing robotiq_preamble.py to the Python search path
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(current_dir, 'robotiq'))

from utils.UR_Functions import URfunctions as URControl

HOST = "192.168.0.2"
PORT = 30003

def main():
    robot = URControl(ip="192.168.0.2", port=30003)
    print(robot.get_current_joint_positions().tolist())
    print(robot.get_current_tcp())
if __name__ == '__main__':
     main()

##PICK VAIL
# [1.6737594604492188, -1.2379534405520936, 1.9010451475726526, -2.286394258538717, -1.5171406904803675, 0.0003061056195292622]
# [ 0.20381291 -0.53599716  0.06497283 -0.1689313   3.08148916 -0.07700357]
