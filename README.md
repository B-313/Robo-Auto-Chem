# Chemistry Automation

This project integrates the automation of vial handling, plate control, and data collection of the red-light reaction. Computer vision with YOLO is used to gather and analyse visual data.

# Project file system structure

**Test_position.py** is the main file that handles the robotic arm.

**test_get_joints** is the get joints one.

**plate_control_pablo.py** defines the class for the plate and contains an example for usage

# Usage

## Create and activate the UR5e environment:

### Activation
```sh
conda activate /home/robot/anaconda3/envs/ur5
```

## Usage of movement sequences

Just call the files from the terminal:
```sh
python3 chem504-robot-tools-groupB/<path_to_file>

```

Examples:
```sh
python3 chem504-robot-tools-groupB/full_routine_100326
python3 chem504-robot-tools-groupB/untested_full_routine
python3 chem504-robot-tools-groupB/untested_full_routine.py

python3 chem504-robot-tools-groupB/multi_thread.py
```

