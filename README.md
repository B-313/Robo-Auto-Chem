# Chemistry Automation

This project integrates the automation of vial handling, plate control, and data collection of the red-light reaction. Computer vision with YOLO is used to gather and analyse visual data.

# Project file system structure

main.py
manual_move.py
notes/
|-- demo_insights.md
`-- lab_notes.md
utils/

# Usage

## Activate the UR5e environment:

```sh
conda activate /home/robot/anaconda3/envs/ur5
```

## Usage of movement sequences

**main.py** contains the code to program and run an experiment

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



[File tree designer used](https://tree.nathanfriend.com/?s=(%27options!(%27fancy-~fullPath!false~trailingSlash-~rootDot-)~2(%272%27main3manual_move3notes%2F0demo_insights.md0lab_notes.md*utils%2F%27)~version!%271%27)*%5Cn-!true0*%20%202source!3.py*%01320-*)