# Chemistry Automation

This project integrates the automation of vial handling, plate control, and data collection of the red-light reaction. Computer vision with YOLO is used to gather and analyse visual data.

📄 **[View the Digital Poster →](docs/poster.md)** — one-page explainer covering the system overview, workflow, key components, data artifacts, methods, and how to run.

# Project file system structure

```
.
├── archived/
├── notes/
└── utils/
    └── __pycache__/
```

**scripts that use the UR5e robot run from root `./`**

`archived/` contains deprecated routines, kept as reference.
`notes/` contains recipes, insights from working on the lab, &c.
`utils` contains critical imports and functionality related to the UR5e robot.

# Usage

Make sure that the required dependencies are installed on a Conda environment:

```sh
pip install numpy
pip install opencv-python
pip install Pillow
pip install requests
pip install ur-rtde
pip install opencv-python
```

## Activate the UR5e environment:
E.G.:
```sh
conda activate /home/robot/anaconda3/envs/ur5
```

## Routines

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
## Output files:

The current output paths for the photo snaps and video recordings on `main.py` and `demo.py` are hardcoded.

**Create an output directory of choice, add to `.gitignore` and update the path on the routine files to match your desired naming conventions**


[File tree designer used](https://tree.nathanfriend.com/?s=(%27options!(%27fancy-~fullPath!false~trailingSlash-~rootDot-)~2(%272%27main3manual_move3notes%2F0demo_insights.md0lab_notes.md*utils%2F%27)~version!%271%27)*%5Cn-!true0*%20%202source!3.py*%01320-*)