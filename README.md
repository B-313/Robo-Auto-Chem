# Chemistry Automation

This project integrates the automation of vial handling, plate control, and data collection of the red-light reaction. Computer vision with YOLO is used to gather and analyse visual data.

## Documentation

| Document | Description |
|---|---|
| [docs/digital_workflow.md](docs/digital_workflow.md) | Implemented workflow with Mermaid flowchart and links to every script |
| [docs/linkedin_post_academic.md](docs/linkedin_post_academic.md) | Academic LinkedIn post draft (workflow rigour + traceability) |
| [notes/demo_insights.md](notes/demo_insights.md) | Chemistry context, stirring parameters, run order |
| [notes/lab_notes.md](notes/lab_notes.md) | Reaction recipes used across sessions |

# Project file system structure

```
.
├── archived/        deprecated routines, kept as reference
├── docs/            workflow documentation
├── notes/           recipes, insights from working in the lab
└── utils/           UR5e robot control imports (UR_Functions.py, RTDE)
```

**Scripts that use the UR5e robot must be run from the repository root `./`**

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

**`ur5_main.py`** is the primary run script (full routine with a separate exhausted rack).  
**`demo.py`** is the demonstration variant used on demo day.

Run from the repository root:
```sh
python3 ur5_main.py
# or
python3 demo.py
```

Post-run video analysis:
```sh
python3 AI_main.py            # YOLO-based liquid detection
python3 AI_highlight_main.py  # manual ROI analysis
```

See [docs/digital_workflow.md](docs/digital_workflow.md) for the full step-by-step workflow.

## Output files:

The current output paths for the photo snaps and video recordings on `main.py` and `demo.py` are hardcoded.

**Create an output directory of choice, add to `.gitignore` and update the path on the routine files to match your desired naming conventions**


[File tree designer used](https://tree.nathanfriend.com/?s=(%27options!(%27fancy-~fullPath!false~trailingSlash-~rootDot-)~2(%272%27main3manual_move3notes%2F0demo_insights.md0lab_notes.md*utils%2F%27)~version!%271%27)*%5Cn-!true0*%20%202source!3.py*%01320-*)