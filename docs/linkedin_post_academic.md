# LinkedIn Post Draft — Academic (Workflow Rigour + Traceability)

*Ready-to-paste draft. Edit the bracketed placeholders before posting.*

---

## Post text

During my final-year module *[Robotics and Automation in Chemistry, University of Liverpool]* I built **Robo-Auto-Chem** — an end-to-end pipeline that automates a *traffic-light* colour-change reaction and converts every run into **structured, traceable data**.

The goal was never just "make the robot move." It was to make the experiment **reproducible and auditable** from the first line of code to the last saved file.

**What the system does (per run):**

- A **UR5e robot** handles all vial transfers: picks each vial from a rack, places it on the stirring plate, waits through the reaction, then deposits the reacted vial into a separate rack position — keeping unreacted and reacted samples physically separated.
- An **IKA stirring plate** runs at a fixed RPM for a fixed duration, controlled by a Python serial driver.
- A **camera** records the full reaction window for each vial, saving an MP4 per vial. A snapshot is also taken before stirring starts.
- All timing and motion parameters (`STIR_RPM`, `STIR_SECONDS`, `RECORD_SECONDS`) are defined as named constants at the top of the run script — one place, version-controlled, no hidden state.

**Post-run analysis (two pipelines):**

- A **YOLO model** detects the liquid region in each MP4 on the first frame, then steps through the video computing per-frame RGB and HSV means for that region — quantifying the RED → YELLOW → GREEN colour transition without manual annotation.
- An alternative **manual-ROI pipeline** lets an operator draw the region of interest directly on the first frame of each video, giving the same statistics without a trained model.

Both pipelines write per-video CSVs and plots, plus a combined CSV across all vials in a batch — making it straightforward to compare reaction timing across vials or across experimental sessions.

**Why this matters to me (and maybe to you):**

The bottleneck in small-scale chemistry automation is rarely the hardware — it is the *data quality* on the other side of the run. Consistent camera framing, named output files, version-controlled scripts, and machine-readable logs mean that a result from three months ago is still interpretable and comparable today.

If you are working on **lab automation, vision-based measurement, or reproducible experimental workflows**, I would be glad to compare notes.

[Link to repo / video of demo day if desired]

\#LabAutomation \#Robotics \#Chemistry \#ComputerVision \#UR5e \#YOLO \#Reproducibility \#DataEngineering \#WorkflowDesign

---

## What the post references (implemented artefacts only)

| Claim in post | Implemented in |
|---|---|
| UR5e vial pick / transfer / return | [`ur5_main.py`](../ur5_main.py) — `run_vial()` |
| Separate unreacted / reacted rack positions | `exhausted_*` waypoints in [`ur5_main.py`](../ur5_main.py) |
| IKA stirring plate, fixed RPM + duration | [`archived/stirring_plate.py`](../archived/stirring_plate.py) · `STIR_RPM` / `STIR_SECONDS` constants |
| Per-vial MP4 recording | `basic_recorder()` in [`ur5_main.py`](../ur5_main.py) |
| Per-vial snapshot (.jpg) | `take_photo()` in [`ur5_main.py`](../ur5_main.py) |
| Named constants, version-controlled scripts | All run scripts in repo root |
| YOLO liquid detection + per-frame CSV / plot | [`AI_main.py`](../AI_main.py) → `batch_results/` |
| Manual-ROI analysis + per-frame CSV / plot | [`AI_highlight_main.py`](../AI_highlight_main.py) → `batch_results/` |
| Combined CSV across vials | `batch_results/all_videos_combined_results.csv` |

---

## Intentionally omitted (not implemented)

The following features appear in some earlier drafts of this project description
but are **not present** in the repository and should not be claimed:

- Automated COSHH / safety forms
- Live dashboard or real-time monitoring UI
- Automated sign-off / approval workflow
- Phase-gated pipeline with human review gates
