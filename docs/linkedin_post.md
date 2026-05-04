# LinkedIn Post Draft — Robo-Auto-Chem

> Copy the text between the `---` dividers and paste directly into LinkedIn.
> Choose Option A (professional) or Option B (concise) depending on your audience.

---

## Option A — Professional (results + full workflow)

I've been building **Robo-Auto-Chem** — a chemistry-automation module that connects **robotics, lab hardware, and computer vision** to run and analyse a classic traffic-light colour-change reaction end-to-end.

**What it does:**

- 🤖 **UR5e robot** automates vial handling: pick from rack → transfer to stirring plate → return
- 🌀 **Stirring plate control** runs the reaction with defined RPM and time parameters
- 📷 **Camera + ROI calibration** locks onto each vial for consistent, repeatable monitoring
- 📊 **Automated logging** produces MP4 recordings (with ROI overlay) and CSV files capturing timestamps and colour-state transitions (**RED → YELLOW → GREEN**)
- 🔍 A separate analysis pipeline uses **YOLO-based detection + masked pixel statistics** to quantify visual reaction progression from video

**The digital workflow behind every run (7 phases):**

`PLAN → CODE → CHEM PREP → RUN → LOG → CHEM MANAGEMENT → REVIEW`

Each phase is tracked so that every experiment is traceable, reproducible, and comparable across multiple vials.

**Why it matters:**
This kind of setup turns a "watch-the-vial" experiment into **structured, machine-readable data** — making it straightforward to compare runs, tune parameters, and flag anomalies automatically.

If you're working on **lab automation, robotics in chemistry, or vision-based measurement**, I'd love to connect and compare notes.

\#LabAutomation #Robotics #ComputerVision #Chemistry #UR5 #YOLO #Automation #DataEngineering #DigitalWorkflow

---

## Option B — Concise

I've been working on **Robo-Auto-Chem**: a chemistry automation module that combines a **UR5e robot**, a **stirring plate**, and **computer vision** to run and analyse a traffic-light colour-change reaction.

The workflow is fully automated across 7 phases — **PLAN → CODE → CHEM PREP → RUN → LOG → CHEM MANAGEMENT → REVIEW** — and every run outputs **videos + CSV logs** of colour transitions (RED→YELLOW→GREEN). A **YOLO-based** analysis step extracts reliable visual measurements from the recordings.

A fun project at the intersection of **robotics + chemistry + data**.

\#LabAutomation #UR5 #ComputerVision #ChemistryAutomation #DigitalWorkflow

---

## 7-Phase Digital Workflow Reference

| Phase | What happens |
|---|---|
| **PLAN** | Define experiment parameters (vials, RPM, stir time, ROI) |
| **CODE** | Configure routine file and analysis pipeline |
| **CHEM PREP** | Load vials, confirm robot home position, run ROI calibration |
| **RUN** | Execute robot routine: pick → react + record → return |
| **LOG** | Automated CSV + MP4 output per vial, timestamped |
| **CHEM MANAGEMENT** | Return vials, dispose of reagents safely, clean stirrer |
| **REVIEW** | Analyse logs with YOLO pipeline; compare colour-change timing across vials |

---

*See [`docs/export_to_pdf.md`](export_to_pdf.md) for instructions on converting this file to PDF.*
