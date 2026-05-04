# LinkedIn Post — Academic (Workflow Rigor & Traceability)

> **Audience:** Academic peers, lab automation researchers, module supervisors.  
> **Tone:** Research-oriented, precise, professional.  
> **Copy and paste the text below the horizontal rule.**

---

We recently completed the demonstration of **Robo-Auto-Chem**, an end-to-end robotic chemistry automation pipeline developed as part of the CHEM504 module at the University of Liverpool.

The system coordinates a **UR5e collaborative robot**, a **stirring plate**, and a **wrist-mounted camera** to autonomously handle vials, execute a *traffic-light* iodine clock colour-change reaction, and capture structured experimental data — without manual intervention at any stage of the run.

**What distinguishes this work beyond robot motion:**  
The emphasis was placed on building a **reproducible, safe, and fully auditable workflow** around the hardware, structured into seven discrete phases:

`PLAN → CODE → CHEMICAL PREPARATION → RUN → LOG → CHEMICAL MANAGEMENT → REVIEW`

Each phase is governed by defined artefacts and handoffs:
- 📋 **PLAN:** Experiment design form + supervisor sign-off before any code is written
- 💻 **CODE:** Version-controlled scripts (Git); locked script version committed prior to execution
- ⚗️ **CHEMICAL PREPARATION:** Pre-run checklist (reagents, robot home position, ROI calibration, stirrer connectivity)
- 🤖 **RUN:** Automated vial pick → transfer → react/observe → return, with stirrer safety interlock (stirrer OFF before robot moves)
- 📄 **LOG:** Per-vial run log (JSON/PDF), MP4 video with ROI overlay, CSV capturing timestamps and RED→YELLOW→GREEN colour-state transitions
- 🧪 **CHEMICAL MANAGEMENT:** Chemical management log (quantities, disposal, hazard sign-off)
- 🔍 **REVIEW:** Post-run dashboard, results summary, archive bundle

This structure means that any run can be **traced from design intent to raw data artefact**, and that the conditions of a given run (RPM, stir duration, ROI bounds, script version) are recoverable from the log alone — a requirement for any credible comparative study across experimental batches.

Post-run analysis uses a **YOLO-based liquid region detector** to compute masked colour statistics over time from recorded video, providing a quantitative measure of reaction progression independent of manual visual inspection.

The full digital workflow diagram and documentation are available in the repository.

If you are working on **lab automation, robotic chemistry platforms, or reproducibility in experimental science**, I would welcome the opportunity to connect and exchange notes.

\#LabAutomation #Reproducibility #Traceability #RoboticChemistry #UR5e #ComputerVision #YOLO #DigitalWorkflow #CHEM504 #AuditTrail #VersionControl #ExperimentalScience
