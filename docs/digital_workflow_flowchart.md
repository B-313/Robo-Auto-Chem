# Digital Workflow Flowchart — Robo-Auto-Chem (7-Phase)

This Mermaid flowchart represents the end-to-end digital workflow used for the CHEM504 Robo-Auto-Chem demonstration. Each phase produces defined artefacts that form the audit trail for a run.

```mermaid
flowchart TD
    A([🚀 START]) --> PLAN

    subgraph PLAN ["📋 PHASE 1 — PLAN"]
        P1[Define experiment objectives\nand scope]
        P2[Complete Experiment Design Form]
        P3{Safety risk\nacceptable?}
        P4[Obtain supervisor sign-off]
        P1 --> P2 --> P3
        P3 -- No --> P1
        P3 -- Yes --> P4
    end

    PLAN --> CODE

    subgraph CODE ["💻 PHASE 2 — CODE"]
        C1[Write / update automation scripts\ne.g. ur5_main.py, demo.py]
        C2[Peer code review]
        C3[Commit & tag locked script version\nvia Git]
        C1 --> C2 --> C3
    end

    CODE --> CHEMPREP

    subgraph CHEMPREP ["⚗️ PHASE 3 — CHEMICAL PREPARATION"]
        CH1[Prepare reagents\nto specification]
        CH2[Complete pre-run checklist\nrobot home · ROI calibration · stirrer · camera]
        CH3{All checks\npassed?}
        CH1 --> CH2 --> CH3
        CH3 -- No, resolve --> CH2
        CH3 -- Yes --> CH4[Chemical safety sign-off]
    end

    CHEMPREP --> RUN

    subgraph RUN ["🤖 PHASE 4 — RUN"]
        R1[Execute locked script\nUR5e pick → transfer → react → return]
        R2[Stirrer safety interlock:\nstirrer OFF before robot moves vial]
        R3[Camera captures reaction\nROI-locked per vial]
        R4{Run\ncompleted OK?}
        R1 --> R2 --> R3 --> R4
        R4 -- Error / fault --> R5[Log fault & halt\nreview before retry]
        R4 -- Yes --> R6[Run complete]
    end

    RUN --> LOG

    subgraph LOG ["📄 PHASE 5 — LOG"]
        L1[Generate per-vial run log\nJSON + PDF]
        L2[Save MP4 video with ROI overlay]
        L3[Save CSV: timestamp · colour state\nRED→YELLOW→GREEN · elapsed time · pixel counts]
        L4[Run log signed and timestamped]
        R6 --> L1
        L1 --> L2 --> L3 --> L4
    end

    LOG --> CHEMMAN

    subgraph CHEMMAN ["🧪 PHASE 6 — CHEMICAL MANAGEMENT"]
        CM1[Record quantities used\nand remaining]
        CM2[Disposal per COSHH / local procedure]
        CM3[Complete chemical management log\nhazard sign-off]
        CM1 --> CM2 --> CM3
    end

    CHEMMAN --> REVIEW

    subgraph REVIEW ["🔍 PHASE 7 — REVIEW"]
        RV1[Run post-run YOLO analysis\nmasked colour statistics from video]
        RV2[Populate results dashboard\ncompare colour-change timing across vials]
        RV3[Flag anomalies\nno change · wrong sequence · outlier timing]
        RV4[Write results summary]
        RV5[Archive bundle:\nscripts · logs · media · summary]
        RV6([✅ COMPLETE])
        RV1 --> RV2 --> RV3 --> RV4 --> RV5 --> RV6
    end

    style PLAN    fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    style CODE    fill:#dcfce7,stroke:#22c55e,color:#14532d
    style CHEMPREP fill:#fef9c3,stroke:#eab308,color:#713f12
    style RUN     fill:#fee2e2,stroke:#ef4444,color:#7f1d1d
    style LOG     fill:#f3e8ff,stroke:#a855f7,color:#4a044e
    style CHEMMAN fill:#ffedd5,stroke:#f97316,color:#7c2d12
    style REVIEW  fill:#e0f2fe,stroke:#0ea5e9,color:#0c4a6e
```

## Artefact Summary

| Phase | Key Artefacts |
|---|---|
| PLAN | Experiment design form, safety assessment, supervisor sign-off |
| CODE | Version-controlled scripts, locked script commit/tag (Git) |
| CHEMICAL PREPARATION | Pre-run checklist, chemical safety sign-off |
| RUN | Timestamped run record, fault log (if applicable) |
| LOG | Run log (JSON/PDF), MP4 video (ROI overlay), CSV (colour transitions) |
| CHEMICAL MANAGEMENT | Chemical management log, COSHH disposal record |
| REVIEW | YOLO analysis output, results dashboard, results summary, archive bundle |
