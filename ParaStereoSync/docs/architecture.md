# Software Architecture Plotting

This project can be visualized at two levels:

- Level 1: high-level subsystem architecture (easy to read)
- Level 2: import dependency graph (auto-generated from code)

## 1) High-level architecture (Mermaid)

Use this publication-style layout for docs/paper narrative. It separates
interactive tools, runtime pipeline, and persisted data products.

```mermaid
flowchart TB

  %% ---------- Interactive tools ----------
  subgraph S1[Interactive Tools]
    direction LR
    CAL[Calibration GUI\ntkCalib2.py]
    TMP[Template Picker\npickTemplates.py]
    ORCH[Sync Orchestrator\nParaStereoSync.py]
  end

  %% ---------- Runtime pipeline ----------
  subgraph S2[Runtime Processing Pipeline]
    direction LR
    TRK[2D Tracking\neccTrackVideo_Multistep_warp_guess.py]
    TRI[3D Triangulation\ntriangulatePoints2.py]
    VIZ[Visualization\nKineVis3D.py]
  end

  %% ---------- Data products ----------
  subgraph S3[Persisted Data Products]
    direction LR
    CAM[(Camera Parameters)]
    TMPL[(Template Definitions)]
    P2D[(2D Tracks CSV)]
    P3D[(3D Points CSV)]
  end

  %% ---------- Utility layer ----------
  subgraph S4[Shared Utilities]
    direction LR
    CALU[Calibration Utils\nimproCalib.py, readCamera.py, writeCamera.py]
    IO[I/O Helpers\nreadPoints.py, writePoints.py]
  end

  %% control / execution links
  CAL --> CALU
  CALU --> CAM
  TMP --> TMPL
  ORCH --> TRK
  ORCH --> TRI
  ORCH --> IO
  ORCH -. loads .-> CAM
  ORCH -. loads .-> TMPL

  %% data flow links
  TRK --> P2D
  P2D --> TRI
  TRI --> P3D
  P3D --> VIZ

  %% style classes
  classDef tool fill:#fff5e6,stroke:#b45f06,stroke-width:1.5px,color:#2a1a00;
  classDef proc fill:#e9f4ff,stroke:#0b5394,stroke-width:1.5px,color:#07233f;
  classDef data fill:#eef8ee,stroke:#38761d,stroke-width:1.5px,color:#1f3f12;
  classDef util fill:#f3f0ff,stroke:#5b4fa3,stroke-width:1.2px,color:#2c245a;

  class CAL,TMP,ORCH tool;
  class TRK,TRI,VIZ proc;
  class CAM,TMPL,P2D,P3D data;
  class CALU,IO util;
```

Suggested figure caption (Software X style):

"Architecture of ParaStereoSync. Calibration and template-selection tools
produce reusable inputs (camera parameters and template definitions), which
are consumed by the synchronization/tracking orchestrator. The runtime pipeline
performs 2D feature tracking, 3D triangulation, and visualization, with CSV
artifacts persisted between stages."

### Compact two-column variant

Use this version when space is limited (for example, two-column manuscript
layouts). It preserves the same control and data semantics with fewer nodes.

```mermaid
flowchart LR
  IN[Inputs\nCamera Params + Templates]
  ORCH[Orchestrator\nParaStereoSync.py]
  PIPE[Pipeline\nTracking -> Triangulation]
  OUT[Outputs\n2D Tracks + 3D Points]
  CAL[Calibration\ntkCalib2.py + helpers]
  VIZ[Visualization\nKineVis3D.py]

  CAL --> IN
  IN --> ORCH --> PIPE --> OUT --> VIZ

  classDef tool fill:#fff5e6,stroke:#b45f06,stroke-width:1.2px,color:#2a1a00;
  classDef proc fill:#e9f4ff,stroke:#0b5394,stroke-width:1.2px,color:#07233f;
  classDef data fill:#eef8ee,stroke:#38761d,stroke-width:1.2px,color:#1f3f12;

  class CAL,ORCH,VIZ tool;
  class PIPE proc;
  class IN,OUT data;
```

Compact caption:

"Compact architecture view of ParaStereoSync showing calibration-derived
inputs, orchestration, tracking/triangulation pipeline, and visualization of
persisted 2D/3D outputs."

## 2) Auto-generated dependency graphs (pydeps)

Generated files in this repository:

- docs1111/architecture/architecture_main.dot
- docs1111/architecture/architecture_calibration.dot

These DOT files were generated from:

- ParaStereoSync.py
- tkCalib2.py

### Re-generate DOT files

From repository root:

```powershell
C:/Users/mearg/.conda/envs/stereosync/python.exe -m pydeps ParaStereoSync.py -x numpy cv2 scipy tkinter multiprocessing concurrent --max-bacon 2 --cluster --show-dot --dot-output docs1111/architecture/architecture_main.dot --no-output
C:/Users/mearg/.conda/envs/stereosync/python.exe -m pydeps tkCalib2.py -x numpy cv2 tkinter matplotlib --max-bacon 2 --cluster --show-dot --dot-output docs1111/architecture/architecture_calibration.dot --no-output
```

### Render DOT to SVG (optional)

If Graphviz is installed and dot is on PATH:

```powershell
dot -Tsvg docs1111/architecture/architecture_main.dot -o docs1111/architecture/architecture_main.svg
dot -Tsvg docs1111/architecture/architecture_calibration.dot -o docs1111/architecture/architecture_calibration.svg
```

## 3) Suggested plotting workflow for this codebase

1. Keep the Mermaid diagram for the paper/manuscript (readability).
2. Keep DOT/SVG import graphs in docs1111/architecture/ for technical appendix.
3. Update both when adding/removing major modules.
