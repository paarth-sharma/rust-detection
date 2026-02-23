# Architecture Diagrams

Mermaid diagrams for the fastener corrosion detection pipeline. Render these in any Mermaid-compatible viewer (VS Code Markdown Preview, GitHub, mermaid.live, etc.).

---

## 1. High-Level System Architecture

```mermaid
graph TB
    subgraph Data["Data Layer"]
        TR["data/train/<br/>_classes.csv + images"]
        VA["data/valid/<br/>_classes.csv + images"]
        TE["data/test/<br/>_classes.csv + images"]
    end

    subgraph Training["Training Pipeline (train.py)"]
        LS["Load Split<br/>Parse _classes.csv"]
        FN["Filter Normal<br/>Images Only"]
        CN["--combine-normals<br/>Pool all splits' normals"]
        PP["Preprocessing<br/>Pipeline"]
        FE["Feature Extraction<br/>WideResNet-50-2"]
        CS["Coreset<br/>Subsampling (20%)"]
        MB["Memory Bank<br/>+ FAISS Index"]
        TC["Threshold<br/>Calibration (P99)"]
    end

    subgraph Inference["Inference Pipeline"]
        IMG["Input Image"]
        PP2["Preprocessing"]
        FE2["Feature Extraction"]
        NN["Nearest Neighbor<br/>Search (FAISS)"]
        SC["Anomaly Score<br/>+ Heatmap"]
        DEC["Decision:<br/>Normal / Corroded"]
        CA["corrosion_analysis.py<br/>Corrosion Report"]
    end

    subgraph Outputs["Outputs"]
        MOD["saved_models/<br/>patchcore_fastener.pkl"]
        VIZ["outputs/visualizations/<br/>ROC, score dist, heatmaps"]
        MET["Metrics:<br/>AUROC, F1, Precision, Recall"]
        COR["outputs/corrosion_report/<br/>summary.txt, report.json, plots"]
    end

    TR --> LS
    VA --> LS
    TE --> LS

    LS --> FN
    FN -->|"normal images"| PP
    TR -.->|"optional: --combine-normals"| CN
    VA -.->|"optional: --combine-normals"| CN
    TE -.->|"optional: --combine-normals"| CN
    CN -.->|"pooled normals"| PP
    PP --> FE
    FE --> CS
    CS --> MB
    VA -->|"normal images"| TC
    MB --> TC
    TC --> MOD

    TE -->|"all images"| PP2

    IMG --> PP2
    PP2 --> FE2
    MOD --> NN
    FE2 --> NN
    NN --> SC
    SC --> DEC
    DEC --> CA

    TC --> VIZ
    DEC --> MET
    CA --> COR

    style Data fill:#e8f4f8,stroke:#2196F3
    style Training fill:#e8f5e9,stroke:#4CAF50
    style Inference fill:#fff3e0,stroke:#FF9800
    style Outputs fill:#f3e5f5,stroke:#9C27B0
    style CN fill:#f0f4c3,stroke:#827717,stroke-dasharray: 5 5
```

---

## 2. PatchCore Model Architecture

```mermaid
graph LR
    subgraph Backbone["WideResNet-50-2 (Frozen)"]
        IN["Input<br/>3 x 256 x 256"] --> L1["Layer 1<br/>64ch, 64x64"]
        L1 --> L2["Layer 2<br/>512ch, 32x32"]
        L2 --> L3["Layer 3<br/>1024ch, 16x16"]
        L3 --> L4["Layer 4<br/>2048ch, 8x8"]
    end

    subgraph Hooks["Forward Hooks"]
        L2 -.->|"hook"| H2["512 x 32 x 32"]
        L3 -.->|"hook"| H3["1024 x 16 x 16"]
    end

    subgraph Fusion["Feature Fusion"]
        H3 -->|"upsample 2x"| UP["1024 x 32 x 32"]
        H2 --> CAT["Concatenate"]
        UP --> CAT
        CAT -->|"1536 x 32 x 32"| AVG["AvgPool 3x3"]
        AVG --> PROJ["Linear Projection<br/>(frozen, orthogonal)"]
        PROJ -->|"256 x 32 x 32"| FLAT["Reshape"]
        FLAT -->|"1024 patches x 256 dim"| OUT["Patch Features"]
    end

    style Backbone fill:#e3f2fd,stroke:#1565C0
    style Hooks fill:#fff9c4,stroke:#F9A825
    style Fusion fill:#e8f5e9,stroke:#2E7D32
    style L1 fill:#e3f2fd,stroke:#999,stroke-dasharray: 5 5
    style L4 fill:#e3f2fd,stroke:#999,stroke-dasharray: 5 5
```

---

## 3. Preprocessing Pipeline

```mermaid
graph TD
    RAW["Raw Image<br/>(variable size, BGR)"] --> WB["1. White Balance<br/>Gray-world correction"]
    WB --> SH["2. Specular Highlights<br/>Telea inpainting of<br/>overexposed regions"]
    SH --> CLAHE["3. CLAHE<br/>L channel in L*a*b*<br/>clipLimit=2.0, grid=8x8"]
    CLAHE --> SEG["4. Segmentation<br/>Otsu threshold +<br/>morphological close/open"]
    SEG --> ROI["5. ROI Extraction<br/>Bounding box crop<br/>with 10% padding"]
    ROI --> ALIGN["6. Alignment<br/>Moment-based rotation<br/>to canonical pose"]
    ALIGN --> SCALE["7. Scale Normalization<br/>Fill 80% of canvas"]
    SCALE --> MASK["8. Background Masking<br/>Neutral gray fill (128)<br/>≈ ImageNet pixel mean"]
    MASK --> OUT["Preprocessed Image<br/>256 x 256 x 3 BGR"]

    SEG -.->|"contour"| ROI
    SEG -.->|"contour"| ALIGN
    ALIGN -.->|"symmetry detection<br/>hex nut = 60° period"| ALIGN

    style RAW fill:#ffebee,stroke:#c62828
    style OUT fill:#e8f5e9,stroke:#2E7D32
```

---

## 4. Training Data Flow

```mermaid
graph TD
    subgraph Split["Data Splits"]
        CSV_T["train/_classes.csv"]
        CSV_V["valid/_classes.csv"]
        CSV_E["test/_classes.csv"]
    end

    subgraph Parse["Label Parsing"]
        CSV_T -->|"parse"| FILT_T{"Corroded-Bolt OR<br/>Corroded-Nut OR<br/>Corroded-Nut-and-Bolt<br/>== 1?"}
        CSV_V -->|"parse"| FILT_V{"Corroded?"}
        CSV_E -->|"parse"| FILT_E{"Corroded?"}
    end

    subgraph Use["Pipeline Usage"]
        FILT_T -->|"No (normal)"| TRAIN["Memory Bank<br/>Construction"]
        FILT_T -->|"Yes (corroded)"| SKIP_T["Excluded from<br/>training"]
        FILT_V -->|"No (normal)"| THRESH["Threshold<br/>Calibration (P99)"]
        FILT_V -->|"Yes (corroded)"| SKIP_V["Excluded from<br/>calibration"]
        FILT_E -->|"No (normal)"| EVAL["Evaluation<br/>(label = 0)"]
        FILT_E -->|"Yes (corroded)"| EVAL2["Evaluation<br/>(label = 1)"]
    end

    CSV_T -.->|"--combine-normals"| COMBINE["Combined Normal Pool<br/>118 images<br/>80% train / 20% calib"]
    CSV_V -.->|"--combine-normals"| COMBINE
    CSV_E -.->|"--combine-normals"| COMBINE
    COMBINE -.->|"pooled normals"| TRAIN
    COMBINE -.->|"pooled calib set"| THRESH

    TRAIN --> MODEL["PatchCore Model"]
    THRESH --> MODEL
    EVAL --> METRICS["AUROC, F1, Precision,<br/>Recall, Accuracy"]
    EVAL2 --> METRICS

    style SKIP_T fill:#ffebee,stroke:#c62828,stroke-dasharray: 5 5
    style SKIP_V fill:#ffebee,stroke:#c62828,stroke-dasharray: 5 5
    style TRAIN fill:#e8f5e9,stroke:#2E7D32
    style THRESH fill:#fff3e0,stroke:#E65100
    style MODEL fill:#e3f2fd,stroke:#1565C0
    style COMBINE fill:#f0f4c3,stroke:#827717,stroke-dasharray: 5 5
```

---

## 5. Anomaly Scoring at Inference

```mermaid
graph TD
    IMG["Test Image"] --> PP["Preprocess<br/>256 x 256"]
    PP --> FE["Feature Extract<br/>1024 patches x 256D"]
    FE --> FAISS["FAISS L2 Search<br/>K=9 neighbors per patch"]

    subgraph MemBank["Memory Bank (from training)"]
        CORE["Coreset patches<br/>~N x 102 vectors x 256D"]
    end

    CORE --> FAISS

    FAISS --> D1["Nearest neighbor<br/>distance per patch"]
    D1 --> RW["Re-weight by<br/>neighborhood density"]
    RW --> MAX["Image Score =<br/>max(patch scores)"]
    RW --> RESHAPE["Reshape to 32x32"]
    RESHAPE --> UP["Bilinear upsample<br/>to 256x256"]
    UP --> GAUSS["Gaussian smooth<br/>sigma=4"]
    GAUSS --> HMAP["Anomaly Heatmap"]

    MAX --> CMP{"Score ><br/>Threshold?"}
    CMP -->|"Yes"| ANOM["CORRODED"]
    CMP -->|"No"| NORM["NORMAL"]

    MAX --> CONF["Confidence %<br/>threshold-relative<br/>~50% at threshold"]

    style ANOM fill:#ffebee,stroke:#c62828,color:#c62828
    style NORM fill:#e8f5e9,stroke:#2E7D32,color:#2E7D32
    style MemBank fill:#e3f2fd,stroke:#1565C0
```

---

## 6. Module Dependency Graph

```mermaid
graph BT
    CONFIG["config.py<br/>Configuration dataclasses"]

    subgraph Preprocessing
        LIGHT["preprocessing/<br/>lighting.py"]
        SEGM["preprocessing/<br/>segmentation.py"]
        REG["preprocessing/<br/>registration.py"]
        PIPE["preprocessing/<br/>pipeline.py"]
    end

    subgraph Models
        FEAT["models/<br/>feature_extractor.py"]
        CORE["models/<br/>coreset.py"]
        PC["models/<br/>patchcore.py"]
    end

    subgraph Evaluation
        METR["evaluation/<br/>metrics.py"]
        VIS["evaluation/<br/>visualize.py"]
    end

    TRAIN["train.py"]
    INFER["inference.py"]
    CA["corrosion_analysis.py"]

    LIGHT --> PIPE
    SEGM --> PIPE
    REG --> PIPE

    FEAT --> PC
    CORE --> PC

    CONFIG --> TRAIN
    CONFIG --> INFER
    CONFIG --> CA
    PIPE --> TRAIN
    PIPE --> INFER
    PIPE --> CA
    PC --> TRAIN
    PC --> INFER
    PC --> CA
    METR --> TRAIN
    VIS --> TRAIN
    VIS --> INFER

    style CONFIG fill:#fff9c4,stroke:#F9A825
    style TRAIN fill:#e8f5e9,stroke:#2E7D32
    style INFER fill:#fff3e0,stroke:#E65100
    style CA fill:#fff3e0,stroke:#E65100
    style Preprocessing fill:#f3e5f5,stroke:#7B1FA2
    style Models fill:#e3f2fd,stroke:#1565C0
    style Evaluation fill:#fce4ec,stroke:#C62828
```
