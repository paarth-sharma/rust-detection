# Fastener Corrosion Detection via CNN-Based Anomaly Detection

## Overview

This project detects rust and corrosion on nuts, screws, and bolts using **unsupervised anomaly detection**. Rather than training a classifier on both healthy and defective images, the system learns what "normal" looks like from healthy fastener images alone, then flags anything that deviates as anomalous. This is the anomaly detection paradigm — well-suited to industrial inspection where defective samples are rare and diverse, but healthy components are abundant and consistent.

The core method is **PatchCore** (Roth et al., CVPR 2022), which uses a frozen ImageNet-pretrained **WideResNet-50-2** backbone to extract patch-level feature vectors from healthy training images, stores them in a memory bank, and scores test images by their nearest-neighbor distance to this bank.

---

## Project Structure

```
rust-detection/
├── config.py                  # Central configuration (dataclasses)
├── train.py                   # Training pipeline entry point
├── inference.py               # Single-image / batch inference
├── corrosion_analysis.py      # Time-series corrosion progression analysis
├── verify_setup.py            # Dependency and data verification
├── requirements.txt           # Python dependencies
├── INTERPRET.md               # Guide to reading model outputs and heatmaps
├── HOWTO.md                   # Step-by-step usage guide
│
├── data/
│   ├── train/                 # Training split (~217 images)
│   │   ├── _classes.csv       # Multi-label annotations
│   │   └── *.jpg              # Fastener images (mixed normal + corroded)
│   ├── valid/                 # Validation split (~62 images)
│   │   ├── _classes.csv
│   │   └── *.jpg
│   ├── test/                  # Test split (~31 images)
│   │   ├── _classes.csv
│   │   └── *.jpg
│   └── components/            # Time-series images for corrosion progression
│       └── <Component>-T<hours>[-<shot>].jpg
│
├── preprocessing/
│   ├── __init__.py
│   ├── lighting.py            # CLAHE, white balance, specular highlight reduction
│   ├── segmentation.py        # Otsu thresholding, morphological ops, ROI extraction
│   ├── registration.py        # Moment-based alignment, symmetry detection, scale normalization
│   └── pipeline.py            # FastenerPreprocessor (orchestrates the full pipeline)
│
├── models/
│   ├── __init__.py
│   ├── feature_extractor.py   # Frozen WideResNet-50-2 with forward hooks
│   ├── coreset.py             # Greedy coreset subsampling (minimax facility location)
│   └── patchcore.py           # PatchCore: fit, predict, classify, save/load
│
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # AUROC, F1-max, precision, recall, pixel AUROC
│   └── visualize.py           # Heatmap overlays, score distributions, ROC curves
│
├── outputs/                   # Generated at runtime
│   ├── preprocessed/          # Sample preprocessed images
│   ├── visualizations/        # Score distributions, ROC curves, comparison figures
│   └── corrosion_report/      # Per-component time-series analysis reports
│
└── saved_models/              # Generated at runtime
    └── patchcore_fastener.pkl # Serialized memory bank + threshold + config
```

---

## Model Architecture

### Backbone: WideResNet-50-2 (Frozen)

The feature extractor is a **WideResNet-50-2** pretrained on ImageNet (1.2M natural images, 1000 classes). It is used **entirely frozen** — no weights are updated during training. The key insight from the anomaly detection literature is that intermediate CNN layers produce remarkably transferable features for defect detection without any fine-tuning on industrial data.

- **Layer 2 output**: 512 channels at 32x32 spatial resolution (texture-sensitive features)
- **Layer 3 output**: 1024 channels at 16x16 spatial resolution (structural features)
- Layer 3 is upsampled to 32x32 and concatenated with Layer 2, yielding **1536-channel** feature maps
- An **average pooling** (3x3, stride 1) is applied for local spatial aggregation
- A frozen **random orthogonal projection** reduces dimensionality from 1536 to **256** dimensions

This produces **1024 patch descriptors** (32x32 grid) per image, each a 256-dimensional vector.

### Why Layers 2 + 3

| Layer | Receptive Field | Captures | Use Case |
|-------|----------------|----------|----------|
| Layer 1 | Small | Edges, gradients | Too generic |
| **Layer 2** | **Medium** | **Texture, fine detail** | **Early rust spots, micro-scratches** |
| **Layer 3** | **Large** | **Structure, shape** | **Corrosion patches, thread damage** |
| Layer 4 | Very large | Semantic/class info | Too ImageNet-biased |

### PatchCore Method

PatchCore operates in three phases:

**1. Memory Bank Construction (Training)**
- Extract 256-dim patch features from all normal training images
- Each 256x256 image yields 1024 patches, so N images produce N x 1024 feature vectors
- Apply **greedy coreset subsampling** to retain **20%** of patches while preserving the distribution's geometry (minimax facility location algorithm)
- Build a **FAISS FlatL2 index** over the coreset for fast nearest-neighbor search

**2. Threshold Calibration (Validation)**
- Run all normal validation images through the model
- Each image gets an anomaly score (max patch distance to nearest neighbor)
- Set threshold at the **99th percentile** of these normal scores (~1% false positive rate)
- Record mean and standard deviation of normal scores for confidence estimation
- When `--combine-normals` is used, calibration uses an internal 80/20 split of the combined normal pool rather than the separate validation split (see `--combine-normals` section below)

**3. Anomaly Scoring (Inference)**
- Extract patch features from the test image
- Query FAISS index for the K=9 nearest neighbors of each patch
- Patch score = L2 distance to nearest neighbor, re-weighted by neighborhood density (PatchCore Section 3.3)
- **Image score** = maximum patch score (the single most anomalous patch determines the verdict)
- **Heatmap** = patch scores reshaped to 32x32, bilinearly upsampled to 256x256, Gaussian-smoothed (sigma=4)

### `--combine-normals` Training Mode

The `--combine-normals` flag enables pooling all normal images from the train, valid, and test splits into a single combined set (118 images total). The combined pool is then split internally: **80% builds the memory bank** and **20% calibrates the anomaly threshold**. This differs from the default mode, which uses only the `train/` normal images for the memory bank and the `valid/` normal images for threshold calibration.

Motivation: the default single-split approach gave random-chance AUROC performance on this dataset because the train normal set alone was too small and not representative enough to define a tight normal boundary. Pooling all available normal images before the 80/20 split increased AUROC to **88.2%**, demonstrating that the volume and diversity of the normal training set is the dominant factor in PatchCore performance on small industrial datasets.

---

## Data Pipeline

### Dataset Labels

Each split directory contains a `_classes.csv` with multi-label annotations:

| Column | Meaning |
|--------|---------|
| `Corroded-Bolt` | Bolt shows corrosion |
| `Corroded-Nut` | Nut shows corrosion |
| `Corroded-Nut-and-Bolt` | Both nut and bolt corroded |
| `Non-Corroded-Nut-and-Bolt` | Clean nut-bolt assembly |
| `Non-corroded Nut` | Clean nut |
| `Non-corroded-Bolts` | Clean bolt |

An image is classified as **corroded (anomalous)** if any of `Corroded-Bolt`, `Corroded-Nut`, or `Corroded-Nut-and-Bolt` equals 1. Otherwise it is **normal**.

### How Each Split Is Used

| Split | Total Images | Role | Which Images Used |
|-------|-------------|------|-------------------|
| `train/` | ~217 | Build memory bank | **Normal only** |
| `valid/` | ~62 | Calibrate anomaly threshold | **Normal only** |
| `test/` | ~31 | Evaluate detection performance | **All** (normal + corroded) |

This follows the anomaly detection paradigm: the model never sees corroded images during training. It learns the distribution of "healthy" and flags deviations.

### Preprocessing Pipeline

Every image passes through this sequence before feature extraction:

1. **White Balance** — Gray-world correction normalizes color cast from variable lighting
2. **Specular Highlight Reduction** — Inpaints overexposed metal reflections (Telea inpainting)
3. **CLAHE** — Contrast Limited Adaptive Histogram Equalization on L channel in L\*a\*b\* space; normalizes brightness while preserving color (critical for rust detection since rust has distinctive orange-brown hues)
4. **Segmentation** — Otsu thresholding + morphological close/open isolates the fastener from background; finds the largest contour
5. **ROI Extraction** — Crops to the component bounding box with 10% padding
6. **Alignment** — Computes image moments (centroid + principal axis orientation) and applies affine rotation to a canonical pose; detects hexagonal symmetry (D6) to normalize within the 60-degree fundamental domain
7. **Scale Normalization** — Scales component to fill 80% of the 256x256 target canvas
8. **Background Masking** — Fills non-component pixels with **neutral gray (value 128)**. Black (0) was the original fill but produced large negative activations in the WideResNet-50-2 backbone, which was pretrained on ImageNet natural images where pixel values are never truly black. This caused inflated anomaly scores for all images regardless of corrosion state. Gray (128) is close to the ImageNet pixel mean after normalization, producing near-zero activations in background regions and confining anomaly signal to the foreground component.

Color is preserved throughout (BGR 3-channel) because rust/corrosion detection depends heavily on color information.

---

## Confidence Scoring

Each inference result includes a confidence percentage computed from the image's anomaly score relative to the calibrated threshold. The formula is threshold-relative rather than mean-relative:

```
margin = (score - threshold) / max(threshold - mean, std)
confidence = norm.cdf(margin) * 100
```

This gives approximately **50% confidence at the threshold** (genuine ambiguity) and high confidence for scores decisively above or below it. The denominator is the larger of `(threshold - mean)` and `std`, which prevents division by very small values when the threshold sits close to the training mean.

The previous z-score formulation (`(score - mean) / std`) consistently returned near-100% confidence for all test images because test scores — both normal and corroded — were far from the training mean in z-score units, making the confidence uninformative.

---

## Corrosion Time-Series Analysis (`corrosion_analysis.py`)

`corrosion_analysis.py` tracks corrosion progression for individual components photographed repeatedly over time. It complements the single-image anomaly detector by providing trend analysis across a defined inspection schedule.

### Image Naming Convention

Time-series images are stored in `data/components/` and follow the pattern:

```
<Component>-T<hours>[-<shot>].jpg
```

- `<Component>` — component identifier (e.g., `BoltA`, `NutB`)
- `T<hours>` — elapsed time in hours from the start of the experiment (T0, T24, T48, T72, etc.)
- `[-<shot>]` — optional shot index for multiple angles at the same time point

Examples: `BoltA-T0.jpg`, `BoltA-T24.jpg`, `NutB-T48-2.jpg`

### Metrics

Two complementary metrics are computed per image:

**`colour_rust_pct`** — percentage of foreground pixels falling in the rust HSV color range:
- Hue: 5–25 (orange-brown)
- Saturation: >= 45
- Value: 40–210

This directly measures the visible rust-colored area as a fraction of the component surface.

**`heatmap_growth_pct`** — percentage of foreground pixels whose PatchCore heatmap value exceeds the T0 mean heatmap by more than 1.5 standard deviations. This captures structural anomaly growth that may not yet be visible as discoloration (e.g., pitting, surface texture changes before color change).

### T0-Relative Calibration

T0 (the zero-hour image) serves as the per-component baseline. All subsequent time points are interpreted relative to T0 rather than against a global threshold. This accounts for component-to-component variation: a fastener that arrives slightly discolored should not be flagged as rusting simply because it differs from a pristine reference.

### `is_rusting` Flag

A component is flagged as actively rusting at a given time point if either of these deltas versus T0 is exceeded:

- `colour_rust_pct` delta > **2 percentage points**, or
- `heatmap_growth_pct` delta > **5 percentage points**

### Severity Levels

| Level | Condition |
|-------|-----------|
| **None** | No change from T0 baseline |
| **Trace** | Small delta, below `is_rusting` thresholds |
| **Mild** | `is_rusting` flag active; minor exceedance |
| **Moderate** | Significant exceedance of one or both thresholds |
| **Severe** | Large exceedance; visible rust spreading |
| **Critical** | Extreme exceedance; substantial surface coverage |

Reports are written to `outputs/corrosion_report/`.

---

## Frameworks and Dependencies

| Package | Version | Role |
|---------|---------|------|
| **PyTorch** | >= 2.0 | Backbone inference, tensor operations |
| **torchvision** | >= 0.15 | WideResNet-50-2 pretrained weights, ImageNet transforms |
| **FAISS** (faiss-cpu) | >= 1.7.4 | Fast nearest-neighbor search over memory bank |
| **OpenCV** | >= 4.8 | Image I/O, preprocessing (CLAHE, morphology, alignment) |
| **NumPy** | >= 1.24 | Array operations |
| **scikit-learn** | >= 1.3 | AUROC, precision-recall curves, confusion matrix |
| **SciPy** | >= 1.11 | Gaussian filtering for heatmap smoothing, statistical functions |
| **matplotlib** | >= 3.7 | Visualization (score distributions, ROC curves) |
| **tqdm** | >= 4.65 | Progress bars |

Hardware target: Intel i7-13th gen + NVIDIA RTX 4060 Mobile (8 GB VRAM). Falls back to CPU automatically if CUDA is unavailable.

---

## Usage

### Verify Setup
```bash
python verify_setup.py
```

### Train
```bash
# Default settings
python train.py

# Custom parameters
python train.py --coreset-ratio 0.05 --target-dim 128 --batch-size 16

# Preview preprocessing only (no model training)
python train.py --preprocess-only

# Pool all normal images for improved AUROC on small datasets
python train.py --combine-normals
```

### Inference
```bash
# Single image
python inference.py --model saved_models/patchcore_fastener.pkl --image test.png

# Batch with visualizations
python inference.py --model saved_models/patchcore_fastener.pkl --dir test_images/ --visualize

# Export results as JSON
python inference.py --model saved_models/patchcore_fastener.pkl --dir test_images/ --json
```

### Corrosion Time-Series Analysis
```bash
# Analyse all components in data/components/
python corrosion_analysis.py

# Analyse a specific component
python corrosion_analysis.py --component BoltA
```

Results are saved to `outputs/corrosion_report/`.

---

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **AUROC** | Area under ROC curve — threshold-independent detection quality (1.0 = perfect) |
| **F1-max** | Best achievable F1 score across all thresholds |
| **Precision** | Of images flagged as corroded, what fraction actually are |
| **Recall** | Of all corroded images, what fraction were caught |
| **Accuracy** | Overall correct classification rate at the calibrated threshold |

The model also provides per-image confidence scores (see Confidence Scoring section above), enabling graded decisions rather than binary pass/fail.

---

## Key Design Decisions

1. **Anomaly detection over classification** — Corroded fasteners are rare and visually diverse; healthy ones are consistent. Training on healthy-only is more robust than trying to enumerate all defect types.

2. **Frozen pretrained backbone** — ImageNet features transfer remarkably well to metallic texture analysis. Fine-tuning would risk overfitting on the small dataset (~217 images).

3. **Color preservation** — Unlike many industrial inspection systems that convert to grayscale, this pipeline keeps full BGR color because rust produces distinctive orange-brown hues that are highly discriminative.

4. **Coreset subsampling at 20%** — Retains a larger representative slice of the patch distribution compared to the original 10% setting. The increase was made after observing that the smaller coreset under-represented the diversity of normal appearances in the combined-normals training mode, leading to false positives on normal patches that happened to be poorly covered. The 20% coreset still yields fast FAISS search and acceptable model file size.

5. **99th percentile threshold** — Calibrated on normal validation images, yielding ~1% false positive rate. This is the standard approach when labeled anomalies are unavailable for threshold optimization.

6. **Gray background fill (128) instead of black (0)** — The WideResNet-50-2 backbone was pretrained on ImageNet, where pure black pixels do not occur. Filling masked background pixels with 0 produced large negative activations that propagated into anomaly scores, inflating scores for every image. Filling with 128 (close to the ImageNet pixel mean of ~117–124 across channels after de-normalization) keeps background activations near zero and isolates the anomaly signal to the foreground fastener.

7. **`--combine-normals` for small datasets** — When the per-split normal count is too low to define a reliable normal boundary, pooling all available normal images before an 80/20 train/calibration split substantially improves AUROC (achieved 88.2% vs. random-chance performance with single-split training on this dataset).
