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
├── verify_setup.py            # Dependency and data verification
├── requirements.txt           # Python dependencies
│
├── data/
│   ├── train/                 # Training split (~217 images)
│   │   ├── _classes.csv       # Multi-label annotations
│   │   └── *.jpg              # Fastener images (mixed normal + corroded)
│   ├── valid/                 # Validation split (~62 images)
│   │   ├── _classes.csv
│   │   └── *.jpg
│   └── test/                  # Test split (~31 images)
│       ├── _classes.csv
│       └── *.jpg
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
│   └── visualizations/        # Score distributions, ROC curves, comparison figures
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
- Apply **greedy coreset subsampling** to retain 10% of patches while preserving the distribution's geometry (minimax facility location algorithm)
- Build a **FAISS FlatL2 index** over the coreset for fast nearest-neighbor search

**2. Threshold Calibration (Validation)**
- Run all normal validation images through the model
- Each image gets an anomaly score (max patch distance to nearest neighbor)
- Set threshold at the **99th percentile** of these normal scores (~1% false positive rate)
- Record mean and standard deviation of normal scores for confidence estimation

**3. Anomaly Scoring (Inference)**
- Extract patch features from the test image
- Query FAISS index for the K=9 nearest neighbors of each patch
- Patch score = L2 distance to nearest neighbor, re-weighted by neighborhood density (PatchCore Section 3.3)
- **Image score** = maximum patch score (the single most anomalous patch determines the verdict)
- **Heatmap** = patch scores reshaped to 32x32, bilinearly upsampled to 256x256, Gaussian-smoothed (sigma=4)

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
8. **Background Masking** — Zeros out non-component pixels

Color is preserved throughout (BGR 3-channel) because rust/corrosion detection depends heavily on color information.

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

---

## Evaluation Metrics

| Metric | Meaning |
|--------|---------|
| **AUROC** | Area under ROC curve — threshold-independent detection quality (1.0 = perfect) |
| **F1-max** | Best achievable F1 score across all thresholds |
| **Precision** | Of images flagged as corroded, what fraction actually are |
| **Recall** | Of all corroded images, what fraction were caught |
| **Accuracy** | Overall correct classification rate at the calibrated threshold |

The model also provides per-image confidence scores expressed as sigma (standard deviations from the normal distribution mean), enabling graded decisions rather than binary pass/fail.

---

## Key Design Decisions

1. **Anomaly detection over classification** — Corroded fasteners are rare and visually diverse; healthy ones are consistent. Training on healthy-only is more robust than trying to enumerate all defect types.

2. **Frozen pretrained backbone** — ImageNet features transfer remarkably well to metallic texture analysis. Fine-tuning would risk overfitting on the small dataset (~217 images).

3. **Color preservation** — Unlike many industrial inspection systems that convert to grayscale, this pipeline keeps full BGR color because rust produces distinctive orange-brown hues that are highly discriminative.

4. **Coreset subsampling at 10%** — Reduces memory bank from ~N x 1024 patches to ~N x 102 patches with < 0.02 AUROC loss. Makes FAISS search faster and reduces the saved model size.

5. **99th percentile threshold** — Calibrated on normal validation images, yielding ~1% false positive rate. This is the standard approach when labeled anomalies are unavailable for threshold optimization.
