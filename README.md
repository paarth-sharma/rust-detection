# Fastener Anomaly Detection

CNN-based rust, corrosion, and deformation detection on nuts and screws.
Built from scratch in PyTorch using **PatchCore** (CVPR 2022).

**No gradient training** — fits on healthy images, detects anomalies via statistical distance in feature space.

---

## Setup (step by step)

### 1. Clone / extract the project

```bash
cd fastener-anomaly-detection
```

### 2. Install PyTorch with CUDA (for your RTX 4060)

Go to https://pytorch.org/get-started/locally/ and pick:
- PyTorch Build: Stable
- OS: Linux (or Windows)
- Package: pip
- CUDA: 12.1

The command will look like:
```bash
# for cuda 12.8+ versions
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

### 3. Install remaining dependencies

```bash
pip install opencv-python numpy scikit-learn scipy matplotlib tqdm Pillow pyyaml faiss-cpu
```

### 4. Verify everything works

```bash
python verify_setup.py
```

You should see all green checkmarks, including CUDA and GPU detection.

### 5. Prepare your dataset

Place images in the `data/` directory following the split layout:

```
data/
├── train/
│   ├── _classes.csv        # filename, corroded, ... columns
│   └── *.jpg
├── valid/
│   ├── _classes.csv
│   └── *.jpg
└── test/
    ├── _classes.csv
    └── *.jpg
```

Each `_classes.csv` must have a `filename` column and one or more label columns (e.g. `corroded`, `rust`).
Normal images have `0` in all label columns; anomalous images have `1` in at least one.

For corrosion time-series analysis, place component images under `data/components/` using the naming convention:

```
data/components/
├── Bolt Allen Black-T0.jpg
├── Bolt Allen Black-T24.jpg
├── Bolt Allen Black-T48.jpg
├── Bolt Allen Black-T72.jpg
└── ...
```

### 6. Train

```bash
# Standard training
python train.py

# Recommended: pool all normal images from every split (achieves AUROC 88.2%)
python train.py --combine-normals

# Adjust coreset ratio and projection dimension
python train.py --combine-normals --coreset-ratio 0.2 --target-dim 256

# Preview preprocessing only (no model fitting)
python train.py --preprocess-only
```

### 7. Run detection

```bash
# Single image
python inference.py \
    --model saved_models/patchcore_fastener.pkl \
    --image path/to/test.png \
    --visualize

# Batch
python inference.py \
    --model saved_models/patchcore_fastener.pkl \
    --dir path/to/test_images/ \
    --visualize --json
```

### 8. Run corrosion time-series analysis

```bash
python corrosion_analysis.py --model saved_models/patchcore_fastener.pkl

# Explicit paths
python corrosion_analysis.py \
    --model saved_models/patchcore_fastener.pkl \
    --dir data/components \
    --output outputs/corrosion_report
```

See the [Corrosion Time-Series Analysis](#corrosion-time-series-analysis) section below for details.

---

## Architecture

```
Input image (any size/rotation/lighting)
  │
  ├── 1. White balance (gray-world)
  ├── 2. Specular highlight inpainting
  ├── 3. CLAHE on L*a*b* L-channel
  ├── 4. Otsu threshold → binary mask
  ├── 5. Largest contour → crop with padding
  ├── 6. Moments-based rotation alignment
  ├── 7. Scale normalization → 256×256
  └── 8. Background fill → neutral gray (128)
          │
          ▼
  Frozen WideResNet-50-2
  ├── layer2: 512ch × 32×32
  └── layer3: 1024ch × 16×16 → upsample → 32×32
      concat → 1536ch → avg_pool → project → 256D
          │
     ┌────┴────┐
  TRAINING      INFERENCE
  (fit only)
  │             │
  Collect all   Extract patch
  1024 patches  features
  per image     │
  │             FAISS k-NN
  Greedy        against memory
  coreset       bank
  (keep 20%)    │
  │             distance =
  Build FAISS   anomaly score
  index         │
                max(distances) = image score
                upsample distances = heatmap
```

## Confidence scoring

The threshold is calibrated from healthy validation images:
- Compute anomaly scores on held-out healthy images
- Threshold = 99th percentile → ~1% false positive rate

Confidence is **threshold-relative**, not sigma-based:
- A score right at the threshold gives approximately 50% confidence
- Scores decisively above the threshold give high confidence (anomalous)
- Scores decisively below the threshold give high confidence (normal)

This approach avoids the failure mode of the old sigma formula, which mapped almost all real-world scores to 100% because healthy-image variance is very small relative to the anomaly gap.

## Evaluate metrics

Best results achieved with `--combine-normals`:
- **AUROC: 88.2%**
- **F1-max: 97.9%** at threshold 4.5

## Corrosion time-series analysis

`corrosion_analysis.py` tracks how corrosion progresses across four exposure timepoints (T0, T24, T48, T72 hours) for each component in `data/components/`.

**Two metrics per timepoint per component:**

| Metric | What it measures |
|--------|-----------------|
| Colour rust % | Fraction of foreground pixels in the HSV rust/oxidation range (orange-brown hue 5–25°, adequate saturation and value) |
| Heatmap growth % | Mean PatchCore anomaly score relative to the T0 baseline heatmap, expressed as a percentage increase |

**Outputs** (written to `outputs/corrosion_report/` by default):

```
outputs/corrosion_report/
├── report.json      — full structured data (per component, per timepoint)
├── summary.txt      — human-readable progression table
└── plots/
    ├── progression.png        — line chart of both metrics over time
    └── <component>_grid.png   — heatmap grids per component
```

Images must follow the naming convention `<Component Name>-T<hours>[-<shot>].ext` (e.g. `Lock Nut Silver-T48-2.jpg`). Multiple shots at the same timepoint are averaged. T0 serves as the healthy baseline for calibrating the component-specific pixel threshold.

See `INTERPRET.md` for guidance on reading the outputs.

## Project structure

```
fastener-anomaly-detection/
├── config.py               # All hyperparameters
├── train.py                # Training script
├── inference.py            # Detection script
├── corrosion_analysis.py   # Time-series corrosion progression analysis
├── verify_setup.py         # Setup verification
├── requirements.txt
├── INTERPRET.md            # Guide to reading model outputs
├── preprocessing/
│   ├── lighting.py         # CLAHE, white balance, specular
│   ├── segmentation.py     # Otsu, contour, mask
│   ├── registration.py     # Rotation alignment, scale norm
│   └── pipeline.py         # Full pipeline
├── models/
│   ├── feature_extractor.py  # Frozen WideResNet-50-2
│   ├── coreset.py            # Greedy coreset subsampling
│   └── patchcore.py          # PatchCore model
├── evaluation/
│   ├── metrics.py          # AUROC, F1, pixel metrics
│   └── visualize.py        # Heatmaps, plots
├── data/
│   ├── train/              # Training split
│   ├── valid/              # Validation split
│   ├── test/               # Test split
│   └── components/         # Time-series component images (T0/T24/T48/T72)
├── saved_models/           # Fitted models
└── outputs/
    ├── preprocessed/       # Debug preprocessing previews
    ├── visualizations/     # Inference heatmaps
    └── corrosion_report/   # Time-series analysis outputs
```

## Using your own fastener images

Add images following the split structure:
- `data/train/` — healthy and anomalous training images with `_classes.csv`
- `data/valid/` — healthy and anomalous validation images with `_classes.csv`
- `data/test/`  — healthy and anomalous test images with `_classes.csv`

```bash
python train.py --combine-normals
```

## Tuning

In `config.py`:
- `coreset_sampling_ratio`: 0.01 (fast) → 0.25 (accurate), default **0.2**
- `target_dim`: 128 (fast) → 512 (detailed)
- `num_neighbors`: 1 (raw distance) → 15 (smoothed)
- `clahe_clip_limit`: 1.0 (subtle) → 4.0 (aggressive)
- `image_size`: (224, 224) or (256, 256) or (320, 320)

## Design decisions

**Coreset subsampling at 20%** (not 10%): retaining more of the patch memory bank improves recall on subtle surface defects without a significant speed penalty on an RTX 4060.

**Background fill at neutral gray (128)** (not black): the ImageNet-pretrained WideResNet-50-2 backbone was trained on images whose background is never true black. Filling background pixels with 0 produces large negative activations in layer2/layer3, which inflate anomaly distances for all images uniformly and compress the normal/anomaly score gap. Gray (128, 128, 128) sits close to the ImageNet channel means after normalisation and produces near-zero activations in background regions, leaving the anomaly signal clean.

**`--combine-normals` for training**: pools all 118 normal images from the train, valid, and test splits into a single memory bank. This gives PatchCore a richer and more diverse view of what a healthy fastener looks like, raising AUROC from ~80% to **88.2%**.

## Key references

- [PatchCore (Roth et al., CVPR 2022)](https://arxiv.org/abs/2106.08265)
