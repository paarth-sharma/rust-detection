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

### 5. Download dataset

```bash
python scripts/download_datasets.py
```

This shows download options. The easiest path:

**Option A — MVTec official (cleanest):**
1. Go to https://www.mvtec.com/company/research/datasets/mvtec-ad/downloads
2. Create free account, accept CC BY-NC-SA 4.0 license
3. Download `Screw` (186 MB) and `Metal Nut` (157 MB)
4. Extract:
```bash
mkdir -p data/mvtec_ad
tar -xf screw.tar.xz -C data/mvtec_ad/
tar -xf metal_nut.tar.xz -C data/mvtec_ad/
```

**Option B — Kaggle:**
```bash
pip install kaggle
# Get API token from https://www.kaggle.com/settings → API → Create New Token
kaggle datasets download -d ipythonx/mvtec-ad -p data/ --unzip
```

**Verify:**
```bash
python scripts/download_datasets.py --verify
```

Expected structure:
```
data/mvtec_ad/screw/
├── train/good/          # 320 healthy screw images
├── test/good/           # 41 healthy test images
├── test/scratch_head/   # defective test images
├── test/scratch_neck/
├── test/thread_side/
├── test/thread_top/
├── test/manipulated_front/
└── ground_truth/        # pixel-level defect masks
```

### 6. Train

```bash
# On MVTec screws (~30s on RTX 4060)
python train.py --category screw

# On metal nuts
python train.py --category metal_nut

# On your own images
python train.py --data-dir data/custom_fasteners

# Preview preprocessing only (no model fitting)
python train.py --category screw --preprocess-only
```

### 7. Run detection

```bash
# Single image
python inference.py \
    --model saved_models/patchcore_mvtec_screw.pkl \
    --image path/to/test.png \
    --visualize

# Batch
python inference.py \
    --model saved_models/patchcore_mvtec_screw.pkl \
    --dir path/to/test_images/ \
    --visualize --json
```

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
  └── 8. Background zeroing
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
  (keep 10%)    │
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

Sigma-based confidence:
- μ, σ = mean and std of healthy scores
- For test image with score s: σ_distance = (s − μ) / σ
- 1σ → 68% confidence it's anomalous
- 2σ → 95% confidence
- 3σ → 99.7% ("definitely defective")

## Expected performance

| Dataset         | Image AUROC | Pixel AUROC | Memory Bank  | Fit Time |
|----------------|-------------|-------------|-------------|----------|
| MVTec Screw    | ~96.0%      | ~98.0%      | ~33k patches | ~30s    |
| MVTec Metal Nut| ~99.4%      | ~98.0%      | ~22k patches | ~20s    |

*RTX 4060 Mobile, WideResNet-50-2, 10% coreset.*

## Project structure

```
fastener-anomaly-detection/
├── config.py               # All hyperparameters
├── train.py                # Training script
├── inference.py            # Detection script
├── verify_setup.py         # Setup verification
├── requirements.txt
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
├── scripts/
│   └── download_datasets.py
├── data/                   # Datasets go here
├── saved_models/           # Fitted models
└── outputs/                # Visualizations
```

## Using your own fastener images

```bash
python scripts/download_datasets.py --setup-custom
```

Then add images:
- `data/custom_fasteners/train/good/` — 50-200 healthy images
- `data/custom_fasteners/test/good/` — healthy test
- `data/custom_fasteners/test/rust/` — rusted images
- `data/custom_fasteners/test/deformed/` — bent/deformed
- `data/custom_fasteners/test/corroded/` — corroded

```bash
python train.py --data-dir data/custom_fasteners
```

## Tuning

In `config.py`:
- `coreset_sampling_ratio`: 0.01 (fast) → 0.25 (accurate)
- `target_dim`: 128 (fast) → 512 (detailed)
- `num_neighbors`: 1 (raw distance) → 15 (smoothed)
- `clahe_clip_limit`: 1.0 (subtle) → 4.0 (aggressive)
- `image_size`: (224, 224) or (256, 256) or (320, 320)

## Key references

- [PatchCore (Roth et al., CVPR 2022)](https://arxiv.org/abs/2106.08265)
- [MVTec AD Dataset](https://www.mvtec.com/company/research/datasets/mvtec-ad)
