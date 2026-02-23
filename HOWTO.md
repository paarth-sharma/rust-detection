# HOWTO — Fastener Corrosion Detection

Step-by-step instructions for training, testing, running inference, and running
the time-series corrosion analysis.

---

## 1. Activate the virtual environment

```bash
cd ~/.vscode/rust-detection
source .venv/bin/activate
```

Always activate the venv first. Every command below assumes it is active.

---

## 2. Verify the setup

```bash
python verify_setup.py
```

Checks all dependencies, CUDA availability, preprocessing pipeline, and feature
extractor. All items should show a checkmark. If the GPU is not detected, training
will fall back to CPU (much slower).

---

## 3. Understand the data layout

```
data/
├── train/
│   ├── _classes.csv          ← labels for every image in this split
│   └── *.jpg
├── valid/
│   ├── _classes.csv
│   └── *.jpg
└── test/
    ├── _classes.csv
    └── *.jpg
```

**`_classes.csv` format:**
```
filename,Corroded-Bolt,Corroded-Nut,Corroded-Nut-and-Bolt,Non-Corroded-Nut-and-Bolt,...
my_image.jpg,1,0,0,0,...
```

A `1` in any of the three `Corroded-*` columns marks an image as anomalous.
PatchCore only trains on normal (non-corroded) images. Corroded images are only
used for evaluation on the test set.

Current split sizes:
- Train  — 87 normal, 129 corroded
- Valid  — 24 normal, 37 corroded
- Test   — 7 normal, 23 corroded

---

## 4. Preview preprocessing (optional but recommended)

```bash
python train.py --preprocess-only
```

Runs the full preprocessing pipeline (white balance → CLAHE → segmentation →
alignment → 256×256 resize → neutral-gray background fill) without fitting the
model. Saves sample outputs to `outputs/preprocessed/`. Check those images before
a full training run to confirm the fasteners are correctly centred, masked, and
scaled.

---

## 5. Train the model

### Standard training

```bash
python train.py
```

What it does:
1. Parses `_classes.csv` in each split, separates normal from corroded images
2. Preprocesses the normal training images
3. Extracts patch features via frozen WideResNet-50-2 (on CUDA if available)
4. Builds a memory bank and applies 20% coreset subsampling
5. Calibrates the anomaly threshold on normal validation images
6. Evaluates on the test set — prints AUROC, F1, precision, recall
7. Saves visualisations to `outputs/visualizations/`
8. Saves the model to `saved_models/patchcore_fastener.pkl`

### Recommended training (larger, more generalised memory bank)

```bash
python train.py --combine-normals
```

The `--combine-normals` flag pools all normal images from train + valid + test
splits (118 total vs. 87 with standard training) and uses an 80/20 internal
shuffle for threshold calibration. This is the approach that achieved AUROC 88.2%
and should be used for any serious deployment.

### All training flags

```bash
python train.py --combine-normals          # pool all normal images (recommended)
python train.py --coreset-ratio 0.3        # larger memory bank, better recall
python train.py --coreset-ratio 0.1        # smaller memory bank, faster inference
python train.py --target-dim 128           # lower feature dim, less VRAM
python train.py --target-dim 512           # higher feature dim, more detail
python train.py --batch-size 16            # reduce if you hit VRAM limits
python train.py --device cpu               # force CPU (no GPU)
python train.py --data-dir data/my_set     # use a different data directory
```

Flags can be combined:
```bash
python train.py --combine-normals --coreset-ratio 0.3 --target-dim 256
```

### Retraining from scratch

The model file is overwritten on each run. To retrain cleanly:
```bash
python train.py --combine-normals
```

No need to delete the old model first — it is replaced automatically when
training finishes.

---

## 6. Check training results

```bash
ls outputs/visualizations/
```

| File | What it shows |
|---|---|
| `score_dist.png` | Histogram of anomaly scores — normal vs corroded on the test set. The two distributions should be separated with the threshold line between them. |
| `roc.png` | ROC curve with AUROC value. Higher is better; 0.88+ is good for this dataset. |
| `test_normal_*.png` | Per-image comparison for normal test images — original photo, heatmap overlay, verdict. |
| `test_corroded_*.png` | Same for corroded test images. |

---

## 7. Run inference on new images

### Single image

```bash
python inference.py \
    --model saved_models/patchcore_fastener.pkl \
    --image path/to/image.jpg
```

### Entire folder

```bash
python inference.py \
    --model saved_models/patchcore_fastener.pkl \
    --dir path/to/folder/
```

### With heatmap visualisations saved to disk

```bash
python inference.py \
    --model saved_models/patchcore_fastener.pkl \
    --dir path/to/folder/ \
    --visualize \
    --output outputs/my_run
```

Heatmap overlays are saved to the `--output` directory as
`det_<name>.png` (comparison figure) and `overlay_<name>.png` (heatmap blended
onto the original image).

### Export results to JSON

```bash
python inference.py \
    --model saved_models/patchcore_fastener.pkl \
    --dir path/to/folder/ \
    --json
```

Writes `outputs/inference/results.json` containing per-image score, threshold,
confidence, sigma, and is_anomalous flag.

### Reading inference output

Each image prints one line:
```
ANOMALY | score=14.2312 | σ=3.1 | conf=87.4% | image.jpg
NORMAL  | score=6.1045  | σ=0.5 | conf=43.2% | image.jpg
```

| Field | Meaning |
|---|---|
| `score` | Raw PatchCore distance. Higher = more different from the healthy memory bank. |
| `σ` | How many standard deviations the score sits above the mean normal score. |
| `conf` | How decisively the score sits on one side of the threshold. A score right at the threshold gives ~50%. Well above or below gives high %. |
| `ANOMALY / NORMAL` | Verdict based on whether score exceeds the calibrated threshold. |

---

## 8. Run the corrosion time-series analysis

For sets of images named `<Component>-T<hours>.jpg` or
`<Component>-T<hours>-<shot>.jpg` (e.g. `Bolt Allen Black-T48-1.jpg`).

### Run on the components folder

```bash
python corrosion_analysis.py \
    --model saved_models/patchcore_fastener.pkl \
    --dir data/components
```

### Custom output directory

```bash
python corrosion_analysis.py \
    --model saved_models/patchcore_fastener.pkl \
    --dir data/components \
    --output outputs/my_corrosion_run
```

### Skip plot generation (faster)

```bash
python corrosion_analysis.py \
    --model saved_models/patchcore_fastener.pkl \
    --dir data/components \
    --no-plots
```

### Outputs

```
outputs/corrosion_report/
├── summary.txt          ← human-readable table (see INTERPRET.md for column guide)
├── report.json          ← machine-readable full data
└── plots/
    ├── progression_all.png          ← line chart: all components over time
    ├── delta_bars.png               ← bar chart: rust growth vs T0 per interval
    ├── Bolt_Allen_Black_heatmaps.png
    ├── Lock_Nut_Silver_heatmaps.png
    └── ...                          ← one heatmap grid per component
```

### Adding new components for time-series analysis

Name your images using this exact convention:
```
<Component Name>-T<hours>.jpg
<Component Name>-T<hours>-<shot_number>.jpg
```

Examples:
```
Stainless Bolt M8-T0.jpg
Stainless Bolt M8-T0-1.jpg
Stainless Bolt M8-T0-2.jpg
Stainless Bolt M8-T24.jpg
Stainless Bolt M8-T48.jpg
Stainless Bolt M8-T72.jpg
```

Rules:
- `T0` images are the healthy baseline — always include them, they are required for calibration
- Multiple shots per timepoint are supported and averaged automatically
- The component name can contain spaces and any characters except `-T` followed by digits
- All images for all components can live in the same flat folder — the script groups them by name automatically
- Supported formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

Place images in `data/components/` (or any folder) and run:
```bash
python corrosion_analysis.py --model saved_models/patchcore_fastener.pkl --dir data/components
```

---

## 9. Tuning guide

All hyperparameters live in `config.py`. The most impactful ones:

| Parameter | Where | Default | Effect |
|---|---|---|---|
| `coreset_sampling_ratio` | `train.py --coreset-ratio` | 0.2 | Higher = larger memory bank = better recall, slower inference and training |
| `target_dim` | `train.py --target-dim` | 256 | Feature projection dimension. 128 is faster, 512 is more detailed |
| `num_neighbors` | `config.py → PatchCoreConfig` | 9 | More neighbors = smoother scores, less sensitive to single-patch noise |
| `threshold_percentile` | `config.py → TrainingConfig` | 99.0 | Lower = more sensitive (more false positives). Raise if getting too many false alarms |
| `image_size` | `config.py → PreprocessConfig` | (256, 256) | Larger = more detail but more VRAM and time |
| `clahe_clip_limit` | `config.py → PreprocessConfig` | 2.0 | Controls contrast enhancement. Raise for dark/low-contrast images |

After changing `config.py`, retrain the model for changes to take effect.

---

## 10. Troubleshooting

**Everything is flagged as ANOMALY during inference:**
The model was likely trained on images from a different visual domain than your test images. Retrain using `--combine-normals` and make sure your training normal images represent the same kinds of photos you will test on.

**Confidence is always near 50%:**
This is correct behaviour for images that score very close to the threshold. The confidence formula is threshold-relative, not training-mean-relative (the old formula always gave 100%).

**VRAM out of memory during training:**
Add `--batch-size 16` or `--target-dim 128`. The coreset step is the most memory-intensive; lowering `--coreset-ratio` also helps.

**Corrosion analysis shows 0% HM Growth for all timepoints:**
Check that your T0 images are included and named correctly (`-T0` suffix). The T0 heatmaps are required to compute the per-component baseline. Without them the growth metric cannot be computed.

**Colour rust % stays near 0% for black components:**
This is expected. The colour method detects orange/brown pixels and cannot see rust under a black coating. Use the HM Growth % column instead, which detects texture change regardless of surface colour. See INTERPRET.md for a full explanation.

**Score distribution in `score_dist.png` shows heavy overlap:**
The dataset is the main constraint — more diverse normal training images will improve separation. Consider sourcing additional clean images and retraining with `--combine-normals`.
