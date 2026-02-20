##  Step 1: Activate the virtual environment
```
  cd ~/.vscode/rust-detection
  source .venv/bin/activate
```

##  Step 2: Verify setup
```
  python verify_setup.py
```
  This checks all dependencies, runs a smoke test on the preprocessing pipeline
   and feature extractor, and reports how many normal/corroded images are in
  each split. Make sure everything shows a checkmark.

##  Step 3: Preview preprocessing (optional but recommended)
```
  python train.py --preprocess-only
```
  This runs the full preprocessing pipeline on your images without training the
   model. Check outputs/preprocessed/ to verify the images look correctly
  segmented, aligned, and normalized before committing to a full training run.

##  Step 4: Train the model
```
  python train.py
```
  This will:
  1. Parse _classes.csv in each split and separate normal from corroded
  2. Preprocess normal training images (lighting, segmentation, alignment)
  3. Extract patch features via frozen WideResNet-50-2 on your RTX 4060
  4. Build the memory bank and apply 10% coreset subsampling
  5. Calibrate the anomaly threshold on normal validation images (99th
  percentile)
  6. Evaluate on the test set and print AUROC, F1, precision, recall
  7. Save visualizations to outputs/visualizations/
  8. Save the model to saved_models/patchcore_fastener.pkl

  Optional flags:
```
  python train.py --coreset-ratio 0.05    # smaller memory bank (faster
  inference)
  python train.py --target-dim 128        # lower feature dim (less memory)
  python train.py --batch-size 16         # if you hit VRAM limits
```

##  Step 5: Check results

  Look at the generated outputs:
```
  ls outputs/visualizations/
```

  - ```score_dist.png``` — distribution of anomaly scores for normal vs corroded test
   images
  - ```roc.png``` — ROC curve with AUROC value
  - ```test_normal_*.png / test_corroded_*.png``` — per-image comparison figures
  (original, heatmap overlay, detection)

##  Step 6: Run inference on new images

  #### Single image
  ```
  python inference.py --model saved_models/patchcore_fastener.pkl --image path/to/image.jpg
  ```

  #### Entire folder with heatmap visualizations
  ```
  python inference.py --model saved_models/patchcore_fastener.pkl --dir path/to/folder/ --visualize
```

  #### Export results to JSON
  ```
  python inference.py --model saved_models/patchcore_fastener.pkl --dir path/to/folder/ --json
  ```

  Each image gets a verdict (NORMAL/ANOMALY), an anomaly score, a confidence
  percentage, and a sigma value (standard deviations from normal). With
  --visualize, you also get heatmap overlays showing where the corrosion was
  detected.