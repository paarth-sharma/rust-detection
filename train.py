#!/usr/bin/env python3
"""
Train PatchCore on fastener images for corrosion detection.

Uses the data/train, data/valid, data/test splits with _classes.csv labels.
PatchCore trains on non-corroded (healthy) images only, then detects
corroded fasteners as anomalies via nearest-neighbor distance.

Usage:
    python train.py
    python train.py --coreset-ratio 0.1 --target-dim 256
    python train.py --preprocess-only
"""

import argparse, csv, sys, cv2, numpy as np, torch
from pathlib import Path
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent))
from config import cfg, CORRODED_COLUMNS
from preprocessing.pipeline import FastenerPreprocessor
from models.patchcore import PatchCore
from evaluation.metrics import compute_image_metrics, print_metrics
from evaluation.visualize import plot_score_distribution, plot_roc, comparison_figure


def load_components_normal(components_dir, baseline_tag="T0"):
    """Load T0 (baseline) images from the components directory as normal examples.

    These are real-world reference images of the components under test.  They
    must be included in the memory bank so the model understands what deployed
    components look like at healthy state — the train/valid/test splits alone
    are stock/web images with a different visual distribution.

    Args:
        components_dir: Path to data/components (or similar).
        baseline_tag: Filename substring identifying the baseline timepoint (default "T0").

    Returns:
        List of Path objects for matched files.
    """
    components_dir = Path(components_dir)
    if not components_dir.exists():
        return []

    exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
    paths = [
        p for p in components_dir.iterdir()
        if p.suffix.lower() in exts and baseline_tag.lower() in p.stem.lower()
    ]
    return sorted(paths)


def load_split(split_dir):
    """Load image paths and labels from a split directory with _classes.csv.

    Returns dict with:
        paths: list of Path objects
        labels: list of int (1=corroded/anomalous, 0=normal)
        normal_paths: list of Path (non-corroded only)
        anomalous_paths: list of Path (corroded only)
    """
    split_dir = Path(split_dir)
    csv_path = split_dir / "_classes.csv"

    if not csv_path.exists():
        print(f"  [ERROR] No _classes.csv in {split_dir}")
        return None

    paths, labels = [], []
    normal_paths, anomalous_paths = [], []

    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            img_path = split_dir / row["filename"]
            if not img_path.exists():
                continue

            is_corroded = any(int(row.get(col, 0)) == 1 for col in CORRODED_COLUMNS)
            label = 1 if is_corroded else 0

            paths.append(img_path)
            labels.append(label)

            if is_corroded:
                anomalous_paths.append(img_path)
            else:
                normal_paths.append(img_path)

    return {
        "paths": paths,
        "labels": labels,
        "normal_paths": normal_paths,
        "anomalous_paths": anomalous_paths,
    }


def preprocess_images(paths, pp, desc="Preprocessing"):
    out, fail = [], 0
    for p in tqdm(paths, desc=f"  {desc}"):
        r = pp.process_file(p)
        if r.success:
            out.append(r.image)
        else:
            print(f"    [WARN] {p.name}: {r.error}")
            fail += 1
    if fail:
        print(f"  {fail}/{len(paths)} failed")
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--data-dir", default=None, help="Root data directory (default: data/)")
    pa.add_argument("--preprocess-only", action="store_true")
    pa.add_argument("--coreset-ratio", type=float, default=0.2,
                    help="Coreset sampling ratio (default: 0.2). Higher = larger memory bank.")
    pa.add_argument("--target-dim", type=int, default=256)
    pa.add_argument("--batch-size", type=int, default=32)
    pa.add_argument("--device", default=None)
    pa.add_argument("--combine-normals", action="store_true",
                    help="Train on normal images from ALL splits (train+valid+test). "
                         "Greatly improves generalisation when splits have different visual domains. "
                         "An 80/20 internal shuffle is used for threshold calibration.")
    pa.add_argument("--components-dir", default=None,
                    help="Path to real-world component images (default: data/components). "
                         "T0 baseline images are added to the normal training pool so the "
                         "model learns the deployed domain, not just the stock-image domain. "
                         "Pass an empty string to disable.")
    pa.add_argument("--baseline-tag", default="T0",
                    help="Filename substring identifying baseline component images (default: T0).")
    args = pa.parse_args()

    device = args.device or cfg.training.device
    print(f"\n  Device: {device}")
    if device == "cuda":
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    torch.manual_seed(cfg.training.seed)
    np.random.seed(cfg.training.seed)

    data_dir = Path(args.data_dir) if args.data_dir else cfg.paths.data_dir

    # Load all three splits
    print("\n  Loading datasets...")
    train_split = load_split(data_dir / "train")
    valid_split = load_split(data_dir / "valid")
    test_split = load_split(data_dir / "test")

    # Load real-world component baselines (T0 images).
    # These are the images the model will actually be deployed on.  Without them
    # the memory bank only knows stock/web-scrape images, causing a distribution
    # shift that makes even healthy real-world components score as anomalous.
    if args.components_dir == "":
        components_normal_paths = []
    else:
        comp_dir = Path(args.components_dir) if args.components_dir else data_dir / "components"
        components_normal_paths = load_components_normal(comp_dir, args.baseline_tag)

    if components_normal_paths:
        print(f"  Components (T0 baseline): {len(components_normal_paths)} images from {comp_dir}")
    else:
        print("  Components: none found (pass --components-dir to specify)")

    if not train_split:
        print("  [ERROR] Could not load train split"); sys.exit(1)

    print(f"  Train: {len(train_split['normal_paths'])} normal, "
          f"{len(train_split['anomalous_paths'])} corroded")
    if valid_split:
        print(f"  Valid: {len(valid_split['normal_paths'])} normal, "
              f"{len(valid_split['anomalous_paths'])} corroded")
    if test_split:
        print(f"  Test:  {len(test_split['normal_paths'])} normal, "
              f"{len(test_split['anomalous_paths'])} corroded")

    if not train_split["normal_paths"]:
        print("  [ERROR] No normal (non-corroded) training images!"); sys.exit(1)

    # Optionally pool all normal images across splits for a larger, more diverse
    # memory bank.  Uses an 80/20 internal shuffle for threshold calibration so
    # no separate validation split is needed.
    import random
    if args.combine_normals:
        # Pool stock normals across all splits, shuffle, 80/20 split.
        # Components T0 images are added to training ONLY — never calibration.
        # If they land in the calibration set their high out-of-domain scores
        # push the 99th-percentile threshold up, tanking stock-test recall.
        stock_normals = list(train_split["normal_paths"])
        if valid_split:
            stock_normals += valid_split["normal_paths"]
        if test_split:
            stock_normals += test_split["normal_paths"]
        random.seed(cfg.training.seed)
        random.shuffle(stock_normals)
        calib_n = max(1, int(len(stock_normals) * 0.2))
        calib_paths_combined = stock_normals[:calib_n]
        train_normal_paths = stock_normals[calib_n:] + components_normal_paths
        print(f"\n  --combine-normals: {len(stock_normals)} stock normals "
              f"+ {len(components_normal_paths)} component baselines (train only)"
              f" → {len(train_normal_paths)} train / {len(calib_paths_combined)} calib")
    else:
        # Without --combine-normals: T0 components go into the memory bank.
        # Threshold calibration stays on stock validation images only so the
        # test-set threshold is not corrupted by high OOD component scores.
        train_normal_paths = train_split["normal_paths"] + components_normal_paths
        calib_paths_combined = None  # calibration will use valid_split (stock only)

    # Build preprocessor
    pp = FastenerPreprocessor(
        target_size=cfg.preprocess.image_size,
        clahe_clip_limit=cfg.preprocess.clahe_clip_limit,
        clahe_tile_grid=cfg.preprocess.clahe_tile_grid,
        blur_ksize=cfg.preprocess.gaussian_blur_ksize,
        morph_kernel_size=cfg.preprocess.morph_kernel_size,
        morph_iterations=cfg.preprocess.morph_iterations,
        min_component_area=cfg.preprocess.min_component_area_ratio,
        padding_fraction=cfg.preprocess.padding_fraction,
        preserve_color=cfg.preprocess.preserve_color,
        align=cfg.preprocess.align_to_canonical,
    )

    # Preprocess training images (normal only — PatchCore paradigm)
    print("\n  Preprocessing normal training images...")
    train_images = preprocess_images(train_normal_paths, pp, "Train (normal)")

    # Save sample preprocessed images
    pp_dir = cfg.paths.output_dir / "preprocessed"
    pp_dir.mkdir(parents=True, exist_ok=True)
    for i, img in enumerate(train_images[:10]):
        cv2.imwrite(str(pp_dir / f"train_normal_{i:03d}.png"), img)

    if args.preprocess_only:
        if test_split:
            test_images = preprocess_images(test_split["paths"], pp, "Test")
            for i, img in enumerate(test_images[:20]):
                tag = "corroded" if test_split["labels"][i] == 1 else "normal"
                cv2.imwrite(str(pp_dir / f"test_{tag}_{i:03d}.png"), img)
        print("  Preprocessing done.")
        return

    # Preprocess validation images for threshold calibration
    val_images = None
    if args.combine_normals and calib_paths_combined:
        print("\n  Preprocessing combined calibration images...")
        val_images = preprocess_images(calib_paths_combined, pp, "Calib (normal)")
    elif valid_split and valid_split["normal_paths"]:
        print("\n  Preprocessing normal validation images...")
        val_images = preprocess_images(valid_split["normal_paths"], pp, "Valid (normal)")

    # Fit PatchCore on normal training images
    model = PatchCore(
        backbone=cfg.backbone.name,
        layers=cfg.backbone.layers,
        target_dim=args.target_dim,
        coreset_sampling_ratio=args.coreset_ratio,
        num_neighbors=cfg.patchcore.num_neighbors,
        device=device,
        image_size=cfg.preprocess.image_size,
    )
    model.fit(train_images, args.batch_size, val_images, cfg.training.threshold_percentile)

    # Evaluate on test set
    if test_split and test_split["paths"]:
        print("\n  Evaluating on test set...")
        test_images = preprocess_images(test_split["paths"], pp, "Test")
        test_labels = np.array(test_split["labels"][: len(test_images)])
        test_scores, test_hmaps = model.predict_batch(test_images)

        print_metrics(
            compute_image_metrics(test_scores, test_labels, model.threshold),
            "Test Set Results",
        )

        # Visualizations
        vd = cfg.paths.output_dir / "visualizations"
        vd.mkdir(parents=True, exist_ok=True)

        normal_scores = test_scores[test_labels == 0]
        anomaly_scores = test_scores[test_labels == 1]

        if len(normal_scores) > 0:
            plot_score_distribution(
                normal_scores,
                anomaly_scores if len(anomaly_scores) > 0 else None,
                model.threshold,
                str(vd / "score_dist.png"),
            )

        if len(np.unique(test_labels)) > 1:
            plot_roc(test_labels, test_scores, str(vd / "roc.png"))

        # Save comparison figures for a sample of test images
        for i in range(min(10, len(test_images))):
            tag = "corroded" if test_labels[i] == 1 else "normal"
            comparison_figure(
                test_images[i],
                test_hmaps[i],
                test_scores[i],
                model.threshold,
                title=f"{test_split['paths'][i].name} — {tag.upper()}",
                save_path=str(vd / f"test_{tag}_{i:03d}.png"),
            )

        print(f"  Visualizations saved to {vd}")

    # Save model
    model.save(str(cfg.paths.model_dir / "patchcore_fastener.pkl"))
    print(f"\n  DONE | Threshold: {model.threshold:.4f}\n")


if __name__ == "__main__":
    main()
