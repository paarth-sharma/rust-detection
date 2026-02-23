#!/usr/bin/env python3
"""
Run inference on fastener images.

Usage:
    python inference.py --model saved_models/patchcore_fastener.pkl --image test.png
    python inference.py --model saved_models/patchcore_fastener.pkl --dir test_images/ --visualize
    python inference.py --model saved_models/patchcore_fastener.pkl --dir test_images/ --baseline-tag T0 --adaptive-threshold

Baseline handling:
    Images whose filename contains --baseline-tag (default: T0) are always reported
    as BASELINE (healthy reference), regardless of their anomaly score.

    With --adaptive-threshold the threshold is raised to sit above all baseline
    scores (+ 10% margin) before scoring successors, so components that were
    already slightly worn at T0 don't produce spurious ANOMALY flags on later
    images that are only marginally worse.
"""

import argparse, sys, cv2, json
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import cfg
from preprocessing.pipeline import FastenerPreprocessor
from models.patchcore import PatchCore
from evaluation.visualize import comparison_figure, heatmap_overlay


def is_baseline(img_path: Path, tag: str) -> bool:
    """Return True if the filename contains the baseline tag (case-insensitive)."""
    return tag.lower() in img_path.stem.lower()


def run_one(model, pp, img_path, out_dir=None, viz=True,
            baseline_override=False, effective_threshold=None):
    image = cv2.imread(str(img_path))
    if image is None:
        return {"error": f"Can't read: {img_path}", "path": str(img_path)}

    result = pp.process_file(img_path)
    if not result.success:
        return {"error": f"Preprocess failed: {result.error}", "path": str(img_path)}

    det = model.classify(result.image)
    score = det["score"]
    thr = effective_threshold if effective_threshold is not None else det["threshold"]

    if baseline_override:
        is_anomalous = False
        status = "BASELINE"
        c = "\033[94m"   # blue
    else:
        is_anomalous = (score > thr) if thr > 0 else bool(det["is_anomalous"])
        status = "ANOMALY" if is_anomalous else "NORMAL"
        c = "\033[91m" if is_anomalous else "\033[92m"

    print(f"  {c}{status}\033[0m | score={score:.4f} | thr={thr:.4f} | "
          f"σ={det['sigma_from_normal']:.1f} | conf={det['confidence']:.1f}% | {img_path.name}")

    out = {
        "path": str(img_path),
        "is_anomalous": is_anomalous,
        "is_baseline": baseline_override,
        "score": float(score),
        "threshold": float(thr),
        "confidence": float(det["confidence"]),
        "sigma": float(det["sigma_from_normal"]),
    }

    if viz and out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        comparison_figure(
            result.image, det["heatmap"], score, thr,
            title=f"{img_path.name} — {status}",
            save_path=str(out_dir / f"det_{img_path.stem}.png"),
        )
        overlay = heatmap_overlay(result.image, det["heatmap"])
        cv2.imwrite(str(out_dir / f"overlay_{img_path.stem}.png"), overlay)

    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", required=True)
    pa.add_argument("--image", default=None)
    pa.add_argument("--dir", default=None)
    pa.add_argument("--output", default=None)
    pa.add_argument("--visualize", action="store_true")
    pa.add_argument("--device", default=None)
    pa.add_argument("--json", action="store_true")
    pa.add_argument(
        "--baseline-tag", default="T0",
        help="Filename substring that identifies baseline images (default: T0). "
             "These are always reported as HEALTHY/BASELINE.",
    )
    pa.add_argument(
        "--adaptive-threshold", action="store_true",
        help="Raise the anomaly threshold to sit above all baseline image scores "
             "(+ 10%% margin) before evaluating successors.",
    )
    args = pa.parse_args()
    if not args.image and not args.dir:
        pa.error("Provide --image or --dir")

    model = PatchCore.load(args.model, args.device or cfg.training.device)
    pp = FastenerPreprocessor(
        target_size=cfg.preprocess.image_size,
        clahe_clip_limit=cfg.preprocess.clahe_clip_limit,
        clahe_tile_grid=cfg.preprocess.clahe_tile_grid,
        preserve_color=cfg.preprocess.preserve_color,
        align=cfg.preprocess.align_to_canonical,
    )

    out_dir = Path(args.output) if args.output else cfg.paths.output_dir / "inference"
    results = []

    if args.image:
        p = Path(args.image)
        override = is_baseline(p, args.baseline_tag)
        results.append(run_one(model, pp, p, out_dir, args.visualize,
                               baseline_override=override))

    if args.dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        paths = sorted(f for f in Path(args.dir).iterdir() if f.suffix.lower() in exts)
        print(f"\n  Processing {len(paths)} images  (baseline tag: '{args.baseline_tag}')\n")

        # Start with the model's calibrated threshold.
        effective_threshold = model.threshold

        if args.adaptive_threshold:
            baseline_paths = [p for p in paths if is_baseline(p, args.baseline_tag)]
            if baseline_paths:
                print(f"  Calibrating adaptive threshold on {len(baseline_paths)} "
                      f"baseline image(s)...")
                baseline_scores = []
                for p in baseline_paths:
                    res = pp.process_file(p)
                    if res.success:
                        s, _ = model.predict(res.image)
                        baseline_scores.append(s)

                if baseline_scores:
                    adaptive_thr = float(np.max(baseline_scores)) * 1.10
                    if adaptive_thr > effective_threshold:
                        print(f"  Threshold raised: {effective_threshold:.4f} → "
                              f"{adaptive_thr:.4f}  "
                              f"(max baseline score {max(baseline_scores):.4f} + 10% margin)")
                        effective_threshold = adaptive_thr
                    else:
                        print(f"  Threshold unchanged at {effective_threshold:.4f} "
                              f"(baseline scores already below model threshold)")
                print()

        for p in paths:
            override = is_baseline(p, args.baseline_tag)
            results.append(run_one(model, pp, p, out_dir, args.visualize,
                                   baseline_override=override,
                                   effective_threshold=effective_threshold))

    valid = [r for r in results if "error" not in r]
    if valid:
        baselines = sum(1 for r in valid if r.get("is_baseline"))
        anom = sum(1 for r in valid if r["is_anomalous"])
        normal = len(valid) - baselines - anom
        print(f"\n  Summary: {baselines} baseline | {normal} normal | "
              f"{anom} anomalous / {len(valid)} total")

    if args.json:
        jp = out_dir / "results.json"
        jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w") as f:
            json.dump(results, f, indent=2)
        print(f"  JSON → {jp}")


if __name__ == "__main__":
    main()
