#!/usr/bin/env python3
"""
Run inference on fastener images.

Usage:
    python inference.py --model saved_models/patchcore_fastener.pkl --image test.png
    python inference.py --model saved_models/patchcore_fastener.pkl --dir test_images/ --visualize
"""

import argparse, sys, cv2, json, numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from config import cfg
from preprocessing.pipeline import FastenerPreprocessor
from models.patchcore import PatchCore
from evaluation.visualize import comparison_figure, heatmap_overlay


def run_one(model, pp, img_path, out_dir=None, viz=True):
    image = cv2.imread(str(img_path))
    if image is None:
        return {"error": f"Can't read: {img_path}", "path": str(img_path)}

    result = pp.process_file(img_path)
    if not result.success:
        return {"error": f"Preprocess failed: {result.error}", "path": str(img_path)}

    det = model.classify(result.image)
    status = "ANOMALY" if det["is_anomalous"] else "NORMAL"
    c = "\033[91m" if det["is_anomalous"] else "\033[92m"
    print(f"  {c}{status}\033[0m | score={det['score']:.4f} | "
          f"σ={det['sigma_from_normal']:.1f} | conf={det['confidence']:.1f}% | {img_path.name}")

    out = {"path": str(img_path), "is_anomalous": det["is_anomalous"],
           "score": float(det["score"]), "threshold": float(det["threshold"]),
           "confidence": float(det["confidence"]), "sigma": float(det["sigma_from_normal"])}

    if viz and out_dir:
        out_dir.mkdir(parents=True, exist_ok=True)
        comparison_figure(result.image, det["heatmap"], det["score"], det["threshold"],
                                 title=f"{img_path.name} — {status}", save_path=str(out_dir / f"det_{img_path.stem}.png"))
        overlay = heatmap_overlay(result.image, det["heatmap"])
        cv2.imwrite(str(out_dir / f"overlay_{img_path.stem}.png"), overlay)
    return out


def main():
    pa = argparse.ArgumentParser()
    pa.add_argument("--model", required=True); pa.add_argument("--image", default=None)
    pa.add_argument("--dir", default=None); pa.add_argument("--output", default=None)
    pa.add_argument("--visualize", action="store_true"); pa.add_argument("--device", default=None)
    pa.add_argument("--json", action="store_true")
    args = pa.parse_args()
    if not args.image and not args.dir: pa.error("Provide --image or --dir")

    model = PatchCore.load(args.model, args.device or cfg.training.device)
    pp = FastenerPreprocessor(target_size=cfg.preprocess.image_size,
        clahe_clip_limit=cfg.preprocess.clahe_clip_limit, clahe_tile_grid=cfg.preprocess.clahe_tile_grid,
        preserve_color=cfg.preprocess.preserve_color, align=cfg.preprocess.align_to_canonical)

    out_dir = Path(args.output) if args.output else cfg.paths.output_dir / "inference"
    results = []

    if args.image:
        results.append(run_one(model, pp, Path(args.image), out_dir, args.visualize))
    if args.dir:
        exts = {".png", ".jpg", ".jpeg", ".bmp", ".tiff"}
        paths = sorted(f for f in Path(args.dir).iterdir() if f.suffix.lower() in exts)
        print(f"\n  Processing {len(paths)} images\n")
        for p in paths:
            results.append(run_one(model, pp, p, out_dir, args.visualize))

    valid = [r for r in results if "error" not in r]
    if valid:
        anom = sum(1 for r in valid if r["is_anomalous"])
        print(f"\n  Summary: {len(valid)-anom} normal, {anom} anomalous / {len(valid)} total")

    if args.json:
        jp = out_dir / "results.json"; jp.parent.mkdir(parents=True, exist_ok=True)
        with open(jp, "w") as f: json.dump(results, f, indent=2)
        print(f"  JSON → {jp}")


if __name__ == "__main__":
    main()
