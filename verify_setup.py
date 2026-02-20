#!/usr/bin/env python3
"""
Verify your setup is working correctly.
Run this after installing dependencies: python verify_setup.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))


def check(name, fn):
    try:
        result = fn()
        print(f"  ✓ {name}" + (f" ({result})" if result else ""))
        return True
    except Exception as e:
        print(f"  ✗ {name}: {e}")
        return False


def main():
    print("\n" + "="*60)
    print("  SETUP VERIFICATION")
    print("="*60 + "\n")
    ok = True

    print("--- Python Dependencies ---")
    ok &= check("numpy", lambda: __import__("numpy").__version__)
    ok &= check("opencv", lambda: __import__("cv2").__version__)
    ok &= check("scikit-learn", lambda: __import__("sklearn").__version__)
    ok &= check("scipy", lambda: __import__("scipy").__version__)
    ok &= check("matplotlib", lambda: __import__("matplotlib").__version__)
    ok &= check("faiss", lambda: __import__("faiss").__version__)
    ok &= check("tqdm", lambda: __import__("tqdm").__version__)

    print("\n--- PyTorch ---")
    torch_ok = check("torch", lambda: __import__("torch").__version__)
    ok &= torch_ok
    if torch_ok:
        import torch
        ok &= check("torchvision", lambda: __import__("torchvision").__version__)
        cuda = torch.cuda.is_available()
        check("CUDA", lambda: str(cuda))
        if cuda:
            check("GPU", lambda: torch.cuda.get_device_name(0))
            check("VRAM", lambda: f"{torch.cuda.get_device_properties(0).total_mem/1e9:.1f} GB")

    print("\n--- Project Modules ---")
    ok &= check("config", lambda: (__import__("config"), "ok")[1])
    ok &= check("preprocessing.lighting", lambda: (__import__("preprocessing.lighting", fromlist=["apply_clahe"]), "ok")[1])
    ok &= check("preprocessing.segmentation", lambda: (__import__("preprocessing.segmentation", fromlist=["create_component_mask"]), "ok")[1])
    ok &= check("preprocessing.registration", lambda: (__import__("preprocessing.registration", fromlist=["align_to_canonical"]), "ok")[1])
    ok &= check("preprocessing.pipeline", lambda: (__import__("preprocessing.pipeline", fromlist=["FastenerPreprocessor"]), "ok")[1])
    ok &= check("evaluation.metrics", lambda: (__import__("evaluation.metrics", fromlist=["compute_image_metrics"]), "ok")[1])
    ok &= check("evaluation.visualize", lambda: (__import__("evaluation.visualize", fromlist=["create_heatmap_overlay"]), "ok")[1])

    if torch_ok:
        ok &= check("models.feature_extractor", lambda: (__import__("models.feature_extractor", fromlist=["FeatureExtractor"]), "ok")[1])
        ok &= check("models.coreset", lambda: (__import__("models.coreset", fromlist=["greedy_coreset_subsampling"]), "ok")[1])
        ok &= check("models.patchcore", lambda: (__import__("models.patchcore", fromlist=["PatchCore"]), "ok")[1])

    print("\n--- Preprocessing Smoke Test ---")
    import cv2, numpy as np
    from preprocessing.pipeline import FastenerPreprocessor
    pp = FastenerPreprocessor(target_size=(256, 256))
    img = np.ones((400, 300, 3), dtype=np.uint8) * 200
    cv2.rectangle(img, (100, 50), (200, 280), (80, 80, 90), -1)
    r = pp.process(img)
    ok &= check("synthetic image", lambda: f"shape={r.image.shape} success={r.success}")

    if torch_ok:
        print("\n--- Feature Extractor Smoke Test ---")
        import torch
        from models.feature_extractor import FeatureExtractor, get_imagenet_transforms
        try:
            fe = FeatureExtractor(target_dim=256)
            dev = "cuda" if torch.cuda.is_available() else "cpu"
            fe = fe.to(dev)
            t = get_imagenet_transforms()(r.image[:,:,::-1].copy()).unsqueeze(0).to(dev)
            with torch.no_grad():
                f = fe.extract_patch_features(t)
            check("feature extraction", lambda: f"patches={f.shape[1]} dim={f.shape[2]}")
            fe.remove_hooks()
        except Exception as e:
            print(f"  ✗ feature extraction: {e}"); ok = False

    print("\n--- Dataset Status ---")
    from config import cfg, CORRODED_COLUMNS
    import csv as csv_mod
    for split in ["train", "valid", "test"]:
        split_dir = cfg.paths.data_dir / split
        csv_path = split_dir / "_classes.csv"
        if csv_path.exists():
            normal, corroded = 0, 0
            with open(csv_path) as f:
                for row in csv_mod.DictReader(f):
                    is_corr = any(int(row.get(c, 0)) == 1 for c in CORRODED_COLUMNS)
                    if is_corr:
                        corroded += 1
                    else:
                        normal += 1
            print(f"  ✓ {split}: {normal} normal, {corroded} corroded")
        else:
            print(f"  ✗ {split}: _classes.csv not found")

    print("\n" + "="*60)
    print(f"  {'✓ ALL OK' if ok else '✗ SOME CHECKS FAILED'}")
    print("="*60 + "\n")


if __name__ == "__main__":
    main()
