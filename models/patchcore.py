"""
PatchCore (Roth et al., CVPR 2022) — from scratch.

1. Extract patch features from frozen CNN
2. Store healthy patches in memory bank
3. Coreset subsampling to reduce bank
4. Score via nearest-neighbor distance (FAISS)
5. Image score = max patch distance; heatmap = upsampled distances
"""

import numpy as np
import torch
import torch.nn.functional as F
import faiss
import pickle
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from tqdm import tqdm
from scipy.ndimage import gaussian_filter
from scipy.stats import norm

from .feature_extractor import FeatureExtractor, get_imagenet_transforms
from .coreset import greedy_coreset_subsampling


class PatchCore:
    def __init__(
        self,
        backbone: str = "wide_resnet50_2",
        layers: list = None,
        target_dim: int = 256,
        coreset_sampling_ratio: float = 0.1,
        num_neighbors: int = 9,
        device: str = "cuda",
        image_size: Tuple[int, int] = (256, 256),
    ):
        self.backbone_name = backbone
        self.layers = layers or ["layer2", "layer3"]
        self.target_dim = target_dim
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.num_neighbors = num_neighbors
        self.device = device if torch.cuda.is_available() else "cpu"
        self.image_size = image_size

        self.feature_extractor = FeatureExtractor(
            backbone_name=backbone, layers=self.layers,
            pretrained=True, target_dim=target_dim,
        ).to(self.device)
        self.feature_extractor.eval()

        self.transform = get_imagenet_transforms()

        self.memory_bank: Optional[np.ndarray] = None
        self.faiss_index: Optional[faiss.Index] = None
        self.threshold: float = 0.0
        self.train_scores_mean: float = 0.0
        self.train_scores_std: float = 0.0
        self.spatial_h = self.feature_extractor.spatial_size
        self.spatial_w = self.feature_extractor.spatial_size

    def _image_to_tensor(self, image: np.ndarray) -> torch.Tensor:
        rgb = image[:, :, ::-1].copy()
        return self.transform(rgb)

    def _extract_features_batch(self, images: list, batch_size: int = 32) -> np.ndarray:
        all_features = []
        for i in tqdm(range(0, len(images), batch_size), desc="  Extracting features"):
            batch_imgs = images[i:i+batch_size]
            tensors = [self._image_to_tensor(img) for img in batch_imgs]
            batch = torch.stack(tensors).to(self.device)
            with torch.no_grad():
                features = self.feature_extractor.extract_patch_features(batch)
            B, P, D = features.shape
            all_features.append(features.cpu().numpy().reshape(-1, D))
        return np.concatenate(all_features, axis=0)

    def fit(
        self, images: list, batch_size: int = 32,
        val_images: Optional[list] = None,
        threshold_percentile: float = 99.0,
    ) -> Dict[str, Any]:
        print(f"\n{'='*60}")
        print(f"  PatchCore Fitting")
        print(f"  Images: {len(images)} | Backbone: {self.backbone_name}")
        print(f"  Layers: {self.layers} | Feature dim: {self.feature_extractor.feature_dim}")
        print(f"  Coreset ratio: {self.coreset_sampling_ratio}")
        print(f"{'='*60}\n")

        # Step 1: Extract features
        print("Step 1/3: Extracting features...")
        all_features = self._extract_features_batch(images, batch_size)
        print(f"  Total: {all_features.shape[0]} patches × {all_features.shape[1]}D")

        # Step 2: Coreset
        print("\nStep 2/3: Coreset subsampling...")
        self.memory_bank, _ = greedy_coreset_subsampling(
            all_features, self.coreset_sampling_ratio, self.device)
        print(f"  Memory bank: {self.memory_bank.shape[0]} patches")

        # Step 3: FAISS index
        print("\nStep 3/3: Building FAISS index...")
        self._build_faiss_index()

        stats = {
            "num_images": len(images),
            "total_patches": all_features.shape[0],
            "memory_bank_size": self.memory_bank.shape[0],
            "feature_dim": self.memory_bank.shape[1],
        }

        if val_images:
            print(f"\nCalibrating threshold on {len(val_images)} validation images...")
            self._calibrate_threshold(val_images, batch_size, threshold_percentile)
            stats["threshold"] = self.threshold
            print(f"  Threshold: {self.threshold:.4f}")
            print(f"  Normal scores: {self.train_scores_mean:.4f} ± {self.train_scores_std:.4f}")

        print(f"\n  ✓ Fit complete.\n")
        return stats

    def _build_faiss_index(self):
        d = self.memory_bank.shape[1]
        self.faiss_index = faiss.IndexFlatL2(d)
        self.faiss_index.add(self.memory_bank.astype(np.float32))

    def _calibrate_threshold(self, val_images, batch_size=32, percentile=99.0):
        scores = [self.predict(img)[0] for img in val_images]
        scores = np.array(scores)
        self.threshold = float(np.percentile(scores, percentile))
        self.train_scores_mean = float(np.mean(scores))
        self.train_scores_std = float(np.std(scores))

    def predict(self, image: np.ndarray) -> Tuple[float, np.ndarray]:
        tensor = self._image_to_tensor(image).unsqueeze(0).to(self.device)
        with torch.no_grad():
            features = self.feature_extractor.extract_patch_features(tensor)
        patches = features[0].cpu().numpy().astype(np.float32)

        distances, _ = self.faiss_index.search(patches, self.num_neighbors)
        patch_scores = distances[:, 0]

        # Re-weighting from PatchCore paper (Section 3.3)
        if self.num_neighbors > 1:
            nn_dists = distances[:, 1:]
            weight = 1.0 - np.exp(-nn_dists.max(axis=1))
            patch_scores = patch_scores * (1.0 + weight)

        image_score = float(np.max(patch_scores))

        # Upsample to image size
        heatmap = patch_scores.reshape(self.spatial_h, self.spatial_w)
        heatmap_t = torch.from_numpy(heatmap).unsqueeze(0).unsqueeze(0).float()
        heatmap_up = F.interpolate(heatmap_t, size=self.image_size, mode='bilinear', align_corners=False)
        heatmap_full = gaussian_filter(heatmap_up.squeeze().numpy(), sigma=4)

        return image_score, heatmap_full

    def predict_batch(self, images: list, batch_size: int = 32):
        scores, heatmaps = [], []
        for img in tqdm(images, desc="  Predicting"):
            s, h = self.predict(img)
            scores.append(s)
            heatmaps.append(h)
        return np.array(scores), heatmaps

    def classify(self, image: np.ndarray) -> Dict[str, Any]:
        score, heatmap = self.predict(image)
        is_anomalous = score > self.threshold if self.threshold > 0 else None
        sigma = (score - self.train_scores_mean) / self.train_scores_std if self.train_scores_std > 0 else 0.0
        confidence = norm.cdf(sigma) * 100 if sigma > 0 else (1 - norm.cdf(abs(sigma))) * 100
        return {
            "is_anomalous": is_anomalous, "score": score,
            "threshold": self.threshold, "confidence": confidence,
            "sigma_from_normal": sigma, "heatmap": heatmap,
        }

    def save(self, path: str):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        state = {
            "memory_bank": self.memory_bank,
            "threshold": self.threshold,
            "train_scores_mean": self.train_scores_mean,
            "train_scores_std": self.train_scores_std,
            "config": {
                "backbone": self.backbone_name, "layers": self.layers,
                "target_dim": self.target_dim,
                "coreset_sampling_ratio": self.coreset_sampling_ratio,
                "num_neighbors": self.num_neighbors,
                "image_size": self.image_size,
            },
        }
        with open(str(path), "wb") as f:
            pickle.dump(state, f)
        print(f"  ✓ Model saved: {path} (bank: {self.memory_bank.shape})")

    @classmethod
    def load(cls, path: str, device: str = "cuda") -> "PatchCore":
        with open(str(path), "rb") as f:
            state = pickle.load(f)
        c = state["config"]
        model = cls(backbone=c["backbone"], layers=c["layers"], target_dim=c["target_dim"],
                    coreset_sampling_ratio=c["coreset_sampling_ratio"],
                    num_neighbors=c["num_neighbors"], device=device,
                    image_size=tuple(c["image_size"]))
        model.memory_bank = state["memory_bank"]
        model.threshold = state["threshold"]
        model.train_scores_mean = state["train_scores_mean"]
        model.train_scores_std = state["train_scores_std"]
        model._build_faiss_index()
        print(f"  ✓ Model loaded: {path} (bank: {model.memory_bank.shape}, threshold: {model.threshold:.4f})")
        return model
