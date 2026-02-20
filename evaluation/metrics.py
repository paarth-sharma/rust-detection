"""Evaluation metrics: AUROC, F1-max, pixel-level AUROC."""

import numpy as np
from sklearn.metrics import roc_auc_score, precision_recall_curve, confusion_matrix
from typing import Dict, Optional


def compute_image_metrics(scores, labels, threshold=None):
    metrics = {}
    if len(np.unique(labels)) > 1:
        metrics["auroc"] = roc_auc_score(labels, scores)
    else:
        metrics["auroc"] = float("nan")

    precision, recall, thresholds = precision_recall_curve(labels, scores)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-8)
    best_idx = np.argmax(f1_scores)
    metrics["f1_max"] = float(f1_scores[best_idx])
    metrics["f1_max_threshold"] = float(thresholds[best_idx]) if best_idx < len(thresholds) else 0.0

    if threshold is not None:
        preds = (scores > threshold).astype(int)
        if len(np.unique(preds)) > 1 and len(np.unique(labels)) > 1:
            tn, fp, fn, tp = confusion_matrix(labels, preds).ravel()
            metrics["accuracy"] = (tp + tn) / (tp + tn + fp + fn)
            metrics["precision"] = tp / (tp + fp) if (tp + fp) > 0 else 0
            metrics["recall"] = tp / (tp + fn) if (tp + fn) > 0 else 0
            metrics["f1"] = 2 * metrics["precision"] * metrics["recall"] / (
                metrics["precision"] + metrics["recall"] + 1e-8)
    return metrics


def compute_pixel_metrics(heatmaps, masks):
    all_s, all_l = [], []
    for hmap, mask in zip(heatmaps, masks):
        h = (hmap - hmap.min()) / (hmap.max() - hmap.min() + 1e-8)
        all_s.append(h.flatten())
        all_l.append((mask > 127).astype(int).flatten())
    s, l = np.concatenate(all_s), np.concatenate(all_l)
    return {"pixel_auroc": roc_auc_score(l, s) if len(np.unique(l)) > 1 else float("nan")}


def print_metrics(metrics, title="Results"):
    print(f"\n{'='*50}\n  {title}\n{'='*50}")
    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"  {k:25s}: {v:.4f}" + (f" ({v*100:.1f}%)" if v <= 1.0 else ""))
    print(f"{'='*50}\n")
