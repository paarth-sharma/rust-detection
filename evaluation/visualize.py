"""Visualization for anomaly detection results."""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional


def heatmap_overlay(image: np.ndarray, heatmap: np.ndarray,
                    alpha: float = 0.5, fg_mask: np.ndarray = None) -> np.ndarray:
    """Overlay anomaly heatmap on image. Both BGR.

    Args:
        image:   Background BGR image (H×W×3).
        heatmap: Anomaly score map (may differ in spatial size from image).
        alpha:   Blend weight for the heatmap colour layer.
        fg_mask: Optional uint8 foreground mask at *image* resolution.
                 When provided, the heatmap is normalised only within the
                 foreground so that background/padding pixels (which often
                 carry artificially high raw scores) don't corrupt the colour
                 scale.  Overlay blending is restricted to foreground pixels;
                 background pixels show the original image unchanged.
    """
    # Resize heatmap to image resolution first
    hm = heatmap.copy().astype(np.float32)
    if hm.shape[:2] != image.shape[:2]:
        hm = cv2.resize(hm, (image.shape[1], image.shape[0]),
                        interpolation=cv2.INTER_LINEAR)

    if fg_mask is not None:
        # Align fg_mask to image size just in case
        if fg_mask.shape[:2] != image.shape[:2]:
            fg_mask = cv2.resize(fg_mask, (image.shape[1], image.shape[0]),
                                 interpolation=cv2.INTER_NEAREST)
        fg = fg_mask > 0
        fg_vals = hm[fg]
        if fg_vals.size > 0 and fg_vals.max() > fg_vals.min():
            h_norm = np.zeros_like(hm)
            h_norm[fg] = (fg_vals - fg_vals.min()) / (fg_vals.max() - fg_vals.min())
        else:
            h_norm = np.zeros_like(hm)
    else:
        if hm.max() > hm.min():
            h_norm = (hm - hm.min()) / (hm.max() - hm.min())
        else:
            h_norm = np.zeros_like(hm)

    h_uint8 = (h_norm * 255).astype(np.uint8)
    h_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    blended = cv2.addWeighted(image, 1 - alpha, h_color, alpha, 0)

    if fg_mask is not None:
        # Paste blended only inside foreground; keep original outside
        result = image.copy()
        result[fg] = blended[fg]
        return result

    return blended


def comparison_figure(
    image: np.ndarray, heatmap: np.ndarray,
    score: float, threshold: float = 0.0,
    gt_mask: np.ndarray = None,
    title: str = "",
    save_path: str = None,
):
    """Multi-panel: original | heatmap overlay | detection | ground truth."""
    n = 4 if gt_mask is not None else 3
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    
    rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    axes[0].imshow(rgb)
    axes[0].set_title("Input")
    axes[0].axis("off")
    
    overlay = heatmap_overlay(image, heatmap)
    axes[1].imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
    axes[1].set_title(f"Score: {score:.3f}")
    axes[1].axis("off")
    
    if heatmap.max() > heatmap.min():
        h_n = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        h_n = np.zeros_like(heatmap)
    axes[2].imshow((h_n > 0.5).astype(np.uint8) * 255, cmap='gray')
    is_anom = score > threshold if threshold > 0 else None
    label = "ANOMALY" if is_anom else "NORMAL" if is_anom is False else "?"
    color = "red" if is_anom else "green" if is_anom is False else "gray"
    axes[2].set_title(label, color=color, fontweight='bold')
    axes[2].axis("off")
    
    if gt_mask is not None:
        axes[3].imshow(gt_mask, cmap='gray')
        axes[3].set_title("Ground Truth")
        axes[3].axis("off")
    
    if title:
        fig.suptitle(title, fontsize=12)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def plot_score_distribution(
    normal_scores: np.ndarray,
    anomaly_scores: np.ndarray = None,
    threshold: float = None,
    save_path: str = None,
    title: str = "Anomaly Score Distribution",
    normal_label: str = None,
    anomaly_label: str = None,
):
    """Plot normal vs anomalous score distributions.

    Automatically selects the right visual encoding:
      - n < 25  → strip chart (individual points + mean±std span) — avoids the
                  misleading wide σ-bands and near-empty histograms that result
                  from small, tightly-clustered sample sets.
      - n ≥ 25  → histogram with density normalisation.

    The model threshold is always shown as a vertical dashed line.
    The x-axis is clamped to the data range so the chart stays readable even
    when the training-time distribution and inference-time scores differ.
    """
    n_norm = len(normal_scores)
    n_anom = len(anomaly_scores) if anomaly_scores is not None else 0

    norm_lbl = normal_label or f"Baseline / Normal  (n={n_norm})"
    anom_lbl = anomaly_label or f"Degraded / Anomalous  (n={n_anom})"

    fig, ax = plt.subplots(figsize=(11, 5))

    STRIP_THRESHOLD = 25  # use strip chart below this count

    groups = [(normal_scores, "#2ca02c", norm_lbl)]
    if anomaly_scores is not None and len(anomaly_scores) > 0:
        groups.append((np.asarray(anomaly_scores), "#d62728", anom_lbl))

    all_vals = np.concatenate([g[0] for g in groups])
    data_lo, data_hi = float(all_vals.min()), float(all_vals.max())
    pad = max((data_hi - data_lo) * 0.1, 0.3)
    x_lo, x_hi = data_lo - pad, data_hi + pad
    if threshold is not None:
        x_lo = min(x_lo, threshold - pad)
        x_hi = max(x_hi, threshold + pad)

    rng = np.random.default_rng(42)
    use_strip = any(len(sc) < STRIP_THRESHOLD for sc, *_ in groups)

    for scores, color, label in groups:
        scores = np.asarray(scores)
        if len(scores) == 0:
            continue

        if len(scores) < STRIP_THRESHOLD:
            mu, std = float(scores.mean()), float(scores.std())
            jitter = rng.uniform(-0.25, 0.25, size=len(scores))
            ax.scatter(scores, jitter, color=color, s=60, alpha=0.85, zorder=5,
                       label=f"{label}\nmean={mu:.3f},  std={std:.3f}")
            # Mean line
            ax.axvline(mu, color=color, lw=2.5, alpha=0.9, zorder=4)
            # ±1 std span
            ax.axvspan(mu - std, mu + std, alpha=0.12, color=color)
        else:
            n_bins = max(10, min(50, len(scores) // 4))
            ax.hist(scores, bins=n_bins, alpha=0.55, color=color,
                    density=True, label=label)

    if threshold is not None:
        ax.axvline(threshold, color="black", ls="--", lw=2,
                   label=f"Model threshold  ({threshold:.3f})", zorder=6)

    ax.set_xlim(x_lo, x_hi)
    if use_strip:
        ax.set_ylim(-0.5, 0.5)
        ax.set_yticks([])
        ax.set_ylabel("(jittered — no scale)", fontsize=10, color="grey")
    else:
        ax.set_ylabel("Density", fontsize=11)

    ax.set_xlabel("Anomaly Score", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9)
    ax.grid(True, axis="x", alpha=0.3)
    fig.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_roc(labels, scores, save_path=None, title="ROC Curve"):
    """Plot ROC curve."""
    from sklearn.metrics import roc_curve, roc_auc_score
    fpr, tpr, _ = roc_curve(labels, scores)
    auc = roc_auc_score(labels, scores)
    
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(fpr, tpr, 'b-', lw=2, label=f'AUC={auc:.4f}')
    ax.plot([0, 1], [0, 1], 'k--')
    ax.set_xlabel("FPR")
    ax.set_ylabel("TPR")
    ax.set_title(f"{title} (AUROC={auc:.4f})")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
