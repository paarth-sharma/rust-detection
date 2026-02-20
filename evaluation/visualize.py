"""Visualization for anomaly detection results."""

import cv2
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from typing import Optional


def heatmap_overlay(image: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """Overlay anomaly heatmap on image. Both BGR."""
    if heatmap.max() > heatmap.min():
        h_norm = (heatmap - heatmap.min()) / (heatmap.max() - heatmap.min())
    else:
        h_norm = np.zeros_like(heatmap)
    
    h_uint8 = (h_norm * 255).astype(np.uint8)
    h_color = cv2.applyColorMap(h_uint8, cv2.COLORMAP_JET)
    
    if h_color.shape[:2] != image.shape[:2]:
        h_color = cv2.resize(h_color, (image.shape[1], image.shape[0]))
    
    return cv2.addWeighted(image, 1 - alpha, h_color, alpha, 0)


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
):
    """Plot normal vs anomalous score distributions with confidence intervals."""
    fig, ax = plt.subplots(figsize=(10, 5))
    
    ax.hist(normal_scores, bins=50, alpha=0.6, color='green',
            label=f'Normal (n={len(normal_scores)})', density=True)
    
    if anomaly_scores is not None and len(anomaly_scores) > 0:
        ax.hist(anomaly_scores, bins=50, alpha=0.6, color='red',
                label=f'Anomalous (n={len(anomaly_scores)})', density=True)
    
    if threshold is not None:
        ax.axvline(threshold, color='black', ls='--', lw=2,
                   label=f'Threshold ({threshold:.3f})')
    
    if len(normal_scores) > 1:
        mu, sigma = np.mean(normal_scores), np.std(normal_scores)
        for n_s, a, lab in [(1, 0.3, '1σ 68%'), (2, 0.15, '2σ 95%'), (3, 0.05, '3σ 99.7%')]:
            ax.axvspan(mu - n_s*sigma, mu + n_s*sigma, alpha=a, color='blue', label=lab)
    
    ax.set_xlabel("Anomaly Score")
    ax.set_ylabel("Density")
    ax.set_title(title)
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
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
