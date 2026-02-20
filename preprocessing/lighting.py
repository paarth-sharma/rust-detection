"""
Lighting normalization for metallic fastener surfaces.

CLAHE on L channel in L*a*b* space normalizes brightness while
preserving color (critical for rust detection - orange-brown hues).
"""

import cv2
import numpy as np
from typing import Tuple


def apply_clahe(
    image: np.ndarray,
    clip_limit: float = 2.0,
    tile_grid: Tuple[int, int] = (8, 8),
) -> np.ndarray:
    """CLAHE on L channel of L*a*b* — normalizes brightness, preserves color."""
    lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l_ch, a_ch, b_ch = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid)
    l_norm = clahe.apply(l_ch)
    lab_norm = cv2.merge([l_norm, a_ch, b_ch])
    return cv2.cvtColor(lab_norm, cv2.COLOR_LAB2BGR)


def normalize_white_balance(image: np.ndarray) -> np.ndarray:
    """Gray-world white balance correction."""
    result = image.copy().astype(np.float32)
    avg_b = np.mean(result[:, :, 0])
    avg_g = np.mean(result[:, :, 1])
    avg_r = np.mean(result[:, :, 2])
    avg_gray = (avg_b + avg_g + avg_r) / 3.0
    if avg_b > 0:
        result[:, :, 0] *= avg_gray / avg_b
    if avg_g > 0:
        result[:, :, 1] *= avg_gray / avg_g
    if avg_r > 0:
        result[:, :, 2] *= avg_gray / avg_r
    return np.clip(result, 0, 255).astype(np.uint8)


def reduce_specular_highlights(
    image: np.ndarray,
    threshold: int = 240,
    kernel_size: int = 15,
) -> np.ndarray:
    """Inpaint overexposed specular highlights on metal surfaces."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, highlight_mask = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
    highlight_ratio = np.sum(highlight_mask > 0) / highlight_mask.size
    if highlight_ratio > 0.05 or np.sum(highlight_mask) == 0:
        return image
    return cv2.inpaint(image, highlight_mask, kernel_size, cv2.INPAINT_TELEA)
