"""
Component segmentation and background removal.

Uses Otsu thresholding + morphological operations to isolate the
fastener from background. Finds largest contour = the fastener.
"""

import cv2
import numpy as np
from typing import Tuple, Optional


def create_component_mask(
    image: np.ndarray,
    blur_ksize: int = 5,
    morph_kernel_size: int = 5,
    morph_iterations: int = 2,
    min_area_ratio: float = 0.05,
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Binary mask isolating the fastener via Otsu + morphology.
    Returns (mask, largest_contour) or (full_mask, None) if no valid contour.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (blur_ksize, blur_ksize), 0)

    # Otsu's thresholding
    _, binary = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )

    # Check if Otsu gave a reasonable result
    fg_ratio = np.sum(binary > 0) / binary.size
    if fg_ratio < min_area_ratio or fg_ratio > 0.95:
        binary = cv2.adaptiveThreshold(
            blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY_INV, 51, 10
        )

    # Morphological close (fill holes) then open (remove noise)
    kernel = cv2.getStructuringElement(
        cv2.MORPH_ELLIPSE, (morph_kernel_size, morph_kernel_size)
    )
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=morph_iterations)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return np.ones_like(gray) * 255, None

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) / (gray.shape[0] * gray.shape[1]) < min_area_ratio:
        return np.ones_like(gray) * 255, None

    clean_mask = np.zeros_like(gray)
    cv2.drawContours(clean_mask, [largest], -1, 255, -1)
    clean_mask = cv2.morphologyEx(clean_mask, cv2.MORPH_CLOSE, kernel, iterations=3)
    return clean_mask, largest


def extract_component_roi(
    image: np.ndarray,
    mask: np.ndarray,
    contour: Optional[np.ndarray] = None,
    padding_fraction: float = 0.1,
    target_size: Optional[Tuple[int, int]] = None,
) -> Tuple[np.ndarray, np.ndarray, dict]:
    """Crop to component bounding box with padding."""
    h, w = image.shape[:2]

    if contour is not None:
        x, y, bw, bh = cv2.boundingRect(contour)
    else:
        coords = cv2.findNonZero(mask)
        if coords is None:
            info = {"x": 0, "y": 0, "w": w, "h": h, "scale": 1.0}
            if target_size:
                image = cv2.resize(image, target_size)
                mask = cv2.resize(mask, target_size)
            return image, mask, info
        x, y, bw, bh = cv2.boundingRect(coords)

    pad_x = int(bw * padding_fraction)
    pad_y = int(bh * padding_fraction)
    x1, y1 = max(0, x - pad_x), max(0, y - pad_y)
    x2, y2 = min(w, x + bw + pad_x), min(h, y + bh + pad_y)

    cropped_image = image[y1:y2, x1:x2]
    cropped_mask = mask[y1:y2, x1:x2]
    info = {"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "original_size": (w, h), "scale": 1.0}

    if target_size and cropped_image.size > 0:
        info["scale"] = (target_size[0] / cropped_image.shape[1], target_size[1] / cropped_image.shape[0])
        cropped_image = cv2.resize(cropped_image, target_size, interpolation=cv2.INTER_AREA)
        cropped_mask = cv2.resize(cropped_mask, target_size, interpolation=cv2.INTER_NEAREST)

    return cropped_image, cropped_mask, info


def apply_background_mask(
    image: np.ndarray,
    mask: np.ndarray,
    bg_value: int = 0,
) -> np.ndarray:
    """Zero out background, keep only component pixels."""
    result = np.full_like(image, bg_value)
    result[mask > 0] = image[mask > 0]
    return result
