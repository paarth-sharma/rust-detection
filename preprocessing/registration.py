"""
Image registration for fasteners.

Aligns to canonical orientation via image moments (centroid + principal axis).
Handles symmetry (hex nuts = 60° periodicity).
"""

import cv2
import numpy as np
import math
from typing import Tuple, Optional


def compute_orientation(
    mask: np.ndarray,
    contour: Optional[np.ndarray] = None,
) -> Tuple[float, float, float, float]:
    """Compute centroid (cx,cy), orientation angle, and aspect ratio from moments."""
    moments = cv2.moments(contour) if contour is not None else cv2.moments(mask)
    if moments["m00"] == 0:
        h, w = mask.shape[:2]
        return w / 2, h / 2, 0.0, 1.0

    cx = moments["m10"] / moments["m00"]
    cy = moments["m01"] / moments["m00"]

    mu20, mu02, mu11 = moments["mu20"], moments["mu02"], moments["mu11"]
    if abs(mu20 - mu02) < 1e-10:
        angle_deg = 0.0
    else:
        angle_deg = math.degrees(0.5 * math.atan2(2 * mu11, mu20 - mu02))

    discriminant = 4 * mu11**2 + (mu20 - mu02)**2
    lambda1 = 0.5 * (mu20 + mu02) + 0.5 * math.sqrt(discriminant)
    lambda2 = 0.5 * (mu20 + mu02) - 0.5 * math.sqrt(discriminant)
    aspect_ratio = math.sqrt(lambda1 / lambda2) if lambda2 > 0 else 1.0

    return cx, cy, angle_deg, aspect_ratio


def detect_symmetry_order(contour: np.ndarray, n_samples: int = 360) -> int:
    """Estimate rotational symmetry order (6 for hex nut, 4 for square, etc.)."""
    if contour is None or len(contour) < 10:
        return 1

    M = cv2.moments(contour)
    if M["m00"] == 0:
        return 1
    cx = M["m10"] / M["m00"]
    cy = M["m01"] / M["m00"]

    pts = contour.reshape(-1, 2).astype(float)
    angles = np.arctan2(pts[:, 1] - cy, pts[:, 0] - cx)
    distances = np.sqrt((pts[:, 0] - cx)**2 + (pts[:, 1] - cy)**2)

    bins = np.linspace(-np.pi, np.pi, n_samples + 1)
    hist, _ = np.histogram(angles, bins=bins, weights=distances)
    if np.max(hist) > 0:
        hist = hist / np.max(hist)

    autocorr = np.correlate(hist, hist, mode='full')
    autocorr = autocorr[len(autocorr)//2:]
    if autocorr[0] > 0:
        autocorr = autocorr / autocorr[0]

    min_dist = n_samples // 12
    peaks = []
    for i in range(min_dist, len(autocorr) - min_dist):
        if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1] and autocorr[i] > 0.5:
            peaks.append(i)

    if not peaks:
        return 1
    sym_order = round(n_samples / peaks[0])
    return sym_order if sym_order in [3, 4, 5, 6, 8, 12] else 1


def align_to_canonical(
    image: np.ndarray,
    mask: np.ndarray,
    contour: Optional[np.ndarray] = None,
    target_size: Optional[Tuple[int, int]] = None,
    symmetry_aware: bool = True,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Align fastener to canonical orientation (centered, rotated to principal axis)."""
    h, w = image.shape[:2]
    cx, cy, angle, aspect_ratio = compute_orientation(mask, contour)

    target_angle = -angle  # Rotate to vertical/horizontal

    # Symmetry: reduce to fundamental domain
    if symmetry_aware and contour is not None and aspect_ratio < 1.5:
        sym_order = detect_symmetry_order(contour)
        if sym_order > 1:
            period = 360.0 / sym_order
            target_angle = target_angle % period

    out_w = target_size[0] if target_size else w
    out_h = target_size[1] if target_size else h

    M_rot = cv2.getRotationMatrix2D((cx, cy), target_angle, 1.0)
    M_rot[0, 2] += out_w / 2 - cx
    M_rot[1, 2] += out_h / 2 - cy

    aligned_image = cv2.warpAffine(image, M_rot, (out_w, out_h),
                                    flags=cv2.INTER_LINEAR,
                                    borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))
    aligned_mask = cv2.warpAffine(mask, M_rot, (out_w, out_h),
                                   flags=cv2.INTER_NEAREST,
                                   borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return aligned_image, aligned_mask, M_rot


def scale_normalize(
    image: np.ndarray,
    mask: np.ndarray,
    target_size: Tuple[int, int] = (256, 256),
    fill_ratio: float = 0.8,
) -> Tuple[np.ndarray, np.ndarray]:
    """Scale component to fill consistent fraction of target canvas."""
    coords = cv2.findNonZero(mask)
    if coords is None:
        return cv2.resize(image, target_size), cv2.resize(mask, target_size)

    _, _, bw, bh = cv2.boundingRect(coords)
    target_w, target_h = target_size
    scale = min(target_w * fill_ratio / max(bw, 1), target_h * fill_ratio / max(bh, 1))

    new_w, new_h = int(image.shape[1] * scale), int(image.shape[0] * scale)
    if new_w <= 0 or new_h <= 0:
        return cv2.resize(image, target_size), cv2.resize(mask, target_size)

    scaled_img = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    scaled_mask = cv2.resize(mask, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    canvas_img = np.zeros((target_h, target_w, 3), dtype=np.uint8)
    canvas_mask = np.zeros((target_h, target_w), dtype=np.uint8)

    off_x = max(0, (target_w - new_w) // 2)
    off_y = max(0, (target_h - new_h) // 2)
    src_x = max(0, (new_w - target_w) // 2)
    src_y = max(0, (new_h - target_h) // 2)
    copy_w = min(new_w - src_x, target_w - off_x)
    copy_h = min(new_h - src_y, target_h - off_y)

    canvas_img[off_y:off_y+copy_h, off_x:off_x+copy_w] = scaled_img[src_y:src_y+copy_h, src_x:src_x+copy_w]
    canvas_mask[off_y:off_y+copy_h, off_x:off_x+copy_w] = scaled_mask[src_y:src_y+copy_h, src_x:src_x+copy_w]
    return canvas_img, canvas_mask
