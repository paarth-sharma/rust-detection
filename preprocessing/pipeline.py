"""
Complete preprocessing pipeline.

Takes any input image → normalized 256x256 with consistent lighting,
background removed, component centered/aligned, scale normalized.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional, Dict, Any
from dataclasses import dataclass

from .lighting import apply_clahe, normalize_white_balance, reduce_specular_highlights
from .segmentation import create_component_mask, extract_component_roi, apply_background_mask
from .registration import align_to_canonical, scale_normalize


@dataclass
class PreprocessResult:
    image: np.ndarray
    mask: np.ndarray
    original: np.ndarray
    metadata: Dict[str, Any]
    success: bool
    error: Optional[str] = None


class FastenerPreprocessor:
    def __init__(
        self,
        target_size: Tuple[int, int] = (256, 256),
        clahe_clip_limit: float = 2.0,
        clahe_tile_grid: Tuple[int, int] = (8, 8),
        blur_ksize: int = 5,
        morph_kernel_size: int = 5,
        morph_iterations: int = 2,
        min_component_area: float = 0.05,
        padding_fraction: float = 0.1,
        preserve_color: bool = True,
        align: bool = True,
        fill_ratio: float = 0.8,
    ):
        self.target_size = target_size
        self.clahe_clip_limit = clahe_clip_limit
        self.clahe_tile_grid = clahe_tile_grid
        self.blur_ksize = blur_ksize
        self.morph_kernel_size = morph_kernel_size
        self.morph_iterations = morph_iterations
        self.min_component_area = min_component_area
        self.padding_fraction = padding_fraction
        self.preserve_color = preserve_color
        self.align = align
        self.fill_ratio = fill_ratio

    def __call__(self, image: np.ndarray) -> PreprocessResult:
        return self.process(image)

    def process(self, image: np.ndarray) -> PreprocessResult:
        """Full pipeline: lighting → segmentation → registration → normalization."""
        original = image.copy()
        metadata = {}

        try:
            if image is None or image.size == 0:
                return PreprocessResult(image=image, mask=np.zeros(self.target_size[::-1], dtype=np.uint8),
                                       original=original, metadata={}, success=False, error="Empty image")

            # Ensure BGR 3-channel
            if len(image.shape) == 2:
                image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
            elif image.shape[2] == 4:
                image = cv2.cvtColor(image, cv2.COLOR_BGRA2BGR)

            # 1. White balance
            image = normalize_white_balance(image)
            # 2. Specular highlight reduction
            image = reduce_specular_highlights(image)
            # 3. CLAHE
            image = apply_clahe(image, self.clahe_clip_limit, self.clahe_tile_grid)
            # 4. Segmentation
            mask, contour = create_component_mask(image, self.blur_ksize,
                                                   self.morph_kernel_size, self.morph_iterations,
                                                   self.min_component_area)
            metadata["has_contour"] = contour is not None
            # 5. ROI extraction
            image, mask, crop_info = extract_component_roi(image, mask, contour, self.padding_fraction)
            metadata["crop_info"] = crop_info
            # 6. Alignment
            if self.align and contour is not None:
                mask_contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if mask_contours:
                    crop_contour = max(mask_contours, key=cv2.contourArea)
                    image, mask, transform = align_to_canonical(image, mask, crop_contour, symmetry_aware=True)
                    metadata["alignment_transform"] = transform
            # 7. Scale normalization
            image, mask = scale_normalize(image, mask, self.target_size, self.fill_ratio)
            # 8. Background masking
            image = apply_background_mask(image, mask, bg_value=0)

            return PreprocessResult(image=image, mask=mask, original=original, metadata=metadata, success=True)

        except Exception as e:
            fallback = cv2.resize(original, self.target_size)
            return PreprocessResult(image=fallback,
                                   mask=np.ones(self.target_size[::-1], dtype=np.uint8) * 255,
                                   original=original, metadata=metadata, success=False, error=str(e))

    def process_file(self, path: Path) -> PreprocessResult:
        image = cv2.imread(str(path))
        if image is None:
            return PreprocessResult(
                image=np.zeros((*self.target_size[::-1], 3), dtype=np.uint8),
                mask=np.zeros(self.target_size[::-1], dtype=np.uint8),
                original=np.zeros((1, 1, 3), dtype=np.uint8),
                metadata={"path": str(path)}, success=False,
                error=f"Could not read image: {path}")
        result = self.process(image)
        result.metadata["path"] = str(path)
        return result
