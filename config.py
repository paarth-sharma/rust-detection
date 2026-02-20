"""
Central configuration for the fastener anomaly detection pipeline.

Hardware target: Intel i7-13th gen + NVIDIA RTX 4060 Mobile (8GB VRAM)
Backbone: WideResNet-50-2 (frozen, ImageNet-pretrained)
Method: PatchCore with greedy coreset subsampling
"""

from pathlib import Path
from dataclasses import dataclass, field
from typing import List, Tuple, Optional


@dataclass
class PathConfig:
    root: Path = Path(__file__).parent
    data_dir: Path = None
    output_dir: Path = None
    model_dir: Path = None

    def __post_init__(self):
        self.data_dir = self.root / "data"
        self.output_dir = self.root / "outputs"
        self.model_dir = self.root / "saved_models"

    def ensure_dirs(self):
        for d in [self.data_dir, self.output_dir, self.model_dir]:
            d.mkdir(parents=True, exist_ok=True)


@dataclass
class PreprocessConfig:
    image_size: Tuple[int, int] = (256, 256)
    clahe_clip_limit: float = 2.0
    clahe_tile_grid: Tuple[int, int] = (8, 8)
    gaussian_blur_ksize: int = 5
    morph_kernel_size: int = 5
    morph_iterations: int = 2
    padding_fraction: float = 0.1
    min_component_area_ratio: float = 0.05
    preserve_color: bool = True
    align_to_canonical: bool = True


@dataclass
class BackboneConfig:
    name: str = "wide_resnet50_2"
    layers: List[str] = field(default_factory=lambda: ["layer2", "layer3"])
    pretrained: bool = True
    freeze: bool = True


@dataclass
class PatchCoreConfig:
    coreset_sampling_ratio: float = 0.1
    num_neighbors: int = 9
    target_dim: Optional[int] = 256
    faiss_index_type: str = "flat"
    neighborhood_size: int = 3


CORRODED_COLUMNS = ["Corroded-Bolt", "Corroded-Nut", "Corroded-Nut-and-Bolt"]


@dataclass
class TrainingConfig:
    batch_size: int = 32
    num_workers: int = 4
    pin_memory: bool = True
    seed: int = 42
    threshold_percentile: float = 99.0

    @property
    def device(self):
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    backbone: BackboneConfig = field(default_factory=BackboneConfig)
    patchcore: PatchCoreConfig = field(default_factory=PatchCoreConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)

    def __post_init__(self):
        self.paths.ensure_dirs()


cfg = Config()
