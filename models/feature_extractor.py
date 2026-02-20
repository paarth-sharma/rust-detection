"""
Feature extractor: frozen pretrained WideResNet-50-2.

Extracts from layers 2+3 via forward hooks, concatenates after spatial alignment.
For 256x256 input: layer2=512x32x32, layer3=1024x16x16 → concat=1536x32x32.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import wide_resnet50_2, Wide_ResNet50_2_Weights
from torchvision import transforms
from typing import List, Dict, Optional


class FeatureExtractor(nn.Module):
    def __init__(
        self,
        backbone_name: str = "wide_resnet50_2",
        layers: List[str] = None,
        pretrained: bool = True,
        target_dim: Optional[int] = None,
    ):
        super().__init__()
        self.layers = layers or ["layer2", "layer3"]
        self.target_dim = target_dim
        self._features: Dict[str, torch.Tensor] = {}

        weights = Wide_ResNet50_2_Weights.IMAGENET1K_V1 if pretrained else None
        self.backbone = wide_resnet50_2(weights=weights)

        for param in self.backbone.parameters():
            param.requires_grad = False
        self.backbone.eval()

        self._hooks = []
        for name in self.layers:
            layer = dict(self.backbone.named_children())[name]
            hook = layer.register_forward_hook(self._make_hook(name))
            self._hooks.append(hook)

        self._feature_dims = self._get_dims()
        self.total_dim = sum(self._feature_dims.values())

        self.projection = None
        if target_dim and target_dim < self.total_dim:
            self.projection = nn.Linear(self.total_dim, target_dim, bias=False)
            nn.init.orthogonal_(self.projection.weight)
            self.projection.weight.requires_grad = False

    def _make_hook(self, name):
        def hook(module, input, output):
            self._features[name] = output
        return hook

    def _get_dims(self):
        dummy = torch.randn(1, 3, 256, 256)
        with torch.no_grad():
            self.backbone(dummy)
        dims = {name: self._features[name].shape[1] for name in self.layers}
        self._features.clear()
        return dims

    @torch.no_grad()
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self._features.clear()
        self.backbone(x)

        target_size = None
        maps = []
        for name in self.layers:
            feat = self._features[name]
            if target_size is None:
                target_size = feat.shape[2:]
            else:
                target_size = (max(target_size[0], feat.shape[2]),
                             max(target_size[1], feat.shape[3]))
            maps.append(feat)

        aligned = []
        for feat in maps:
            if feat.shape[2:] != target_size:
                feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
            aligned.append(feat)
        features = torch.cat(aligned, dim=1)
        features = F.avg_pool2d(features, kernel_size=3, stride=1, padding=1)

        if self.projection is not None:
            B, C, h, w = features.shape
            features = features.permute(0, 2, 3, 1).reshape(-1, C)
            features = self.projection(features)
            features = features.reshape(B, h, w, -1).permute(0, 3, 1, 2)

        self._features.clear()
        return features

    def extract_patch_features(self, x: torch.Tensor) -> torch.Tensor:
        features = self.forward(x)
        B, C, h, w = features.shape
        return features.permute(0, 2, 3, 1).reshape(B, h * w, C)

    @property
    def feature_dim(self):
        return self.target_dim or self.total_dim

    @property
    def spatial_size(self):
        return 32

    def remove_hooks(self):
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()


def get_imagenet_transforms():
    return transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
