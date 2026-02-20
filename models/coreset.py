"""
Greedy coreset subsampling (minimax facility location).

Selects representative subset: "pick the point farthest from all selected points."
At 10% ratio, PatchCore loses <0.02 AUROC — coreset captures essential structure.
"""

import numpy as np
import torch
from typing import Tuple
from tqdm import tqdm


def greedy_coreset_subsampling(
    features: np.ndarray,
    sampling_ratio: float = 0.1,
    device: str = "cuda",
) -> Tuple[np.ndarray, np.ndarray]:
    N, D = features.shape
    target_count = max(1, int(N * sampling_ratio))
    if target_count >= N:
        return features, np.arange(N)

    print(f"  Coreset: {N} → {target_count} patches ({sampling_ratio*100:.1f}%)")

    feat_tensor = torch.from_numpy(features).float()
    if device == "cuda" and torch.cuda.is_available():
        feat_tensor = feat_tensor.cuda()

    selected = [np.random.randint(0, N)]
    first = feat_tensor[selected[0]].unsqueeze(0)
    min_distances = torch.cdist(feat_tensor, first).squeeze()

    for _ in tqdm(range(target_count - 1), desc="  Coreset selection", leave=False):
        new_idx = torch.argmax(min_distances).item()
        selected.append(new_idx)
        new_point = feat_tensor[new_idx].unsqueeze(0)
        new_dists = torch.cdist(feat_tensor, new_point).squeeze()
        min_distances = torch.minimum(min_distances, new_dists)

    selected = np.array(selected)
    print(f"  Coreset complete: {len(selected)} patches selected")
    return features[selected], selected
