from typing import List

import torch
import torchvision.transforms as T
from torchvision.utils import make_grid


def make_image_grid(
    input_tensor: torch.Tensor,
    normalized: bool,
    mean: List[float] = None,
    std: List[float] = None,
) -> torch.Tensor:
    # unnormalize the input
    if normalized:
        mean = torch.tensor(mean, dtype=torch.float)
        std = torch.tensor(std, dtype=torch.float)
        unnormalize = T.Normalize(mean=(-mean / std).tolist(), std=(1.0 / std).tolist())
        input_tensor = unnormalize(input_tensor)
    image_grid = make_grid(input_tensor)

    return image_grid
