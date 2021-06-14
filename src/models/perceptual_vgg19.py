from typing import List

import torch
import torch.nn as nn
from torchvision import models


class PerceptualVgg19(nn.Module):
    """
    Wraps a vgg-19 model that is used as a feature extraction network.

    How it works
    ------------
    The input is foward propagated through the network then the activations
    of `feature_layers` are taken.
    The following layers are used for feature extraction:
    Content features:
        11
    Style Features:
        1, 3, 6, 11, 15
    """

    def __init__(self, feature_layers: List[int]):
        super(PerceptualVgg19, self).__init__()
        self.feature_layers = feature_layers
        self.model = models.vgg19(pretrained=True)
        # set the model to be in evaluation mode
        # to avoid gradient accumulation
        self.model.eval()
        for param in self.parameters():
            param.requires_grad = False

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        features = []
        feature = x
        for layer in range(max(self.feature_layers) + 1):
            feature = self.model.features[layer](feature)
            if layer in self.feature_layers:
                feature_clone = feature.clone()
                features.append(feature_clone)

        return features

