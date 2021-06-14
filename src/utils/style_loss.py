import torch
import torch.nn as nn
import torch.nn.functional as F


class StyleLoss(nn.Module):
    def __init__(self):
        super(StyleLoss, self).__init__()

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        G_inputs = StyleLoss.gram_matrix(inputs)
        G_targets = StyleLoss.gram_matrix(targets)

        return F.mse_loss(G_inputs, G_targets)

    @staticmethod
    def gram_matrix(x: torch.Tensor) -> torch.tensor:
        # extract shape of the input tensor
        b, c, h, w = x.size()
        features = x.view(b * c, h * w)
        # compute the gram product
        G = torch.mm(features, features.t())
        # normalize the values of the gram matrix
        return G.div(b * c * h * w)
