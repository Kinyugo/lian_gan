import torch
import torch.nn as nn
from einops.layers.torch import Reduce


class ClassificationHead(nn.Module):
    """
    Classification head of a the model. 

    How it works
    ------------
    The forward pass takes in a tensor of shape B x L x M where
    M is equal to (`dim`). The input tensor is first passed through
    layer norm. Afterwards a mean over the patch dimension (L) is taken
    to obtain a tensor of shape B x M. This B x M tensor is then projected
    using a linear layer into a B x 1 tensor where 1 is the number of classes
    to be predicted (`num_classes`). Finally a softmax over the 1 dimension
    is taken and passed as the output.

    References
    ----------
    1. 1. https://github.com/lucidrains/vit-pytorch
    """

    def __init__(self, dim: int) -> None:
        """
        Arguments
        ---------
        dim : int 
            last dimension of the tensor, will be the embedding dimension.
        """
        super(ClassificationHead, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce("b l m -> b m", reduction="mean"),
            nn.Linear(dim, 1),
            nn.Sigmoid()
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x L x M -> B x T
        return self.model(x)
