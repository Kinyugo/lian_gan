import torch
import torch.nn as nn
from einops.layers.torch import Reduce


class VectorHead(nn.Module):
    """
    Classification head of a the model. 

    How it works
    ------------
    The forward pass takes in a tensor of shape B x L x M where
    M is equal to (`dim`). The input tensor is first passed through
    layer norm. Afterwards a mean over the patch dimension (L) is taken
    to obtain a tensor of shape B x M. This B x M tensor is then projected
    using a linear layer into a B x M tensor where M is equal to (`dim`).
    Finally a Tanh activation is applied.

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
        super(VectorHead, self).__init__()
        self.model = nn.Sequential(
            nn.LayerNorm(dim),
            Reduce("b l m -> b m", reduction="mean"),
            nn.Linear(dim, dim),
            nn.Tanh(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x L x M -> B x M
        return self.model(x)
