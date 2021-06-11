import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


class PatchEmbedding(nn.Module):
    """
    Linearly projects individual image patches.

    How it works
    ------------
    The forward pass takes in an input tensor of shape B x C x H x W, where C
    is the number of channels in the image and H and W are the spatial dimensions
    of the image. The input tensor is split into patches of `patch_size` spatial
    dimensions and flattened along the channel (C) dimension to a tensor of patches
    of shape B x L x M, where L is the number of patches obtained by taking;
    `H // patch_size * W // patch_size` and M is the patch dimension obtained by taking;
    `patch_size ** 2 * C`. 

    The flattened patches are then projected using a linear layer to obtain a tensor
    of shape B x L x D, where D is the embedding dimension (`dim`)

    References
    ----------
    1. https://github.com/Kinyugo/blind_net
    """

    def __init__(self, patch_size: int, channels: int, dim: int):
        """
        Arguments
        ---------
        patch_size : int
            size of the spatial dimensions of a single patch 
        channels: int 
            number of channels of the image
        dim : int 
            size of the embedding dimension
        """
        super(PatchEmbedding, self).__init__()
        self.patch_size = patch_size
        self.channels = channels

        self.to_patch_embedding = nn.Sequential(
            Rearrange(
                "b c (h p1) (w p2) -> b (h w) (p1 p2 c)", p1=patch_size, p2=patch_size
            ),
            nn.Linear(self.patch_dim, dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x C x H x W -> B x L x M
        return self.to_patch_embedding(x)

    @property
    def patch_dim(self) -> int:
        return self.patch_size ** 2 * self.channels
