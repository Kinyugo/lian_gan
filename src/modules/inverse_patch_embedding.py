import torch
import torch.nn as nn

from einops.layers.torch import Rearrange


class InversePatchEmbedding(nn.Module):
    """
    Transforms embedded patches into image patches by performing an inverse 
    operation of the `PatchEmbedding` layer. 

    How it works
    ------------
    The forward pass takes in an input tensor of shape B x L x M where L is the 
    number of patches and M is the embedding dimension (`dim`). The input tensor is first
    linearly projected to a tensor of flattened patches of  shape B x L x N where 
    N is the patch dimension obtained by taking the `patch_size ** 2 * C` where 
    C is the number of channels in the original image.

    The flattened patches are then rearranged to obtain image tensor of shape 
    B x C x H x W where C is equal to `channels` and H & W are equal to `image_size`. 
    """

    def __init__(
        self, image_size: int, patch_size: int, channels: int, dim: int
    ) -> None:
        """
        Arguments
        ---------
        image_size : int 
            spatial dimensions of the image
        patch_size : int
            size of the spatial dimensions of a single patch 
        channels: int 
            number of channels of the image
        dim : int 
            size of the embedding dimension
        """
        super(InversePatchEmbedding, self).__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.channels = channels

        self.to_image = nn.Sequential(
            nn.Linear(dim, self.patch_dim),
            Rearrange(
                "b (h w) (p1 p2 c) -> b c (h p1) (w p2)",
                h=self.spatial_dim,
                w=self.spatial_dim,
                p1=patch_size,
                p2=patch_size,
            ),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x L x M -> B x C x H x W
        return self.to_image(x)

    @property
    def patch_dim(self) -> int:
        return self.patch_size ** 2 * self.channels

    @property
    def spatial_dim(self) -> int:
        return self.image_size // self.patch_size
