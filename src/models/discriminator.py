import torch
import torch.nn as nn

from src.modules import (
    PatchEmbedding,
    PreNormResidual,
    MatMulTransform,
    FeedForward,
    ClassificationHead,
)


def DiscriminatorBlock(
    image_size: int,
    patch_size: int,
    dim: int,
    ff_dim: int,
    norm: nn.Module,
    dropout: float,
) -> nn.Module:
    l_dim = (image_size // patch_size) ** 2
    return nn.Sequential(
        PreNormResidual(MatMulTransform(l_dim, dropout), dim, norm),
        PreNormResidual(FeedForward(dim, ff_dim, dropout), dim, norm),
    )


def make_n_blocks(
    num_blocks: int,
    image_size: int,
    patch_size: int,
    dim: int,
    ff_dim: int,
    norm: nn.Module,
    dropout: float,
) -> nn.Module:
    blocks = nn.ModuleList()
    for _ in range(num_blocks):
        blocks.append(
            DiscriminatorBlock(image_size, patch_size, dim, ff_dim, norm, dropout)
        )

    return nn.Sequential(*blocks)


class Discriminator(nn.Module):
    """
    Defines a discriminator for the image styling GAN. 

    How it works
    ------------
    The forward pass takes in an input tensor of shape B x C x H x W, 
    where C is the number of image channels and H & W are the spatial
    dimensions of the image. 
    The input tensor is then embedded through a `PatchEmbedding` layer
    to obtain a B x L x M tensor, where L is the number of patches and 
    M is the embedding dimension. 
    The B x L x M tensor is then passed through `num_blocks` `DiscriminatorBlock`(s)
    to extract features of the same shape. 
    The B x L x M features are then passed through a `ClassificationHead` to obtain
    the final predictions of shape B x T, where T is the number of target classes
    (`num_classes`).
    """

    def __init__(
        self,
        num_blocks: int,
        num_classes: int,
        image_size: int,
        channels: int,
        patch_size: int,
        dim: int,
        ff_dim: int,
        norm: nn.Module = nn.LayerNorm,
        dropout: float = 0.0,
    ) -> None:
        """
        Arguments
        ---------
        num_blocks : int 
            number of feature extraction blocks for classification
        num_classes : int 
            number classes to be predicted
        image_size : int 
            spatial dimensions of the image
        channels: int 
            number of channels of the image
        patch_size : int
            size of the spatial dimensions of a single patch 
        dim : int 
            size of the embedding dimension
        ff_dim : int 
            hidden dimension of the feed forward layer
        norm : nn.Module
            normalization layer of the `PreNormResidual` layers
        dropout : float
            dropout probability
        """
        super(Discriminator, self).__init__()

        self.patch_embedding = PatchEmbedding(patch_size, channels, dim)
        self.discriminator_blocks = make_n_blocks(
            num_blocks, image_size, patch_size, dim, ff_dim, norm, dropout
        )
        self.classification_head = ClassificationHead(dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # B x C x H x W -> B x L x M
        embeds = self.patch_embedding(x)
        # B x L x M -> B x L x M
        features = self.discriminator_blocks(embeds)
        # B x L x M -> B x T
        predictions = self.classification_head(features)

        return predictions
