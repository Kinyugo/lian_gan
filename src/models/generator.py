import torch
import torch.nn as nn

from src.modules import (
    FeedForward,
    InversePatchEmbedding,
    MatMulTransform,
    PatchEmbedding,
    PreNormResidual,
)


def GeneratorBlock(
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
            GeneratorBlock(image_size, patch_size, dim, ff_dim, norm, dropout)
        )
    return nn.Sequential(*blocks)


class Generator(nn.Module):
    """
    Defines the generator for the image styling GAN.

    How it works
    ------------
    The forward pass takes in two tensor of shape B x C x H x W where 
    one of the tensor is the `content_images` and the other is the `style_images`. 
    Each of the tensor is embedded separately through a `PatchEmbedding` to 
    obtain tensors of shape B x L x M, where L is the number of patches and M
    is the embedding dimension. 
    Next the inputs are passed through `num_extractor_blocks`separate `GeneratorBlock`(s) 
    to extract the features from each of the images. The features are then summed. 
    The B x L x M sum of both content and style features, is passed through `num_reconstructor_blocks` 
    which act as the image reconstruction blocks.
    Afterwards the reconstructed blocks are passed through an `InversePatchEmbedding` layer to project
    and reshape the B x L x M tensor into a tensor of shape B x C x H x W. Finally a `Tanh` activation
    is applied.
    
    """

    def __init__(
        self,
        num_extractor_blocks: int,
        num_reconstructor_blocks: int,
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
        num_extractor_blocks : int 
            number of feature extraction blocks for both the content and style images
        num_reconstructor_blocks : int 
            number of reconstruction blocks for the final image
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
        super(Generator, self).__init__()

        self.content_patch_embedding = PatchEmbedding(patch_size, channels, dim)
        self.style_patch_embedding = PatchEmbedding(patch_size, channels, dim)
        self.content_extractor_blocks = make_n_blocks(
            num_extractor_blocks, image_size, patch_size, dim, ff_dim, norm, dropout
        )
        self.style_extractor_blocks = make_n_blocks(
            num_extractor_blocks, image_size, patch_size, dim, ff_dim, norm, dropout
        )
        self.reconstructor_blocks = make_n_blocks(
            num_reconstructor_blocks, image_size, patch_size, dim, ff_dim, norm, dropout
        )
        self.inv_patch_embedding = InversePatchEmbedding(
            image_size, patch_size, channels, dim
        )
        self.act = nn.Tanh()

    def forward(
        self, content_images: torch.Tensor, style_images: torch.Tensor
    ) -> torch.Tensor:
        # B x C x H x W -> B x L x M
        content_features = self.content_patch_embedding(content_images)
        style_features = self.style_patch_embedding(style_images)

        # B x L x M -> B x L x M
        content_features = self.content_extractor_blocks(content_features)
        style_features = self.style_extractor_blocks(style_features)

        # B x L x M, B x L x M -> B x L x M
        aggregate_features = torch.add(content_features, style_features)

        # B x L x M -> B x L x M
        styled_images = self.reconstructor_blocks(aggregate_features)
        # B x L x M -> B x C x H x W
        styled_images = self.inv_patch_embedding(styled_images)
        # B x C x H x W -> B x C x H x W
        styled_images = self.act(styled_images)

        return styled_images
