from argparse import ArgumentParser
from typing import List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_optimizer as optim
from src.models import Generator, VectorDiscriminator, Discriminator
from src.utils import make_image_grid


class LitGAN(pl.LightningModule):
    def __init__(
        self,
        num_extractor_blocks: int,
        num_reconstructor_blocks: int,
        num_blocks: int,
        image_size: int,
        channels: int = 3,
        patch_size: int = 16,
        dim: int = 256,
        ff_dim: int = 512,
        norm_layer: str = "layer_norm",
        dropout: float = 0.0,
        gen_learning_rate: float = 1e-3,
        disc_learning_rate: float = 1e-3,
        vec_disc_learning_rate: float = 1e-3,
        normalized: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        super().__init__()
        # save hyperparameters so they can be accessed in other methods
        self.save_hyperparameters()
        self.norm_layer = norm_layer
        self.normalized = normalized
        self.mean = mean
        self.std = std
        self.generator = Generator(
            num_extractor_blocks,
            num_reconstructor_blocks,
            image_size,
            channels,
            patch_size,
            dim,
            ff_dim,
            norm=self.get_norm_module,
            dropout=dropout,
        )
        self.discriminator = Discriminator(
            num_blocks,
            image_size,
            channels,
            patch_size,
            dim,
            ff_dim,
            norm=self.get_norm_module,
            dropout=dropout,
        )
        self.vec_discriminator = VectorDiscriminator(
            num_blocks,
            image_size,
            channels,
            patch_size,
            dim,
            ff_dim,
            norm=self.get_norm_module,
            dropout=dropout,
        )

    def forward(
        self, content_images: torch.Tensor, style_images: torch.Tensor
    ) -> torch.Tensor:
        return self.generator(content_images, style_images)

    def generator_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training step for generator. 

        1. Take content and style images from the batch.
        2. Pass both content and style images to the generator 
            to generate the styled images.
        3. Project the styled images using the VectorDiscriminator. 
        4. Calculate the content loss by sampling a vector around the point 0  and 
            finding the L2 norm pair-wise distance between the styled image projections
            and the sampled point. 
        5. Calculate the style loss by sampling a vector around the point 1 and 
            finding the L2 norm pair-wise distance between the styled image projections
            and the sampled point.
        6. Sum and backpropagate the losses.
        """
        content_images, style_images = x

        # generate styled images
        generated_images = self.generator(content_images, style_images)
        # classify the generate images using the discriminator
        d_output = torch.squeeze(self.discriminator(generated_images))

        # project the generated images using the vector discriminator
        content_projections = self.vec_discriminator(content_images)
        style_projections = self.vec_discriminator(style_images)
        generated_projections = self.vec_discriminator(generated_images)

        # calculate the losses
        g_loss = nn.BCELoss()(d_output, torch.ones_like(d_output))
        pdist = nn.PairwiseDistance(p=2, eps=1e-8)
        content_loss = pdist(generated_projections, content_projections)
        style_loss = pdist(generated_projections, style_projections)

        loss = (g_loss + content_loss + style_loss).mean(dim=-1)

        return loss

    def discriminator_step(self, x: torch.Tensor) -> torch.Tensor:
        content_images, style_images = x

        # Real images
        d_output = torch.squeeze(self.discriminator(style_images))
        loss_real = nn.BCELoss()(d_output, torch.ones_like(d_output))

        # Fake images
        generated_imgs = self.generator(content_images, style_images)
        d_output = torch.squeeze(self.discriminator(generated_imgs))
        loss_fake = nn.BCELoss()(d_output, torch.zeros_like(d_output))

        return loss_real + loss_fake

    def vec_discriminator_step(self, x: torch.Tensor) -> torch.Tensor:
        """
        Training step for discriminator. 

        1. Get the real content and style images.
        2. Pass both content and style images to the generator 
            to generate the styled images.
        3. Project the content, styled and generated images using the VectorDiscriminator. 
        4. Calculate the content loss by sampling a vector around the point 0  and 
            finding the L2 norm pair-wise distance between the styled image projections
            and the sampled point. 
        5. Calculate the style loss by sampling a vector around the point 1 and 
            finding the L2 norm pair-wise distance between the styled image projections
            and the sampled point.
        6. Calculate the generated loss by sampling a vector around the point -1 and 
            finding the L2 norm pair-wise distance between the generated image projections
            and the sampled point.
        """
        content_images, style_images = x

        # generate synthetic images using the generator
        generated_images = self.generator(content_images, style_images)

        # project the images to a point using the discriminator
        content_projections = self.discriminator(content_images)
        style_projections = self.discriminator(style_images)
        generated_projections = self.discriminator(generated_images)

        # sample actual projections from a uniform distribution
        actual_content_projections = torch.distributions.Uniform(-0.1, 0.1).sample(
            content_projections.size()
        )
        actual_style_projections = torch.distributions.Uniform(0.9, 1.1).sample(
            style_projections.size()
        )
        actual_generated_projections = torch.distributions.Uniform(-1.1, -0.9).sample(
            generated_projections.size()
        )

        # calculate the losses for each of projections
        pdist = nn.PairwiseDistance(p=2, eps=1e-8)
        content_loss = pdist(actual_content_projections, content_projections)
        style_loss = pdist(actual_style_projections, style_projections)
        generated_loss = pdist(actual_generated_projections, generated_projections)

        loss = (content_loss + style_loss + generated_loss).mean(dim=-1)

        return loss

    def training_step(
        self, batch: torch.Tensor, batch_idx: int, optimizer_idx: int
    ) -> torch.Tensor:
        # train generator
        if optimizer_idx == 0:
            loss = self.generator_step(batch)
            self.log("G_loss", loss)

        # train discriminator
        if optimizer_idx == 1:
            loss = self.discriminator_step(batch)
            self.log("D_loss", loss)

        # train the vector discriminator
        if optimizer_idx == 2:
            loss = self.vec_discriminator_step(batch)
            self.log("VD_loss", loss)

        return loss

    def validation_step(self, batch, optimizer_idx) -> None:
        content_images, style_images = batch
        generated_images = self.generator(content_images, style_images)
        generated_image_grid = make_image_grid(
            generated_images, self.normalized, self.mean, self.std
        )
        self.logger.experiment.add_image(
            "training_progression", generated_image_grid, self.current_epoch
        )

    def configure_optimizers(self):
        gen_optim = optim.RAdam(
            self.generator.parameters(), lr=self.hparams.gen_learning_rate
        )
        disc_optim = optim.RAdam(
            self.generator.parameters(), lr=self.hparams.disc_learning_rate
        )
        vec_disc_optim = optim.RAdam(
            self.discriminator.parameters(), lr=self.hparams.vec_disc_learning_rate
        )

        return gen_optim, disc_optim, vec_disc_optim

    @property
    def get_norm_module(self) -> nn.Module:
        return nn.LayerNorm if self.norm_layer == "layer_norm" else nn.BatchNorm

    @staticmethod
    def add_model_specific_args(parent_parser: ArgumentParser) -> ArgumentParser:
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        # add generator model arguments
        parser.add_argument("--num_extractor_blocks", type=int, default=1)
        parser.add_argument("--num_reconstructor_blocks", type=int, default=1)
        parser.add_argument("--image_size", type=int, default=64)
        parser.add_argument("--channels", type=int, default=3)
        parser.add_argument("--patch_size", type=int, default=16)
        parser.add_argument("--dim", type=int, default=256)
        parser.add_argument("--ff_dim", type=int, default=512)
        parser.add_argument("--norm_layer", type=str, default="layer_norm")
        parser.add_argument("--dropout", type=float, default=0.0)
        # add discriminator model arguments
        parser.add_argument("--num_blocks", type=int, default=2)
        # add optimizer specific arguments
        parser.add_argument("--gen_learning_rate", type=float, default=1e-2)
        parser.add_argument("--disc_learning_rate", type=float, default=1e-2)

        return parser

