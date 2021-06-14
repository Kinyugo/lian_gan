from argparse import ArgumentParser
from typing import Callable, List

import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch_optimizer as optim
from src.models import Generator
from src.utils import StyleLoss, make_image_grid
from .perceptual_vgg19 import PerceptualVgg19


class LitNSTGAN(pl.LightningModule):
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
        normalized: bool = True,
        mean: List[float] = [0.485, 0.456, 0.406],
        std: List[float] = [0.229, 0.224, 0.225],
    ) -> None:
        super().__init__()
        # record hyperparameters for logging and access in
        # other methods
        self.save_hyperparameters()

        self.generator = Generator(
            num_extractor_blocks,
            num_reconstructor_blocks,
            image_size,
            channels,
            patch_size,
            dim,
            ff_dim,
            norm=self.norm_module,
            dropout=dropout,
        )

        self.content_extractor = PerceptualVgg19(feature_layers=[11])
        self.style_extractor = PerceptualVgg19(feature_layers=[1, 3, 6, 11, 15])
        self.style_loss = StyleLoss()
        self.content_loss = nn.MSELoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        content_images, style_images = x
        return self.generator(content_images, style_images)

    def training_step(self, batch: torch.Tensor, batch_idx: int) -> torch.Tensor:
        # train the generator
        loss = self.generator_training_step(batch)
        self.log("G_loss", loss)

        return loss

    def validation_step(self, batch, batch_idx) -> None:
        content_images, style_images = batch
        generated_images = self.generator(content_images, style_images)
        generated_image_grid = make_image_grid(
            generated_images,
            self.hparams.normalized,
            self.hparams.mean,
            self.hparams.std,
        )
        self.logger.experiment.add_image(
            "training_progression", generated_image_grid, self.current_epoch
        )

    def configure_optimizers(self):
        g_optim = optim.RAdam(
            self.generator.parameters(), lr=self.hparams.gen_learning_rate
        )

        return g_optim

    def generator_training_step(self, x: torch.Tensor) -> torch.Tensor:
        content_images, style_images = x
        # generate styled image from the generator
        generated_images = self.generator(content_images, style_images)
        # extract style and content features
        real_content_features = self.content_extractor(content_images)
        generated_content_features = self.content_extractor(generated_images)
        real_style_features = self.style_extractor(style_images)
        generated_style_features = self.style_extractor(generated_images)

        avg_content_loss = self.calculate_avg_loss(
            generated_content_features, real_content_features, self.style_loss
        )
        avg_style_loss = self.calculate_avg_loss(
            generated_style_features, real_style_features, self.content_loss
        )

        return avg_content_loss + avg_style_loss

    def calculate_avg_loss(
        self, preds: List[torch.Tensor], targets: List[torch.Tensor], loss_fn: Callable
    ) -> torch.Tensor:
        total_loss = 0.0
        for pred, target in zip(preds, targets):
            total_loss += loss_fn(pred, target)

        return total_loss / len(preds)

    @property
    def norm_module(self) -> nn.Module:
        return nn.LayerNorm if self.hparams.norm_layer == "layer_norm" else nn.BatchNorm

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
