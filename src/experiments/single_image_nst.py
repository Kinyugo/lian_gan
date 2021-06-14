# All experiments should be run from the root directory
# so that the src directory can be found by imports
# an alternative is to install the src package
# ------------
# setup
# ------------
import os
import sys

sys.path.append(os.getcwd())

# ------------
# imports
# ------------
from argparse import ArgumentParser

import pytorch_lightning as pl
from src.data import NSTDataModule
from src.models import LitNSTGAN


def cli_main():
    pl.seed_everything(254)

    # ------------
    # args
    # ------------
    parser = ArgumentParser()
    parser.add_argument("--logs_name", type=str, default="single_image_nst")
    parser.add_argument(
        "--checkpoints_dir", type=str, default="checkpoints/single_image_nst"
    )

    parser.add_argument(
        "--checkpoint_filename",
        type=str,
        default="single_image_nst_{epoch:.2f}_{G_loss:.2f}",
    )
    parser = LitNSTGAN.add_model_specific_args(parser)
    parser = NSTDataModule.add_argparse_args(parser)
    parser = pl.Trainer.add_argparse_args(parser)
    args = parser.parse_args()

    # ------------
    # data
    # ------------
    datamodule = NSTDataModule(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        val_samples=args.val_samples,
        dims=(args.image_size, args.image_size),
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
    )

    # ------------
    # model
    # ------------
    model = LitNSTGAN(
        num_extractor_blocks=args.num_extractor_blocks,
        num_reconstructor_blocks=args.num_reconstructor_blocks,
        num_blocks=args.num_blocks,
        image_size=args.image_size,
        channels=args.channels,
        patch_size=args.patch_size,
        dim=args.dim,
        ff_dim=args.ff_dim,
        norm_layer=args.norm_layer,
        dropout=args.dropout,
        gen_learning_rate=args.gen_learning_rate,
    )

    # ------------
    # training
    # ------------
    logger = pl.loggers.TensorBoardLogger(save_dir="logs", name=args.logs_name)
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=args.checkpoints_dir,
        monitor="G_loss",
        mode="min",
        filename=args.checkpoint_filename,
    )

    trainer = pl.Trainer.from_argparse_args(args)
    trainer.callbacks = [checkpoint_callback, *trainer.callbacks]
    trainer.logger = logger

    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    cli_main()
