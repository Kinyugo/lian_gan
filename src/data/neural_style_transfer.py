import os
import random
from argparse import ArgumentParser
from typing import Callable, Optional, Tuple, Union

import pytorch_lightning as pl
import torch
import torchvision.transforms as T
from PIL import Image
from torch.utils.data import DataLoader, Dataset, random_split


class NSTDataset(Dataset):
    def __init__(
        self,
        root_dir: str = None,
        shuffle: str = True,
        content_transform: Callable = None,
        style_transform: Callable = None,
    ) -> None:
        super().__init__()
        self.root_dir = root_dir
        self.content_transform = content_transform
        self.style_transform = style_transform

        content_filenames = os.listdir(os.path.join(root_dir, "content"))
        style_filenames = os.listdir(os.path.join(root_dir, "style"))
        if shuffle:
            random.shuffle(content_filenames)
            random.shuffle(style_filenames)

        self.content_filenames = content_filenames
        self.style_filenames = style_filenames

    def __len__(self):
        return len(self.content_filenames)

    def __getitem__(self, index) -> Tuple[torch.Tensor, torch.Tensor]:
        content_filename = self.content_filenames[index]
        style_filename = self.style_filenames[index]

        content_image = Image.open(
            os.path.join(self.root_dir, "content", content_filename)
        )
        style_image = Image.open(os.path.join(self.root_dir, "style", style_filename))

        if self.content_transform:
            content_image = self.content_transform(content_image)
        if self.style_transform:
            style_image = self.style_transform(style_image)

        return (content_image, style_image)


class NSTDataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        val_samples: int,
        content_transform: Callable = None,
        style_transform: Callable = None,
        dims: Union[Tuple[int, int], int] = (64, 64),
        num_workers: int = 1,
        shuffle: bool = True,
        pin_memory: bool = False,
        drop_last: bool = False,
    ) -> None:

        super().__init__()

        self.data_dir = data_dir
        self.batch_size = batch_size
        self.val_samples = val_samples
        self.content_transform = content_transform
        self.style_transform = style_transform
        self.dims = dims
        self.num_workers = num_workers
        self.shuffle = shuffle
        self.pin_memory = pin_memory
        self.drop_last = drop_last

        # setup default transforms incase the user didn't pass any
        if self.content_transform is None or self.style_transform is None:
            self._setup_default_transforms_()

    def setup(self, stage: Optional[str] = None) -> None:
        full_dataset = NSTDataset(
            self.data_dir,
            shuffle=self.shuffle,
            content_transform=self.content_transform,
            style_transform=self.style_transform,
        )
        if len(full_dataset) > 1:
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [len(full_dataset) - self.val_samples, self.val_samples]
            )
        else:
            self.train_dataset, self.val_dataset = full_dataset, full_dataset

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            drop_last=self.drop_last,
        )

    def _setup_default_transforms_(self) -> None:
        default_transform = T.Compose(
            [
                T.Resize(size=self.dims),
                T.ToTensor(),
                T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
        if self.content_transform is None:
            self.content_transform = default_transform
        if self.style_transform is None:
            self.style_transform = default_transform

    @staticmethod
    def add_argparse_args(parent_parser: ArgumentParser):
        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument(
            "--data_dir",
            type=str,
            help="directory containing the content and style folders",
        )
        parser.add_argument(
            "--batch_size",
            type=int,
            default=64,
            help="number of samples in a mini batch",
        )
        parser.add_argument(
            "--val_samples", type=int, default=64, help="number of validation samples"
        )
        parser.add_argument(
            "--img_dims",
            type=int,
            default=64,
            help="spatial dimensions of the preprocessed image",
        )
        parser.add_argument(
            "--num_workers",
            type=int,
            default=4,
            help="number of worker processes for data loading",
        )
        parser.add_argument(
            "--shuffle",
            type=bool,
            default=True,
            help="whether to shuffle training data",
        )
        parser.add_argument(
            "--pin_memory",
            type=bool,
            default=False,
            help="transfer batches into gpu memory",
        )
        parser.add_argument(
            "--drop_last",
            type=bool,
            default=False,
            help="whether to drop the last incomplete batch",
        )

        return parser
