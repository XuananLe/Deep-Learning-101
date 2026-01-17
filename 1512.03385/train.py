import os
from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
import pytorch_lightning as pl

RESOLUTION_SIZE = 224
# input (3, 224, 224)

class CIFAR10DataModule(pl.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "/tmp",
        batch_size: int = 256,
        num_workers: int = 4,
        val_split: int = 5000,
        seed: int = 42,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2023, 0.1994, 0.2010)

    def prepare_data(self):
        datasets.CIFAR10(self.data_dir, train=True, download=True)
        datasets.CIFAR10(self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None):

        train_transform = transforms.Compose([
            transforms.Resize((RESOLUTION_SIZE, RESOLUTION_SIZE)),
            transforms.RandomCrop(RESOLUTION_SIZE, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        test_transform = transforms.Compose([
            transforms.Resize((RESOLUTION_SIZE, RESOLUTION_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        if stage == "fit" or stage is None:
            full_train = datasets.CIFAR10(
                self.data_dir, train=True, transform=train_transform
            )

            val_size = self.val_split
            train_size = len(full_train) - val_size

            self.train_set, self.val_set = random_split(
                full_train,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(self.seed),
            )

            self.val_set.dataset.transform = test_transform

        if stage == "test" or stage is None:
            self.test_set = datasets.CIFAR10(
                self.data_dir, train=False, transform=test_transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_set,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_set,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

# torch.nn.Conv2d(
#     in_channels,
#     out_channels,
#     kernel_size,
#     stride=1,
#     padding=0,
#     dilation=1,
#     groups=1,
#     bias=True
# )


class StemLayer(nn.Module):
    def __init__(self):
        super(StemLayer, self).__init__()
        self.conv = nn.Conv2d(3, 64, 7, 2, padding = 3)
        self.bn = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.max_pool = nn.MaxPool2d(kernel_size=3, stride = 2, padding = 1)

    def forward(self, x):
        x = self.conv(x) # (B, 64, 112, 112)
        x = self.bn(x) 
        x = self.relu(x)
        x = self.max_pool(x) # (B, 64, 56, 56)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels, kernel_size=3,
            stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels, kernel_size=3,
            stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(identity)

        out += identity
        out = self.relu(out)

        return out


class Resnet18(nn.Module):
    def __init__(self):
        super(Resnet18, self).__init__()
        self.in_channels = 64
        self.num_classes = 10
        self.stem = StemLayer()
        self.layer_1 = self._make_layer(64, 2, 1)
        self.layer_2 = self._make_layer(128, 2, 2)
        self.layer_3 = self._make_layer(256, 2, 2)
        self.layer_4 = self._make_layer(512, 2, 2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, self.num_classes)


    def _make_layer(self, out_channels, num_blocks, stride):
        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(
                BasicBlock(self.in_channels , out_channels, stride)
            )
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.stem(x)
        x = self.layer_1(x)
        x = self.layer_2(x)
        x = self.layer_3(x)
        x = self.layer_4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class LitResNet18(pl.LightningModule):
    def __init__(self, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()
        self.model = Resnet18()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("train_loss", loss, prog_bar=True)
        self.log("train_acc", acc, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        self.log("test_loss", loss)
        self.log("test_acc", acc)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


if __name__ == "__main__":
    datamodule = CIFAR10DataModule(batch_size=512)

    model = LitResNet18(lr=1e-3)

    trainer = pl.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule)
