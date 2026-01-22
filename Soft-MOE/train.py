from __future__ import annotations
from dataclasses import dataclass
from typing import Optional, Tuple
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, random_split
from torchvision import transforms
from torchvision.datasets import CIFAR10
import torchvision.models as models
import lightning as L
import warnings

warnings.filterwarnings("ignore")

N_EXPERTS = 4


@dataclass
class CIFAR10DataModule(L.LightningDataModule):
    data_dir: str = "/tmp"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True
    persistent_workers: bool = True

    val_split: float = 0.1
    seed: int = 42

    # optional
    download: bool = True

    def __post_init__(self):
        super().__init__()
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

        # CIFAR-10 normalization stats (standard)
        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)

    def _train_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _test_transforms(self):
        return transforms.Compose(
            [
                transforms.Resize(size=(224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def prepare_data(self) -> None:
        # Download only (no state)
        CIFAR10(self.data_dir, train=True, download=self.download)
        CIFAR10(self.data_dir, train=False, download=self.download)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            full_train = CIFAR10(
                self.data_dir, train=True, transform=self._train_transforms(), download=False
            )

            n_total = len(full_train)
            n_val = int(n_total * self.val_split)
            n_train = n_total - n_val

            generator = torch.Generator().manual_seed(self.seed)
            self.train_dataset, self.val_dataset = random_split(
                full_train, [n_train, n_val], generator=generator
            )

            # IMPORTANT: val should NOT use train augmentations
            # random_split shares the same underlying dataset object,
            # so we clone val dataset by reloading with test transforms and using indices
            base_val = CIFAR10(
                self.data_dir, train=True, transform=self._test_transforms(), download=False
            )
            self.val_dataset.dataset = base_val  # swap transforms under the hood

        if stage in (None, "test", "predict"):
            self.test_dataset = CIFAR10(
                self.data_dir, train=False, transform=self._test_transforms(), download=False
            )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
            drop_last=True,
        )

    def val_dataloader(self) -> DataLoader:
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )

    def test_dataloader(self) -> DataLoader:
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=self.persistent_workers and self.num_workers > 0,
        )


class SoftMoE(nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        num_experts: int,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.gate = nn.Linear(input_dim, num_experts)
        self.experts = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(input_dim, hidden_dim),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim, output_dim),
                )
                for _ in range(num_experts)
            ]
        )

    def forward(
        self, x: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        if x.dim() == 3:
            batch_size, num_tokens, hidden = x.shape
            x_flat = x.reshape(batch_size * num_tokens, hidden)
            gate_logits = self.gate(x_flat)
            gate = F.softmax(gate_logits, dim=-1)
            expert_out = torch.stack([expert(x_flat) for expert in self.experts], dim=1)
            mixed = (gate.unsqueeze(-1) * expert_out).sum(dim=1)
            mixed = mixed.reshape(batch_size, num_tokens, -1)
            if return_gate:
                return mixed, gate
            return mixed

        gate_logits = self.gate(x)
        gate = F.softmax(gate_logits, dim=-1)
        expert_out = torch.stack([expert(x) for expert in self.experts], dim=1)
        mixed = (gate.unsqueeze(-1) * expert_out).sum(dim=1)
        if return_gate:
            return mixed, gate
        return mixed


class SoftMoEClassifier(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        num_experts: int = N_EXPERTS,
        hidden_dim: int = 1024,
        dropout: float = 0.1,
    ):
        super().__init__()
        backbone = models.resnet18(weights=None)
        backbone.fc = nn.Identity()
        self.backbone = backbone
        self.moe_head = SoftMoE(
            input_dim=512,
            hidden_dim=hidden_dim,
            output_dim=num_classes,
            num_experts=num_experts,
            dropout=dropout,
        )

    def forward(
        self, x: torch.Tensor, return_gate: bool = False
    ) -> torch.Tensor | Tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(x)
        if return_gate:
            logits, gate = self.moe_head(features, return_gate=True)
            return logits, gate
        logits = self.moe_head(features)
        return logits


class LitSoftMoE(L.LightningModule):
    def __init__(
        self,
        model: nn.Module,
        lr: float = 3e-4,
        weight_decay: float = 0.05,
        aux_loss_coef: float = 0.1,
    ):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay
        self.aux_loss_coef = aux_loss_coef

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)

    def _shared_step(self, batch, stage: str) -> torch.Tensor:
        x, y = batch
        logits, gate = self(x, return_gate=True)
        loss = F.cross_entropy(logits, y)
        mean_gate = gate.mean(dim=0)
        uniform_kl = (mean_gate * (mean_gate.clamp_min(1e-8).log() + math.log(gate.size(-1)))).sum()
        aux_loss = self.aux_loss_coef * uniform_kl
        loss = loss + aux_loss
        acc = (logits.argmax(dim=1) == y).float().mean()
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_aux_loss", aux_loss, prog_bar=False, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sch}


if __name__ == "__main__":
    dm = CIFAR10DataModule()
    model = SoftMoEClassifier(num_classes=10, num_experts=N_EXPERTS)
    lit_model = LitSoftMoE(model)

    trainer = L.Trainer(
        max_epochs=10,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(lit_model, datamodule=dm)
    trainer.test(lit_model, datamodule=dm)
