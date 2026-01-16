import warnings
warnings.filterwarnings("ignore")
import torch
from einops import rearrange
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from einops import repeat
from torch.utils.data import random_split, DataLoader
import torchvision
import torchvision.transforms as transforms
import lightning as L
import torch
import torch.nn.functional as F
from lightning.pytorch.callbacks import ModelCheckpoint
torch.set_float32_matmul_precision('medium')

## Load and processing data
C, H, W = 3, 224, 224
BATCH_SIZE = 128
PATCH_SIZE = 16
EMBEDDING_DIM = 768

class CIFAR10DataModule(L.LightningDataModule):
    def __init__(self, data_dir="/tmp", batch_size=16, num_workers=2):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transforms.Compose([
            transforms.Resize((H, W)),
            transforms.ToTensor(),
            transforms.Lambda(lambda x: rearrange(
                x, 'c (h p1) (w p2) -> (h w) (c p1 p2)',
                p1=PATCH_SIZE, p2=PATCH_SIZE
            ))
        ])

    def setup(self, stage=None):
        full = torchvision.datasets.CIFAR10(self.data_dir, train=True, download=True, transform=self.transform)
        total = len(full)
        train_size = int(0.7 * total)
        val_size = int(0.1 * total)
        test_size = total - train_size - val_size
        self.train_set, self.val_set, self.test_set = random_split(full, [train_size, val_size, test_size])

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, shuffle=True,
                          num_workers=self.num_workers, pin_memory=True)

    def val_dataloader(self):
        return DataLoader(self.val_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True)


class LinearProjection(nn.Module):
    def __init__(self, output_dim):
        super(LinearProjection, self).__init__()
        self.patch_embed = nn.Linear(768, output_dim)
    def forward(self, x):
        return self.patch_embed(x)

class ViTEmbedding(nn.Module):
    def __init__(self, embedding_size : int):
        super(ViTEmbedding, self).__init__()
        self.cls_token = nn.Parameter(torch.randn((1, 1, embedding_size)))
        self.pos_embedding = nn.Parameter(torch.randn((1, 196 + 1, embedding_size)))

    def forward(self, x):
        num_batch = x.shape[0]
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b = num_batch)
        x = torch.cat([cls_tokens, x], dim = 1)
        x += self.pos_embedding
        return x

class TransformerEncoder(nn.Module):
    def __init__(self, n_layers : int, n_heads : int, mlp_dim : int, embedding_dim : int):
        super(TransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(
                d_model=embedding_dim, 
                nhead=n_heads, 
                dim_feedforward=mlp_dim, 
                batch_first=True,
                norm_first = True
            )
            for _ in range(n_layers)
        ])
        self.final_norm = nn.LayerNorm(embedding_dim)        

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        x = self.final_norm(x)
        return x
    
class ClassificationHead(nn.Module):
    def __init__(self, embedding_dim, num_classes):
        super(ClassificationHead, self).__init__()
        self.fc = nn.Linear(embedding_dim, num_classes)
    
    def forward(self, x):
        cls_token = x[:, 0]  # (B, embedding_dim)
        logits = self.fc(cls_token)  # (B, num_classes)
        return logits
    
class ViT(nn.Module):
    def __init__(self, embedding_dim=768, n_layers=6, n_heads=8, mlp_dim=2048, num_classes=10):
        super().__init__()
        self.proj = LinearProjection(embedding_dim)              # (B,196,768) -> (B,196,D)
        self.embed = ViTEmbedding(embedding_dim)                 # add CLS + pos -> (B,197,D)
        self.encoder = TransformerEncoder(n_layers, n_heads, mlp_dim, embedding_dim)
        self.head = ClassificationHead(embedding_dim, num_classes)

    def forward(self, x):
        x = self.proj(x)
        x = self.embed(x)
        x = self.encoder(x)
        x = self.head(x)
        return x


class LitViT(L.LightningModule):
    def __init__(self, model: nn.Module, lr=3e-4, weight_decay=0.05):
        super().__init__()
        self.model = model
        self.lr = lr
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def _shared_step(self, batch, stage: str):
        x, y = batch                       # x: (B,196,768), y: (B,)
        logits = self(x)                   # (B,10)
        loss = F.cross_entropy(logits, y)
        acc = (logits.argmax(dim=1) == y).float().mean()

        # log to progress bar + logger
        self.log(f"{stage}_loss", loss, prog_bar=True, on_epoch=True, on_step=False)
        self.log(f"{stage}_acc", acc,   prog_bar=True, on_epoch=True, on_step=False)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._shared_step(batch, "test")

    def configure_optimizers(self):
        opt = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        # optional scheduler (simple, safe default)
        sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=self.trainer.max_epochs)
        return {"optimizer": opt, "lr_scheduler": sch}



datamodule = CIFAR10DataModule(batch_size=BATCH_SIZE)

vit = ViT(
    embedding_dim=EMBEDDING_DIM,
    n_layers=6,
    n_heads=8,
    mlp_dim=2048,
    num_classes=10
)

lit_model = LitViT(vit, lr=3e-4, weight_decay=0.05)

ckpt = ModelCheckpoint(monitor="val_acc", mode="max", save_top_k=1)

trainer = L.Trainer(
    max_epochs=10,
    accelerator="auto",
    devices="auto",
    callbacks=[ckpt],
    log_every_n_steps=10
)

trainer.fit(lit_model, datamodule=datamodule)
trainer.test(lit_model, datamodule=datamodule, ckpt_path="best")
