from dataclasses import dataclass
from typing import Optional
import lightning
import pytorch_lightning as pl
from datasets import load_dataset
import torch
import os
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import transforms
import warnings
from transformers import AutoTokenizer, AutoModelForCausalLM
import torchvision
warnings.filterwarnings("ignore")

RESOLUTION_SIZE = 224
INIT_TEMPERATURE = 0.07

@dataclass
class DataConfig:
    data_dir: str = "~/.cache/"
    batch_size: int = 128
    num_workers: int = 4
    pin_memory: bool = True

class Flickr8kDataModule(pl.LightningDataModule):
    def __init__(self, cfg: DataConfig):
        super().__init__()
        self.cfg = cfg
        self.transforms = transforms.Compose([
            transforms.Resize((RESOLUTION_SIZE, RESOLUTION_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],  # ImageNet stats
                std=[0.229, 0.224, 0.225]
            )
        ])
        self.train_ds = None
        self.val_ds = None
        self.test_ds = None

    def collate_fn(self, batch):
        return {
            'image': torch.stack([item['image'] for item in batch]),
            'caption': [item['caption'] for item in batch]
        }

    def transform_fn(self, examples):
        images = examples["image"]
        captions = examples["caption"]
        
        
        return {
            "image": [self.transforms(img) for img in images],
            "caption": captions
        }

    def setup(self, stage: Optional[str] = None):
        print("Loading data...")
        ds = load_dataset("jxie/flickr8k", cache_dir=self.cfg.data_dir)
        
        ds = ds.map(
            lambda x: {"caption": x["caption_0"]},
            remove_columns=[f"caption_{i}" for i in range(5)]
        )
        
        ds = ds.with_transform(self.transform_fn)
        
        self.train_ds = ds["train"]
        self.val_ds = ds["validation"]
        self.test_ds = ds["test"]

    def train_dataloader(self):
        return DataLoader(
            self.train_ds, 
            batch_size=self.cfg.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_ds, 
            batch_size=self.cfg.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds, 
            batch_size=self.cfg.batch_size, 
            collate_fn=self.collate_fn,
            num_workers=self.cfg.num_workers,
            pin_memory=self.cfg.pin_memory
        )


class AttentionPooling(nn.Module):
    def __init__(self):
        super(AttentionPooling, self).__init__()

    def forward(self, x):

        return x

class ModifiedResnet(nn.Module):
    def __init__(self):
        super(ModifiedResnet, self).__init__()

    def forward(self, x):

        return x



class ImageEncoder:
    def __init__(self, backbone : str):
        if backbone == "resnet":
            self.backbone_model = None
        elif backbone == "vit":
            self.backbone_model = None
        
class TextEncoder(nn.Module):
    def __init__(self, backbone_model: str):
        super().__init__()

        self.tokenizer = AutoTokenizer.from_pretrained(backbone_model)
        self.model = AutoModelForCausalLM.from_pretrained(
            backbone_model,
            output_hidden_states=True
        )
        self.model.eval()

    def forward(self, x):
        inputs = self.tokenizer(
            x,
            return_tensors="pt",
            padding=True,
            truncation=True
        )

        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True
            )

        token_embeddings = outputs.hidden_states[-1]

        return token_embeddings



class ContrastiveLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.logit_scale = nn.Parameter(
            torch.ones([]) * torch.log(torch.tensor(1 / INIT_TEMPERATURE))
        )

    def forward(self, image_vec: torch.Tensor, text_vec: torch.Tensor):
        image_vec = F.normalize(image_vec, dim=1)
        text_vec  = F.normalize(text_vec, dim=1)
        logits = self.logit_scale.exp() * image_vec @ text_vec.T
        N = image_vec.size(0)
        labels = torch.arange(N, device=image_vec.device)
        loss_i2t = F.cross_entropy(logits, labels)
        loss_t2i = F.cross_entropy(logits.T, labels)

        return (loss_i2t + loss_t2i) / 2

if __name__ == "__main__":
    cfg = DataConfig(batch_size=64)
    flickr8k = Flickr8kDataModule(cfg)
    flickr8k.setup()
    
    train_loader = flickr8k.train_dataloader()
    batch = next(iter(train_loader))
    
    print("Image shape:", batch["image"].shape)
    print("Caption type:", type(batch["caption"]))
    print("Caption length:", len(batch["caption"]))
    print("First caption:", batch["caption"][0])