# ResNet-18 Image Classification on CIFAR-10

This implementation provides a complete training pipeline for ResNet-18 on the CIFAR-10 dataset using PyTorch Lightning.

---

## Overview

**Architecture:** ResNet-18 (Residual Network with 18 layers)  
**Dataset:** CIFAR-10 (10-class image classification)  
**Framework:** PyTorch Lightning (for structured training)  
**Input resolution:** 224×224 pixels  
**Output:** 10 class logits

---

## Components

### 1. **CIFAR10DataModule** (LightningDataModule)
Handles data loading, preprocessing, and splitting.

**Configuration:**
- `data_dir`: Cache directory for datasets (default: `/tmp`)
- `batch_size`: Batch size for training (default: 256)
- `num_workers`: Number of data loading workers (default: 4)
- `val_split`: Validation set size (default: 5000 samples)
- `seed`: Random seed for reproducibility (default: 42)

**Data normalization:** CIFAR-10 ImageNet statistics
- Mean: `(0.4914, 0.4822, 0.4465)`
- Std: `(0.2023, 0.1994, 0.2010)`

**Augmentations (training only):**
- Resize to 224×224
- Random crop with padding=4
- Random horizontal flip
- Normalize with CIFAR-10 stats

**Data splits:**
- Train: 45,000 samples
- Validation: 5,000 samples
- Test: 10,000 samples

---

### 2. **StemLayer**
Initial feature extraction block (applied to all input images).

**Architecture:**
- Conv2d(3→64, kernel=7, stride=2, padding=3) → **(B, 64, 112, 112)**
- BatchNorm2d(64)
- ReLU activation
- MaxPool2d(kernel=3, stride=2, padding=1) → **(B, 64, 56, 56)**

---

### 3. **BasicBlock**
Residual block (used in ResNet layers).

**Components:**
- Conv2d(in_channels → out_channels, kernel=3, stride, padding=1)
- BatchNorm2d
- ReLU
- Conv2d(out_channels → out_channels, kernel=3, stride=1, padding=1)
- BatchNorm2d
- **Residual connection** (shortcut) with optional downsample projection
- ReLU

**Downsample:** Applied when stride≠1 or in_channels≠out_channels
- 1×1 Conv projection + BatchNorm

---

### 4. **Resnet18**
Full ResNet-18 model.

**Architecture:**
```
Input: (B, 3, 224, 224)
  ↓
StemLayer: (B, 64, 56, 56)
  ↓
Layer1 (2 blocks, stride=1): (B, 64, 56, 56)
  ↓
Layer2 (2 blocks, stride=2): (B, 128, 28, 28)
  ↓
Layer3 (2 blocks, stride=2): (B, 256, 14, 14)
  ↓
Layer4 (2 blocks, stride=2): (B, 512, 7, 7)
  ↓
AdaptiveAvgPool2d: (B, 512)
  ↓
Linear(512 → 10): (B, 10) logits
```

**Total blocks:** 4 residual layers × 2 blocks = 8 BasicBlocks (16 conv layers + stem = 18 total)

---

### 5. **LitResNet18** (LightningModule)
PyTorch Lightning wrapper for training.

**Components:**
- `model`: Resnet18 instance
- `criterion`: CrossEntropyLoss for classification
- `optimizer`: Adam (configurable learning rate)

**Training:**
- `training_step`: Computes loss + accuracy, logs metrics
- `validation_step`: Evaluates on validation set
- `test_step`: Evaluates on test set
- `configure_optimizers`: Returns Adam optimizer

---

## Training Configuration

**Trainer settings:**
- `max_epochs`: 10
- `accelerator`: "auto" (CPU/GPU auto-detection)
- `devices`: "auto" (use all available GPUs/CPUs)
- `log_every_n_steps`: 10

**Optimizer:**
- **Adam** with learning rate = 1e-3 (0.001)

---

## Running the Script

```bash
python train.py
```

This will:
1. Load/download CIFAR-10 dataset to `/tmp`
2. Create train/val/test splits
3. Train ResNet-18 for 10 epochs
4. Log training/validation metrics
5. Run evaluation on the test set

---

## Expected Output

Logs during training:
```
Epoch 0: train_loss=..., train_acc=..., val_loss=..., val_acc=...
...
Epoch 9: train_loss=..., train_acc=..., val_loss=..., val_acc=...
Testing: test_loss=..., test_acc=...
```

---

## Performance Notes

- **Batch size 512** enables faster training on GPUs
- **pin_memory=True** optimizes GPU data transfer
- **num_workers=4** enables parallel data loading
- Training typically converges in 10 epochs with good accuracy

---

## Customization

To modify training, edit the `__main__` block:

```python
# Change batch size
datamodule = CIFAR10DataModule(batch_size=256)

# Change learning rate
model = LitResNet18(lr=1e-4)

# Change epochs
trainer = pl.Trainer(max_epochs=20, ...)
```
