This is the implementation of Vision Transformer (ViT) paper (“An Image is Worth 16x16 Words”, ICLR 2021). 
---
Pipeline:

1. **Image preprocessing + patching (einops)**

* Each CIFAR-10 image is resized to **224×224**.
* Converted to a float tensor in **[0, 1]** with shape `(C, H, W)`.
* Patchified with patch size **16×16** using einops:

  * Input: `(C, 224, 224)`
  * Output tokens: `(N, P)` where

    * `N = (224/16) * (224/16) = 14 * 14 = 196` patches
    * `P = C * 16 * 16 = 3 * 256 = 768` values per patch (flattened)

So each sample becomes a sequence: **`(196, 768)`**.

2. **Patch embedding (Linear projection)**

* A linear layer maps each patch vector from `768 → embedding_dim` (default **768**).
* Output becomes `(B, 196, 768)`.

3. **ViT embeddings**

* A learnable **CLS token** of shape `(1,1,768)` is repeated across the batch and prepended:

  * `(B, 196, 768) → (B, 197, 768)`
* A learnable **positional embedding** of shape `(1, 197, 768)` is added.

4. **Transformer encoder**

* Stack of `n_layers` Transformer blocks (default **6** layers).
* Each block uses:

  * Multi-head self attention with `n_heads` (default **8**)
  * Feedforward/MLP hidden size `mlp_dim` (default **2048**)
  * `norm_first=True` → **Pre-LN** style (LayerNorm before attention/MLP)
* Final LayerNorm after the stack.

5. **Classification head**

* Uses the **CLS token output** `x[:, 0]` (shape `(B, 768)`)
* Linear layer maps `768 → 10` classes (CIFAR-10 logits).

---

## Training setup (Lightning)

### Data

* Dataset: **CIFAR-10** (train split downloaded to `/tmp`)
* Split ratios:

  * Train: **70%**
  * Val: **10%**
  * Test: **20%**
* Batch size: **128**
* DataLoader:

  * `shuffle=True` for training
  * `pin_memory=True`, `num_workers=2`

### Optimizer & schedule

* Optimizer: **AdamW**

  * learning rate: **3e-4**
  * weight decay: **0.05**
* LR schedule: **CosineAnnealingLR**

  * `T_max = max_epochs` (cosine over epochs)

### Training

* `max_epochs = 10`
* `accelerator="auto"`, `devices="auto"`
* Checkpoint:

  * monitors **`val_acc`**
  * saves best model (`save_top_k=1`)

### Compute tweak

* `torch.set_float32_matmul_precision('medium')` to speed up matmul on supported hardware.
