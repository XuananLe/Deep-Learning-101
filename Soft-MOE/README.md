# Soft-MoE (CIFAR-10)

This folder implements a small, single-file Soft Mixture-of-Experts (Soft-MoE) classifier for CIFAR-10.

Paper reference
- Soft Mixture of Experts (Soft-MoE) and related routing ideas from: "Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer" (Shazeer et al., 2017) and later Soft-MoE variants. This implementation follows the "soft" routing idea (dense probabilities over experts) with a simple load-balancing auxiliary loss.

Architecture
- Backbone: `torchvision.models.resnet18` (no custom attention/encoder/decoder blocks).
- MoE head: a Soft-MoE layer that mixes `N_EXPERTS` MLP experts using a dense softmax gate.
- Output: class logits for CIFAR-10.
- Auxiliary loss: KL divergence between the mean expert assignment and a uniform distribution, encouraging balanced expert usage.

File layout
- `train.py`: data module, Soft-MoE layer, model, Lightning training loop.

How to run
1) From the repo root:

```bash
python Soft-MOE/train.py
```

Notes
- CIFAR-10 is downloaded automatically into `/tmp` by default.
- You can tweak `N_EXPERTS`, `hidden_dim`, and `aux_loss_coef` inside `train.py`.
