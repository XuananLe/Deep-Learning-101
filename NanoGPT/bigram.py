import torch
import torch.nn as nn
from torch.nn import functional as F
import sys

# Ensure we use CUDA if it's available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.cuda.empty_cache()

# Read the input text
with open('./input.txt', mode='r') as f:
    text = f.read()

# Prepare the vocab and encoding/decoding maps
vocab = sorted(set(text))
vocab_size = len(vocab)
stoi = {ch: i for i, ch in enumerate(vocab)}
itos = {i: ch for i, ch in enumerate(vocab)}
encode = lambda s: [stoi[c] for c in s]
decode = lambda s: ''.join([itos[c] for c in s])
dataset = torch.tensor(encode(text), dtype=torch.long).to(device)  # Move dataset to device
split = int(0.9 * len(dataset))
train_data = dataset[:split]
test_data = dataset[split:]

block_size = 8
batch_size = 4
max_iter_size = 10000
log_steps = 1000
eval_iters = 200
eval_interval = 200


def get_batch(split):
    data = train_data if split == "train" else test_data
    ix = torch.randint(len(data) - block_size, (batch_size, ))
    x = torch.stack([data[i: i + block_size] for i in ix]).to(device)
    y = torch.stack([data[i + 1:i + block_size + 1] for i in ix]).to(device)
    return x, y


@torch.no_grad()
def estimate_loss():
    out = {}
    m.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters).to(device)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            _, losses[k] = m(X, Y)
        out[split] = losses.mean()
    m.train()
    return out


class BigramLanguageModel(nn.Module):
    def __init__(self, vocab_size):
        super().__init__()
        self.emb = nn.Embedding(vocab_size, vocab_size).to(device)
        
    def forward(self, idx, targets=None):
        logits = self.emb(idx)
        if targets is None:
            return logits, None
        else:
            B, T, C = logits.shape
            logits = logits.view(B * T, C)
            targets = targets.view(B * T)
            loss = F.cross_entropy(logits, targets)
            return logits, loss
    
    def generate(self, idx, max_tokens):
        for _ in range(max_tokens):
            logits, _ = self.forward(idx)
            logits = logits[:, -1, :]
            sft = F.softmax(logits, dim=1)
            idx_next = torch.multinomial(sft, 1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx


m = BigramLanguageModel(vocab_size).to(device)

optimizer = torch.optim.AdamW(m.parameters(), lr=1e-3)

xb, yb = get_batch("train")

for steps in range(max_iter_size):
    if steps % eval_interval == 0:
        losses = estimate_loss()
        print(f"At step {steps}, the loss at the train is {losses['train']}, the loss at the test is {losses['val']}")

    xb, yb = get_batch('train')
    logits, loss = m(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

print(loss.item())
print(decode(m.generate(idx=torch.ones((1, 1), dtype=torch.long).to(device), max_tokens=500)[0].tolist()))
