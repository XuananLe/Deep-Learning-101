import torch
import torch.nn as nn
import torch.nn.functional as F

class RawPositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len=5000):
        super(RawPositionalEncoding, self).__init__()
        self.embed_dim = embed_dim
        self.max_len = max_len

    def forward(self, x):
        position = torch.arange(0, x.size(1), dtype=torch.float, device=x.device).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2, dtype=torch.float, device=x.device) * (-torch.log(torch.tensor(10000.0)) / self.embed_dim))
        pe = torch.zeros_like(x, device=x.device)
        pe[:, :, 0::2] = torch.sin(position * div_term)
        pe[:, :, 1::2] = torch.cos(position * div_term)
        return x + pe

class RawMultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.1):
        super(RawMultiheadAttention, self).__init__()
        assert embed_dim % num_heads == 0, "Embedding dimension must be divisible by number of heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)

        self.out_proj = nn.Linear(embed_dim, embed_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, attn_mask=None):
        batch_size = query.size(0)

        Q = self.q_proj(query).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.k_proj(key).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.v_proj(value).view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / torch.sqrt(torch.tensor(self.head_dim, dtype=torch.float32))
        if attn_mask is not None:
            scores = scores.masked_fill(attn_mask == 0, float('-inf'))
        attn_weights = torch.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)

        attn_output = torch.matmul(attn_weights, V)

        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, -1, self.embed_dim)
        output = self.out_proj(attn_output)

        return output, attn_weights

class RawTransformerEncoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(RawTransformerEncoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': RawMultiheadAttention(embed_dim, num_heads, dropout=dropout),
                'linear1': nn.Linear(embed_dim, ff_dim),
                'dropout': nn.Dropout(dropout),
                'linear2': nn.Linear(ff_dim, embed_dim),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'dropout_ff': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])

    def forward(self, src, src_mask=None):
        for layer in self.layers:
            attn_output, _ = layer['self_attn'](src, src, src, attn_mask=src_mask)
            src = layer['norm1'](src + attn_output)

            ff_output = layer['linear2'](layer['dropout'](F.relu(layer['linear1'](src))))
            src = layer['norm2'](src + layer['dropout_ff'](ff_output))

        return src

class RawTransformerDecoder(nn.Module):
    def __init__(self, embed_dim, num_heads, ff_dim, num_layers, dropout=0.1):
        super(RawTransformerDecoder, self).__init__()
        self.layers = nn.ModuleList([
            nn.ModuleDict({
                'self_attn': RawMultiheadAttention(embed_dim, num_heads, dropout=dropout),
                'cross_attn': RawMultiheadAttention(embed_dim, num_heads, dropout=dropout),
                'linear1': nn.Linear(embed_dim, ff_dim),
                'dropout': nn.Dropout(dropout),
                'linear2': nn.Linear(ff_dim, embed_dim),
                'norm1': nn.LayerNorm(embed_dim),
                'norm2': nn.LayerNorm(embed_dim),
                'norm3': nn.LayerNorm(embed_dim),
                'dropout_ff': nn.Dropout(dropout)
            }) for _ in range(num_layers)
        ])

    def forward(self, tgt, memory, tgt_mask=None, memory_mask=None):
        for layer in self.layers:
            self_attn_output, _ = layer['self_attn'](tgt, tgt, tgt, attn_mask=tgt_mask)
            tgt = layer['norm1'](tgt + self_attn_output)

            cross_attn_output, _ = layer['cross_attn'](tgt, memory, memory, attn_mask=memory_mask)
            tgt = layer['norm2'](tgt + cross_attn_output)

            ff_output = layer['linear2'](layer['dropout'](F.relu(layer['linear1'](tgt))))
            tgt = layer['norm3'](tgt + layer['dropout_ff'](ff_output))

        return tgt

class TransformerEncoderDecoder(nn.Module):
    def __init__(self, input_dim, output_dim, embed_dim, num_heads, num_layers, ff_dim, dropout=0.1):
        super(TransformerEncoderDecoder, self).__init__()

        self.src_embedding = nn.Embedding(input_dim, embed_dim)
        self.tgt_embedding = nn.Embedding(output_dim, embed_dim)

        self.encoder = RawTransformerEncoder(embed_dim, num_heads, ff_dim, num_layers, dropout)
        self.decoder = RawTransformerDecoder(embed_dim, num_heads, ff_dim, num_layers, dropout)

        self.fc_out = nn.Linear(embed_dim, output_dim)

        self.positional_encoding = RawPositionalEncoding(embed_dim)

    def forward(self, src, tgt, src_mask=None, tgt_mask=None, memory_mask=None):
        src = self.src_embedding(src) * torch.sqrt(torch.tensor(src.size(-1), dtype=torch.float32))
        tgt = self.tgt_embedding(tgt) * torch.sqrt(torch.tensor(tgt.size(-1), dtype=torch.float32))

        src = self.positional_encoding(src)
        tgt = self.positional_encoding(tgt)

        src = src.transpose(0, 1)
        tgt = tgt.transpose(0, 1)

        memory = self.encoder(src, src_mask)

        output = self.decoder(tgt, memory, tgt_mask=tgt_mask, memory_mask=memory_mask)

        output = self.fc_out(output)

        output = output.transpose(0, 1)

        return output

if __name__ == "__main__":
    input_dim = 1000
    output_dim = 1000
    embed_dim = 512
    num_heads = 8
    num_layers = 6
    ff_dim = 2048

    model = TransformerEncoderDecoder(input_dim, output_dim, embed_dim, num_heads, num_layers, ff_dim)

    src = torch.randint(0, input_dim, (10, 32))
    tgt = torch.randint(0, output_dim, (20, 32))

    output = model(src, tgt)
    print(output.shape)
