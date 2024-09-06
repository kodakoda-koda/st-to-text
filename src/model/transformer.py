import torch
import torch.nn as nn
from torch import Tensor
from typing import Optional


class SelfAttention(nn.Module):
    def __init__(self, d_model: int, n_heads: int):
        super(SelfAttention, self).__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        head_dim = self.d_model // self.n_heads
        self.scaler = self.n_heads ** (1 / 2)

        self.v_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.k_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.q_proj = nn.Linear(head_dim, head_dim, bias=False)
        self.fc_out = nn.Linear(n_heads * head_dim, d_model)

    def forward(self, query: Tensor, key: Tensor, value: Tensor, mask:Optional[Tensor] = None) -> Tensor:
        """
        B, L, H, D: batch size, sequence length, n_heads, d_model
        query: (B, L, D)
        key: (B, L, D)
        value: (B, L, D)
        """
        B, L, _ = query.shape

        queries = query.view(B, L, self.n_heads, -1)
        keys = key.view(B, L, self.n_heads, -1)
        values = value.view(B, L, self.n_heads, -1)

        queries = self.q_proj(queries)
        keys = self.k_proj(keys)
        values = self.v_proj(values)

        attention_weight = torch.einsum("bqhd,bkhd->bhqk", [queries, keys]) / self.scaler  # (B, H, L, L)
        if mask is not None:
            mask = mask.repeat(1, self.n_heads, 1, 1)
            attention_weight = attention_weight.masked_fill(mask == 1, float("-inf"))
        attention_weight = torch.softmax(attention_weight, dim=3)

        out = torch.einsum("bhqk,bkhd->bqhd", [attention_weight, values])  # (B, H, L, D)
        out = self.fc_out(out.reshape(B, L, -1))  # (B, L, D)

        return out


class FeedForward(nn.Module):
    def __init__(self, d_model: int, d_ff: int, dropout: float):
        super(FeedForward, self).__init__()
        self.fc1 = nn.Linear(d_model, d_ff)
        self.fc2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor) -> Tensor:
        """
        B, L, D: batch size, sequence length, d_model
        """
        x = self.dropout(torch.relu(self.fc1(x)))
        x = self.fc2(x)
        return x  # (B, L, D)


class TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float):
        super(TransformerEncoderLayer, self).__init__()
        self.self_attention = SelfAttention(d_model, n_heads)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        self.layer_norm1 = nn.LayerNorm(d_model)
        self.layer_norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: Tensor, mask: Optional[Tensor] = None) -> Tensor:
        """
        B, L, D: batch size, sequence length, d_model
        x: (B, L, D)
        mask: (B, L, L)
        """
        attention_out = self.self_attention(x, x, x, mask)
        x = self.layer_norm1(x + self.dropout(attention_out))

        feed_forward_out = self.feed_forward(x)
        x = self.layer_norm2(x + self.dropout(feed_forward_out))

        return x  # (B, L, D)
