import math
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, n_channels: int, d_model: int):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(n_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model)

    def forward(self, x: Tensor) -> Tensor:
        x = self.embedding(x)
        x += self.positional_encoding(x)
        return x


class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, max_len: int = 5000):
        super(PositionalEncoding, self).__init__()
        self.d_model = d_model
        self.max_len = max_len
        self.register_buffer("positional_encoding", self._get_positional_encoding(max_len, d_model))

    def _get_positional_encoding(self, max_len: int, d_model: int) -> Tensor:
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * -(torch.log(torch.tensor(10000.0)) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        return pe

    def forward(self, x: Tensor) -> Tensor:
        return self.positional_encoding[: x.size(1)].to(x.dtype)
