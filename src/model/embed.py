import math

import torch
import torch.nn as nn
from torch import Tensor


class Embedding(nn.Module):
    def __init__(self, n_channels: int, d_model: int, n_coord: int = None):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(n_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        if n_coord is not None:
            self.coord_embedding = LocationEmbedding(n_coord, d_model)

    def forward(self, x: Tensor, x_coord: Tensor = None) -> Tensor:
        """
        B, T, C, N: batch size, time steps, number of channels, number of time features
        x: (B, T, C)
        """
        x = self.embedding(x)
        x = self.positional_encoding(x)
        if x_coord is not None:
            x_coord = self.coord_embedding(x_coord)
            x += x_coord
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
        return x + self.positional_encoding[: x.size(1)].to(x.dtype)


class LocationEmbedding(nn.Module):
    def __init__(self, n_coord: int, d_model: int):
        super(LocationEmbedding, self).__init__()
        self.x_embedding = nn.Embedding(int(math.sqrt(n_coord)), d_model)
        self.y_embedding = nn.Embedding(int(math.sqrt(n_coord)), d_model)

    def forward(self, x_coord: Tensor) -> Tensor:
        x_x = x_coord[:, :, 0].long()
        x_y = x_coord[:, :, 1].long()

        x_x = self.x_embedding(x_x)
        x_y = self.y_embedding(x_y)

        return x_x + x_y
