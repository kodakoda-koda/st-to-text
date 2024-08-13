import torch
import torch.nn as nn
from torch import Tensor


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
        """
        B, T, D: batch size, time steps, d_model
        x: (B, T, D)
        """
        return x + self.positional_encoding[: x.size(1)].to(x.dtype)


class Embedding(nn.Module):
    def __init__(self, n_channels: int, d_model: int, time: bool = True):
        super(Embedding, self).__init__()
        self.d_model = d_model
        self.embedding = nn.Linear(n_channels, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        if time:
            self.time_embedding = TimeFeatureEmbedding(d_model)

    def forward(self, x: Tensor, x_time: Tensor = None) -> Tensor:
        """
        B, T, C, N: batch size, time steps, number of channels, number of time features
        x: (B, T, C)
        x_time: (B, T, N)
        """
        x = self.embedding(x)  # (B, T, D)
        x = self.positional_encoding(x)
        if x_time is not None:
            x_time = self.time_embedding(x_time).to(x.dtype)  # (B, T, D)
            x = x + x_time
        return x  # (B, T, D)


class TimeFeatureEmbedding(nn.Module):
    def __init__(self, d_model: int):
        super(TimeFeatureEmbedding, self).__init__()
        self.d_model = d_model
        self.month_embedding = nn.Embedding(13, d_model)
        self.day_embedding = nn.Embedding(32, d_model)
        self.weekday_embedding = nn.Embedding(7, d_model)
        self.holiday_embedding = nn.Embedding(2, d_model)
        self.hour_embedding = nn.Embedding(24, d_model)
        self.event_embedding = nn.Embedding(2, d_model)
        self.rain_embedding = nn.Embedding(2, d_model)

    def forward(self, x_time: Tensor) -> Tensor:
        """
        B, T, N: batch size, time steps, number of time features
        x_time: (B, T, N)
        """
        month = x_time[:, :, 0].long()
        day = x_time[:, :, 1].long()
        weekday = x_time[:, :, 2].long()
        holiday = x_time[:, :, 3].long()
        hour = x_time[:, :, 4].long()
        is_event = x_time[:, :, 5].long()
        is_rain = x_time[:, :, 6].long()

        month = self.month_embedding(month)
        day = self.day_embedding(day)
        weekday = self.weekday_embedding(weekday)
        holiday = self.holiday_embedding(holiday)
        hour = self.hour_embedding(hour)
        is_event = self.event_embedding(is_event)
        is_rain = self.rain_embedding(is_rain)

        return month + day + weekday + holiday + hour + is_event + is_rain  # (B, T, D)
