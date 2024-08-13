import torch
import torch.nn as nn
from torch import Tensor
from transformers.modeling_outputs import BaseModelOutput

from src.data2desc.embed import Embedding
from src.data2desc.transformer import TransformerEncoderLayer


class STformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        n_locations: int,
    ):
        super(STformer, self).__init__()
        self.layers = nn.ModuleList(
            [STformer_block(d_model, n_heads, d_ff, dropout, n_locations, 7) for _ in range(n_layers)]
        )

    def forward(self, demands: Tensor, time_features: Tensor) -> BaseModelOutput:
        """
        B, T, L, M, N: batch size, time steps, length of text, number of locations, number of time features
        demands: (B, T, M)
        time_features: (B, T, N)
        inst_input_ids: (B, L)
        """
        B, T, _ = demands.size()

        for layer in self.layers:
            demands = layer(demands, time_features)

        outputs = demands[:, -24:]  # (B, 24, M)

        return outputs


class STformer_block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, n_locations: int, hist_days: int):
        super(STformer_block, self).__init__()
        self.t_emb = Embedding(n_locations, d_model, time=True)
        self.s_emb = Embedding((hist_days + 1) * 24, d_model, time=False)

        self.t_transformer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.s_transformer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)

        self.t_out = nn.Linear(d_model, n_locations)
        self.s_out = nn.Linear(d_model, (hist_days + 1) * 24)

    def forward(self, demands: Tensor, time_features: Tensor) -> Tensor:
        """
        B, T, L, M, N: batch size, time steps, length of text, number of locations, number of time features
        demands: (B, T, M)
        time_features: (B, T, N)
        """
        B, T, _ = demands.size()
        t_demands = self.t_emb(demands, time_features)
        s_demands = self.s_emb(demands.permute(0, 2, 1))

        t_mask = torch.ones(B, 1, T, T)
        t_mask = torch.triu(t_mask, diagonal=1).bool()
        t_mask = t_mask.to(t_demands.device)

        t_demands = self.t_transformer(t_demands, t_mask)
        s_demands = self.s_transformer(s_demands)

        t_out = self.t_out(t_demands)
        s_out = self.s_out(s_demands)
        out = t_out + s_out.permute(0, 2, 1)

        return out
