import torch
import torch.nn as nn
from torch import FloatTensor
from torch.nn import TransformerEncoderLayer
from transformers.modeling_outputs import BaseModelOutput

from src.model.embed import SpatialEmbedding, TemporalEmbedding


class GTformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        time_steps: int,
        n_locations: int,
    ):
        super(GTformer, self).__init__()
        self.layers = nn.ModuleList(
            [GTformer_block(d_model, n_heads, d_ff, dropout, time_steps, n_locations) for _ in range(n_layers)]
        )

    def forward(self, st_maps: FloatTensor) -> FloatTensor:
        for layer in self.layers:
            st_maps = layer(st_maps)

        return st_maps


class GTformer_block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, time_steps: int, n_locations: int):
        super(GTformer_block, self).__init__()
        self.t_emb = TemporalEmbedding(n_locations, time_steps, d_model)
        self.s_emb = SpatialEmbedding(time_steps, d_model)

        self.t_transformer = TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )
        self.s_transformer = TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=d_ff, dropout=dropout, batch_first=True
        )

        self.t_out = nn.Linear(d_model, n_locations)
        self.s_out = nn.Linear(d_model, time_steps)
        self.layer_norm = nn.LayerNorm(time_steps)

    def forward(self, st_maps: FloatTensor) -> FloatTensor:
        t_maps = self.t_emb(st_maps)
        s_maps = self.s_emb(st_maps.permute(0, 2, 1))

        t_maps = self.t_transformer(t_maps)
        s_maps = self.s_transformer(s_maps)

        t_out = self.t_out(t_maps)
        s_out = self.s_out(s_maps)
        out = self.layer_norm(t_out.permute(0, 2, 1) + s_out)

        return out.permute(0, 2, 1)
