import torch
import torch.nn as nn
from torch import FloatTensor
from transformers.modeling_outputs import BaseModelOutput

from src.model.embed import Embedding
from src.model.transformer import TransformerEncoderLayer


class GTformer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        n_locations: int,
        lm_d_model: int,
    ):
        super(GTformer, self).__init__()
        self.layers = nn.ModuleList(
            [GTformer_block(d_model, n_heads, d_ff, dropout, n_locations) for _ in range(n_layers)]
        )
        self.fn = nn.Linear(n_locations, lm_d_model)
        self.layer_norm = nn.LayerNorm(lm_d_model)

    def forward(self, st_maps: FloatTensor) -> BaseModelOutput:
        for layer in self.layers:
            st_maps = layer(st_maps)
        out = self.layer_norm(self.fn(st_maps))

        return BaseModelOutput(last_hidden_state=out, hidden_states=None, attentions=None)


class GTformer_block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, n_locations: int):
        super(GTformer_block, self).__init__()
        self.t_emb = Embedding(n_locations, d_model)
        self.s_emb = Embedding(32, d_model)

        self.t_transformer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.s_transformer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)

        self.t_out = nn.Linear(d_model, n_locations)
        self.s_out = nn.Linear(d_model, 32)
        self.layer_norm = nn.LayerNorm(32)

    def forward(self, st_maps: FloatTensor) -> FloatTensor:
        t_maps = self.t_emb(st_maps)
        s_maps = self.s_emb(st_maps.permute(0, 2, 1))

        t_maps = self.t_transformer(t_maps)
        s_maps = self.s_transformer(s_maps)

        t_out = self.t_out(t_maps)
        s_out = self.s_out(s_maps)
        out = self.layer_norm(t_out.permute(0, 2, 1) + s_out)

        return out.permute(0, 2, 1)
