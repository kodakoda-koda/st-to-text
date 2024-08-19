import torch
import torch.nn as nn
from torch import Tensor
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
    ):
        super(GTformer, self).__init__()
        self.layers = nn.ModuleList(
            [GTformer_block(d_model, n_heads, d_ff, dropout, n_locations) for _ in range(n_layers)]
        )

    def forward(self, st_maps: Tensor, coords: Tensor) -> BaseModelOutput:
        """
        B, T, L, M, N: batch size, time steps, length of text, number of locations, number of time features
        st_maps: (B, T, M)
        coords: (B, M, 2)
        """
        for layer in self.layers:
            st_maps = layer(st_maps, coords)

        return BaseModelOutput(last_hidden_state=st_maps, hidden_states=None, attentions=None)


class GTformer_block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int, dropout: float, n_locations: int):
        super(GTformer_block, self).__init__()
        self.t_emb = Embedding(n_locations, d_model)
        self.s_emb = Embedding(30, d_model, n_coord=n_locations)

        self.t_transformer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)
        self.s_transformer = TransformerEncoderLayer(d_model, n_heads, d_ff, dropout)

        self.t_out = nn.Linear(d_model, n_locations)
        self.s_out = nn.Linear(d_model, 30)

    def forward(self, st_maps: Tensor, coords: Tensor) -> Tensor:
        # B, T, _ = st_maps.size()
        t_maps = self.t_emb(st_maps)
        s_maps = self.s_emb(st_maps.permute(0, 2, 1), coords)

        # t_mask = torch.ones(B, 1, T, T)
        # t_mask = torch.triu(t_mask, diagonal=1).bool()
        # t_mask = t_mask.to(t_maps.device)

        t_maps = self.t_transformer(t_maps)  # , t_mask)
        s_maps = self.s_transformer(s_maps)

        t_out = self.t_out(t_maps)
        s_out = self.s_out(s_maps)
        out = t_out + s_out.permute(0, 2, 1)

        return out
