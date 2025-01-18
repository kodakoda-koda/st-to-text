import torch
import torch.nn as nn
from torch import FloatTensor
from torch.nn import TransformerEncoderLayer
from transformers import T5Config, T5EncoderModel
from transformers.modeling_outputs import BaseModelOutput

from src.model.embed import SpatialEmbedding, TemporalEmbedding


class Encoder(nn.Module):
    def __init__(
        self,
        gtformer_config: dict,
        t5_config: T5Config,
    ):
        super(Encoder, self).__init__()

        self.t5 = T5EncoderModel(t5_config)
        self.gtformer = GTformer(**gtformer_config)
        self.gtformer_fn = nn.Linear(gtformer_config["time_steps"], self.t5.config.d_model)
        self.layer_norm = nn.LayerNorm(self.t5.config.d_model)

    def forward(
        self,
        st_maps: FloatTensor,
        encoder_input_ids: FloatTensor,
    ) -> BaseModelOutput:

        t5enc_output = self.t5(encoder_input_ids)
        t5enc_output = t5enc_output.last_hidden_state[:, 0, :]
        t5enc_output = t5enc_output.view(encoder_input_ids.size(0), encoder_input_ids.size(1), -1)

        gtformer_output = self.gtformer(st_maps)
        gtformer_output = self.gtformer_fn(gtformer_output.permute(0, 2, 1))

        encoder_outputs = t5enc_output + gtformer_output
        encoder_outputs = self.layer_norm(encoder_outputs)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)

        return encoder_outputs


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
    def __init__(
        self,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        time_steps: int,
        n_locations: int,
    ):
        super(GTformer_block, self).__init__()
        self.t_emb = TemporalEmbedding(n_locations, time_steps, d_model)
        self.s_emb = SpatialEmbedding(time_steps, d_model)

        self.t_transformer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
        )
        self.s_transformer = TransformerEncoderLayer(
            d_model=d_model,
            nhead=n_heads,
            dim_feedforward=d_ff,
            dropout=dropout,
            batch_first=True,
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
