import torch.nn as nn
from torch import Tensor
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.model.encoder import GTformer


class Model(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        n_locations: int,
    ):
        super(Model, self).__init__()
        self.gtformer = GTformer(
            n_layers,
            d_model,
            n_heads,
            d_ff,
            dropout,
            n_locations,
        )
        t5_config = T5Config()
        t5_config.decoder_start_token_id = 0
        self.t5 = T5ForConditionalGeneration(t5_config)
        self.fn_emb = nn.Linear(n_locations, self.t5.config.d_model)

    def forward(
        self,
        st_maps: Tensor,
        coords: Tensor,
        decoder_input_ids: Tensor = None,
        decoder_attention_mask: Tensor = None,
        labels: Tensor = None,
    ) -> Seq2SeqLMOutput:

        encoder_outputs = self.gtformer(st_maps, coords)
        encoder_outputs.last_hidden_state = self.fn_emb(encoder_outputs.last_hidden_state)

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        return outputs

    def generate(
        self,
        st_maps: Tensor,
        coords: Tensor,
        **kwargs,
    ) -> Tensor:

        encoder_outputs = self.gtformer(st_maps, coords)
        encoder_outputs.last_hidden_state = self.fn_emb(encoder_outputs.last_hidden_state)

        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

        return outputs
