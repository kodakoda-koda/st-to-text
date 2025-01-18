from typing import Any, Optional

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from src.model.encoder import GTformer


class Model(nn.Module):
    def __init__(
        self,
        gtformer_config: dict,
        t5_config: T5Config,
    ):
        super(Model, self).__init__()

        self.t5 = T5ForConditionalGeneration(t5_config)
        self.gtformer = GTformer(**gtformer_config)
        self.gtformer_fn = nn.Linear(gtformer_config["time_steps"], self.t5.config.d_model)
        self.layer_norm = nn.LayerNorm(self.t5.config.d_model)

    def forward(
        self,
        st_maps: FloatTensor,
        encoder_input_ids: LongTensor,
        decoder_input_ids: LongTensor,
        decoder_attention_mask: LongTensor,
        labels: Optional[LongTensor] = None,
    ) -> Seq2SeqLMOutput:

        t5enc_output = self.t5.encoder(encoder_input_ids.view(-1, encoder_input_ids.size(-1)))
        t5enc_output = t5enc_output.last_hidden_state[:, 0, :]
        t5enc_output = t5enc_output.view(encoder_input_ids.size(0), encoder_input_ids.size(1), -1)

        gtformer_output = self.gtformer(st_maps)
        gtformer_output = self.gtformer_fn(gtformer_output.permute(0, 2, 1))

        encoder_outputs = t5enc_output + gtformer_output
        encoder_outputs = self.layer_norm(encoder_outputs)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        return outputs

    def generate(
        self,
        st_maps: FloatTensor,
        encoder_input_ids: LongTensor,
        **kwargs,
    ) -> Any:

        t5enc_output = self.t5.encoder(encoder_input_ids.view(-1, encoder_input_ids.size(-1)))
        t5enc_output = t5enc_output.last_hidden_state[:, 0, :]
        t5enc_output = t5enc_output.view(encoder_input_ids.size(0), encoder_input_ids.size(1), -1)

        gtformer_output = self.gtformer(st_maps)

        encoder_outputs = t5enc_output + gtformer_output
        encoder_outputs = self.layer_norm(encoder_outputs)
        encoder_outputs = BaseModelOutput(last_hidden_state=encoder_outputs)

        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

        return outputs
