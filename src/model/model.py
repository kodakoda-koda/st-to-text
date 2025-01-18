from typing import Any, Optional

import torch
import torch.nn as nn
from torch import FloatTensor, LongTensor
from transformers import T5Config, T5ForConditionalGeneration
from transformers.modeling_outputs import BaseModelOutput, Seq2SeqLMOutput

from src.model.encoder import Encoder


class Model(nn.Module):
    def __init__(
        self,
        gtformer_config: dict,
        t5_config: T5Config,
    ):
        super(Model, self).__init__()

        self.encoder = Encoder(gtformer_config, t5_config)
        self.t5 = T5ForConditionalGeneration(t5_config)

    def forward(
        self,
        st_maps: FloatTensor,
        encoder_input_ids: LongTensor,
        decoder_input_ids: LongTensor,
        decoder_attention_mask: LongTensor,
        labels: Optional[LongTensor] = None,
    ) -> Seq2SeqLMOutput:

        encoder_outputs = self.encoder(st_maps, encoder_input_ids)
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

        encoder_outputs = self.encoder(st_maps, encoder_input_ids)
        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

        return outputs
