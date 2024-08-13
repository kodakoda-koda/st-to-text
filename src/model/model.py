import torch
import torch.nn as nn
from peft import LoraConfig, TaskType, get_peft_model
from torch import Tensor
from transformers import T5ForConditionalGeneration
from transformers.modeling_outputs import Seq2SeqLMOutput

from src.data2desc.encoder import GTFormer


class Model(nn.Module):
    def __init__(
        self,
        n_layers: int,
        d_model: int,
        n_heads: int,
        d_ff: int,
        dropout: float,
        n_locations: int,
        lm_name: str,
    ):
        super(Model, self).__init__()
        self.gtformer = GTFormer(
            n_layers,
            d_model,
            n_heads,
            d_ff,
            dropout,
            n_locations,
        )
        self.t5 = T5ForConditionalGeneration.from_pretrained(lm_name)
        self.fn_emb = nn.Linear(n_locations, self.t5.config.d_model)

        # PEFT
        lora_config = LoraConfig(
            r=8,
            lora_alpha=32,
            target_modules=["q", "k", "v", "o", "wi", "wo"],
            lora_dropout=0.05,
            bias="none",
            fan_in_fan_out=False,
            task_type=TaskType.SEQ_2_SEQ_LM,
        )
        self.t5 = get_peft_model(self.t5, lora_config)

    def forward(
        self,
        st_maps: Tensor,
        inst_input_ids: Tensor,
        decoder_input_ids: Tensor = None,
        decoder_attention_mask: Tensor = None,
        labels: Tensor = None,
    ) -> Seq2SeqLMOutput:

        pred = self.gtformer(st_maps)
        pred_emb = self.fn_emb(pred)

        encoder_outputs = self.t5.encoder(input_ids=inst_input_ids)

        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states[:, :-1], pred_emb, encoder_hidden_states[:, -1:]], dim=1
        )
        encoder_outputs.last_hidden_state = encoder_hidden_states

        outputs = self.t5(
            encoder_outputs=encoder_outputs,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            labels=labels,
        )

        return outputs

    def generate(
        self,
        demands: Tensor,
        time_features: Tensor,
        inst_input_ids: Tensor,
        **kwargs,
    ) -> Tensor:

        demands_emb = self.gtformer(demands, time_features)
        demands_emb = self.fn_emb(demands_emb)

        encoder_outputs = self.t5.encoder(input_ids=inst_input_ids)

        encoder_hidden_states = encoder_outputs.last_hidden_state
        encoder_hidden_states = torch.cat(
            [encoder_hidden_states[:, :-1], demands_emb, encoder_hidden_states[:, -1:]], dim=1
        )
        encoder_outputs.last_hidden_state = encoder_hidden_states

        outputs = self.t5.generate(
            encoder_outputs=encoder_outputs,
            **kwargs,
        )

        return outputs
