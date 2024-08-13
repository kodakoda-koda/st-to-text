import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset.dataset import CustomDataset
from src.model.model import Model


class TestModel:
    max_length = 128
    n_layers = 12
    d_model = 768
    n_heads = 12
    d_ff = 3072
    dropout = 0.1
    n_locations = 100
    lm_name = "t5-base"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    st_maps = np.load("./data/st_maps.npy")
    with open("./data/labels.json") as f:
        labels = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained(lm_name)

    train_dataset = CustomDataset(st_maps, labels, tokenizer, max_length)
    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)

    batch = next(iter(train_loader))
    st_maps = batch["st_maps"].to(device)
    inst_input_ids = batch["inst_input_ids"].to(device)
    decoder_input_ids = batch["decoder_input_ids"][:, :-1].to(device)
    decoder_attention_mask = batch["decoder_attention_mask"][:, :-1].to(device)
    labels = batch["decoder_input_ids"][:, 1:].to(device)

    model = Model(
        n_layers,
        d_model,
        n_heads,
        d_ff,
        dropout,
        n_locations,
        lm_name,
    ).to(device)

    def test_forward(self):
        outputs = self.model(
            st_maps=self.st_maps,
            inst_input_ids=self.inst_input_ids,
            decoder_input_ids=self.decoder_input_ids,
            decoder_attention_mask=self.decoder_attention_mask,
            labels=self.labels,
        )
        assert outputs.loss is not None
        assert outputs.logits is not None
        assert outputs.loss.item() >= 0.0
        assert outputs.logits.shape == (16, self.max_length - 1, self.model.t5.config.vocab_size)
        assert outputs.loss.requires_grad
