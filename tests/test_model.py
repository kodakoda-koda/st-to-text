import argparse
import json

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from src.dataset.dataset import CustomDataset
from src.model.model import Model


class TestModel:
    def __init__(self):
        args = {
            "max_length": 128,
            "n_layers": 12,
            "d_model": 768,
            "n_heads": 12,
            "d_ff": 3072,
            "dropout": 0.1,
            "time_range": 30,
            "max_fluc_range": 10,
            "n_data": 500,
            "n_locations": 100,
            "map_size": 10,
            "data_dir": "./data/",
            "decoder_max_length": 64,
        }
        self.args = argparse.Namespace(**args)
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16

        tokenizer = AutoTokenizer.from_pretrained("t5-base")

        train_dataset = CustomDataset(self.args, tokenizer, True)
        self.train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)

        self.model = (
            Model(
                self.args.n_layers,
                self.args.d_model,
                self.args.n_heads,
                self.args.d_ff,
                self.args.dropout,
                self.args.n_locations,
            )
            .to(self.device)
            .to(self.dtype)
        )

    def test_forward(self):
        for batch in self.train_loader:
            st_maps = batch["st_maps"].to(self.device).to(self.dtype)
            inst_input_ids = batch["inst_input_ids"].to(self.device)
            decoder_input_ids = batch["decoder_input_ids"].to(self.device)
            decoder_attention_mask = batch["decoder_attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            outputs = self.model(
                st_maps=st_maps,
                inst_input_ids=inst_input_ids,
                decoder_input_ids=decoder_input_ids,
                decoder_attention_mask=decoder_attention_mask,
                labels=labels,
            )
            assert outputs.loss is not None
            assert outputs.logits is not None
            assert outputs.loss.item() >= 0.0
            assert outputs.logits.shape == (16, self.args.decoder_max_length - 1, self.model.t5.config.vocab_size)
            assert outputs.loss.requires_grad
