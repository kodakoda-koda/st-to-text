import json

import numpy as np
import torch
from transformers import AutoTokenizer

from src.dataset.dataset import CustomDataset


class TestDataset:
    st_maps = np.load("./data/st_maps.npy")
    with open("./data/labels.json") as f:
        labels = json.load(f)
    tokenizer = AutoTokenizer.from_pretrained("t5-base")
    max_length = 128

    train_dataset = CustomDataset(st_maps, labels, tokenizer, max_length)

    def test_len(self):
        assert len(self.train_dataset) == 500 * 0.8

    def test_getitem(self):
        assert set(self.train_dataset[0].keys()) == {
            "st_maps",
            "decoder_input_ids",
            "decoder_attention_mask",
            "inst_input_ids",
        }
        assert self.train_dataset[0]["st_maps"].shape == torch.Size([30, 100])
        assert self.train_dataset[0]["decoder_input_ids"].shape == torch.Size([128])
        assert self.train_dataset[0]["decoder_attention_mask"].shape == torch.Size([128])
        assert self.train_dataset[0]["inst_input_ids"].shape == torch.Size([16])
