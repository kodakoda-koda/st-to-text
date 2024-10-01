import argparse

import torch
from transformers import AutoTokenizer

from src.dataset.dataset import CustomDataset


class TestDataset:
    def __init__(self) -> None:
        args = {
            "time_range": 30,
            "max_fluc_range": 10,
            "n_data": 500,
            "n_locations": 100,
            "map_size": 10,
            "data_dir": "./data/",
            "decoder_max_length": 64,
        }
        self.args = argparse.Namespace(**args)
        self.tokenizer = AutoTokenizer.from_pretrained("t5-base")
        self.train_dataset = CustomDataset(self.args, self.tokenizer, train_flag=True)

    def test_len(self) -> None:
        assert len(self.train_dataset) == self.args.n_data * 0.8

    def test_getitem(self) -> None:
        assert set(self.train_dataset[0].keys()) == {
            "st_maps",
            "encoder_input_ids",
            "decoder_input_ids",
            "decoder_attention_mask",
            "coords_labels",
        }
        assert self.train_dataset[0]["st_maps"].shape == torch.Size([self.args.time_range, self.args.map_size**2])
        assert self.train_dataset[0]["decoder_input_ids"].shape == torch.Size([self.args.decoder_max_length])
        assert self.train_dataset[0]["decoder_attention_mask"].shape == torch.Size([self.args.decoder_max_length])
