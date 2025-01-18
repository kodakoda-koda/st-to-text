import argparse
import json
import logging
import os

import numpy as np
import torch
import transformers
from torch.utils.data import Dataset

from src.dataset.create_data import create_data


class CustomDataset(Dataset):
    def __init__(
        self,
        args: argparse.Namespace,
        logger: logging.Logger,
        tokenizer: transformers.PreTrainedTokenizer,
        flag: str = "train",
    ):
        self.args = args
        self.logger = logger
        self.tokenizer = tokenizer
        type_dict = {"train": 0, "val": 1, "test": 2}
        self.type = type_dict[flag]
        self.__load_data__()

    def __len__(self) -> int:
        return len(self.st_maps)

    def __getitem__(self, idx: int) -> dict:
        return {
            "st_maps": self.st_maps[idx],
            "encoder_input_ids": self.encoder_input_ids,
            "decoder_input_ids": self.decoder_input_ids[idx],
            "decoder_attention_mask": self.decoder_attention_mask[idx],
        }

    def __load_data__(self) -> None:
        if not os.path.exists(self.args.data_dir + "data.json"):
            self.logger.info("Creating data...")
            create_data(
                self.args.time_range,
                self.args.max_fluc_range,
                self.args.n_data,
                self.args.map_size,
                self.args.data_dir,
            )

        data = json.load(open(self.args.data_dir + "data.json", "r"))
        st_maps = torch.tensor(data["st_maps"])
        st_maps = st_maps.reshape(st_maps.shape[0], st_maps.shape[1], -1)
        labels = data["labels"]

        border_left_list = [0, int(0.8 * len(st_maps)), int(0.9 * len(st_maps))]
        border_right_list = [int(0.8 * len(st_maps)), int(0.9 * len(st_maps)), len(st_maps)]
        border_left = border_left_list[self.type]
        border_right = border_right_list[self.type]

        self.st_maps = st_maps[border_left:border_right]
        labels = labels[border_left:border_right]

        labels = ["<pad>" + label for label in labels]
        tokenized_labels = self.tokenizer.batch_encode_plus(
            labels,
            max_length=self.args.decoder_max_length,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.decoder_input_ids = tokenized_labels.input_ids
        self.decoder_attention_mask = tokenized_labels.attention_mask

        height = int(self.args.n_locations**0.5)
        coords = [f"[{i}, {j}]" for i in range(height) for j in range(height)]
        tokenized_coords = self.tokenizer.batch_encode_plus(
            coords,
            padding=True,
            truncation=True,
            return_tensors="pt",
        )
        self.encoder_input_ids = tokenized_coords.input_ids
