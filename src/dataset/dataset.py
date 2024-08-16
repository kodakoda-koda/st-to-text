import argparse
import json
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
        tokenizer: transformers.PreTrainedTokenizer,
        train_flag=True,
    ):
        self.args = args
        self.tokenizer = tokenizer
        self.train_flag = train_flag
        self.__load_data__()

    def __len__(self) -> int:
        return len(self.st_maps)

    def __getitem__(self, idx: int) -> dict:
        return {
            "st_maps": self.st_maps[idx],
            "inst_input_ids": self.inst_input_ids[idx],
            "decoder_input_ids": self.decoder_input_ids[idx],
            "decoder_attention_mask": self.decoder_attention_mask[idx],
        }

    def __load_data__(self):
        if not os.path.exists(self.args.data_dir + "labels.json"):
            create_data(
                self.args.time_range, self.args.max_fluc_range, self.args.n_data, self.args.map_size, self.args.data_dir
            )

        st_maps = np.load(self.args.data_dir + "st_maps.npy")
        st_maps = st_maps.reshape(-1, self.args.time_range, self.args.map_size**2)
        with open(self.args.data_dir + "labels.json", "r") as f:
            labels = json.load(f)
        inst = ["Generate a caption for the given spatial-temporal data" for _ in range(len(st_maps))]

        if self.train_flag:
            self.st_maps = st_maps[: int(0.8 * len(st_maps))]
            labels = labels[: int(0.8 * len(labels))]
        else:
            self.st_maps = st_maps[int(0.8 * len(st_maps)) :]
            labels = labels[int(0.8 * len(labels)) :]

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

        tokenized_inst = self.tokenizer.batch_encode_plus(
            inst,
            max_length=16,
            padding="max_length",
            truncation=True,
            return_tensors="pt",
        )
        self.inst_input_ids = tokenized_inst.input_ids
