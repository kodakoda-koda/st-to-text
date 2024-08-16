import argparse
import logging

import tokenizers
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from src.dataset.dataset import CustomDataset
from src.model.model import Model


class Exp_base:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        self.writer = SummaryWriter(log_dir=args.log_dir + f"{args.job_id}")

        self.model = self._build_model()
        self.tokenizer = self._bulid_tokenizer()

    def _build_model(self):
        model = Model(
            self.args.n_layers,
            self.args.d_model,
            self.args.n_heads,
            self.args.d_ff,
            self.args.dropout,
            self.args.n_locations,
            self.args.lm_name,
        )
        model = model.to(self.device).to(self.dtype)
        return model

    def _bulid_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.lm_name, legacy=False)
        newline_token = tokenizers.AddedToken(content="\n", normalized=False)
        tokenizer.add_tokens(list([newline_token]))
        return tokenizer

    def _get_dataloader(self, train_flag: bool):
        dataset = CustomDataset(self.args, self.tokenizer, train_flag)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=train_flag)
        return dataloader
