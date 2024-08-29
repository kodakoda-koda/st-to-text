import argparse
import logging

import torch
import torch.nn as nn
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer

from src.dataset.dataset import CustomDataset
from src.model.model import Model
from src.utils.exp_utils import CustomLoss


class Exp_base:
    def __init__(self, args: argparse.Namespace, logger: logging.Logger):
        self.args = args
        self.logger = logger
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
        self.writer = SummaryWriter(log_dir=args.log_dir + f"{args.job_id}")

        self.model = self._build_model()
        self.tokenizer = self._bulid_tokenizer()
        self.loss_func = self._get_weighted_loss_func()

    def _build_model(self):
        model = Model(
            self.args.n_layers,
            self.args.d_model,
            self.args.n_heads,
            self.args.d_ff,
            self.args.dropout,
            self.args.n_locations,
        )
        model = model.to(self.device).to(self.dtype)
        return model

    def _bulid_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.lm_name, legacy=False)
        tokenizer.add_tokens(["7,"])
        return tokenizer

    def _get_dataloader(self, train_flag: bool):
        dataset = CustomDataset(self.args, self.tokenizer, train_flag)
        batch_size = self.args.train_batch_size if train_flag else self.args.eval_batch_size
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=train_flag)
        return dataloader

    def _get_weighted_loss_func(self):
        loss_weight = torch.ones(self.model.vocab_size)
        loss_weight[[6313, 6734, 10823, 993, 2667]] = 5.0
        # loss_weight[[4347, 4482, 6355, 8525, 11116, 11071, 32100, 11864]] = 5.0
        # loss_weight[[209, 204, 220, 314, 305, 431, 489, 505]] = 5.0
        loss_weight = loss_weight.to(self.device).to(self.dtype)

        loss_func = CustomLoss(loss_weight, self.args.train_batch_size, self.device)
        return loss_func
