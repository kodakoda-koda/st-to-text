import argparse
import logging

import torch
import torch.nn as nn
from tokenizers import AddedToken
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import AutoTokenizer, T5Config

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
        t5_config = T5Config()
        t5_config.decoder_start_token_id = 0
        t5_config.num_layers = 1
        t5_config.num_decoder_layers = 3
        t5_config.output_hidden_states = True

        gtformer_config = {
            "n_layers": self.args.n_layers,
            "d_model": self.args.d_model,
            "n_heads": self.args.n_heads,
            "d_ff": self.args.d_ff,
            "dropout": self.args.dropout,
            "time_steps": self.args.time_steps,
            "n_locations": self.args.n_locations,
        }

        model = Model(
            gtformer_config,
            t5_config,
        )
        model = model.to(self.device).to(self.dtype)
        return model

    def _bulid_tokenizer(self):
        tokenizer = AutoTokenizer.from_pretrained(self.args.lm_name, legacy=False)
        tokenizer.add_tokens(["7,", "0,", "9,"])
        tokenizer.add_tokens(AddedToken("\n", normalized=False))
        return tokenizer

    def _get_dataloader(self, flag: str):
        dataset = CustomDataset(self.args, self.logger, self.tokenizer, flag)
        train_flag = flag == "train"
        batch_size = self.args.train_batch_size if train_flag else self.args.eval_batch_size
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=train_flag)
        return dataloader

    def _get_weighted_loss_func(self):
        loss_weight = torch.ones(self.model.t5.config.vocab_size)

        loss_weight[[6313, 6734, 2007, 993, 2667]] = 3.0  # decrease, increase, flat, peak, bottom
        loss_weight[[5386]] = 3.0  # increases
        loss_weight[[1267, 504, 1535]] = 3.0  # shows, show, reach
        loss_weight[[4347, 4482, 6355, 8525, 11116, 11071, 32100, 11864]] = 3.0  # num,
        loss_weight[[209, 204, 220, 314, 305, 431, 489, 505]] = 3.0  # num
        loss_weight = loss_weight.to(self.device).to(self.dtype)

        loss_func = CustomLoss(loss_weight, self.writer)

        return loss_func
