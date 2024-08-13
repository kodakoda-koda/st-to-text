import argparse
import json
import os

import numpy as np
import pandas as pd
import tokenizers
import torch
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from src.create_data import create_data
from src.data2desc.model import Model
from src.dataset.dataset import CustomDataset, df_to_dict
from src.exp.exp import eval, train
from src.utils.main_utils import log_arguments, set_logger


def main():
    # Set logger
    logger = set_logger()

    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--job_id", type=int, default=0)

    # Experiment arguments
    parser.add_argument("--start", type=str, default="2016-01-01 00:00:00")
    parser.add_argument("--end", type=str, default="2017-01-01 00:00:00")
    parser.add_argument("--hist_days", type=int, default=7)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Model arguments
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--lm_name", type=str, default="t5-base")
    parser.add_argument("--inst_max_length", type=int, default=256)
    parser.add_argument("--decoder_max_length", type=int, default=700)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_locations", type=int, default=263)

    args = parser.parse_args()
    log_arguments(args, logger)

    # Set seed
    set_seed(args.seed)

    # Load model
    logger.info(f"Loading model")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dtype = torch.bfloat16 if args.dtype == "bfloat16" else torch.float32
    model = (
        Model(
            args.n_layers,
            args.d_model,
            args.n_heads,
            args.d_ff,
            args.dropout,
            args.n_locations,
            args.lm_name,
        )
        .to(device)
        .to(dtype)
    )

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.lm_name, legacy=False)
    newline_token = tokenizers.AddedToken(content="\n", normalized=False)
    tokenizer.add_tokens(list([newline_token]))

    # Load data
    if not os.path.exists(args.data_dir + "labels.json"):
        create_data()

    logger.info(f"Loading data from {args.data_dir}")
    st_maps = np.load(args.data_dir + "st_maps.npy")
    with open(args.data_dir + "labels.json", "r") as f:
        labels = json.load(f)

    train_dataset = CustomDataset(st_maps, labels, tokenizer, args.inst_max_length, train_flag=True)
    val_dataset = CustomDataset(st_maps, labels, tokenizer, args.inst_max_length, train_flag=False)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # Train and evaluate
    logger.info("Training model")
    writer = SummaryWriter(log_dir=args.log_dir + f"{args.job_id}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=len(train_loader), num_training_steps=len(train_loader) * args.num_epochs
    )

    best_loss = np.inf
    for epoch in range(args.num_epochs):
        train_loss = train(model, train_loader, optimizer, schduler, writer, device, dtype)
        val_loss = eval(model, val_loader, device, dtype)

        # Log results
        logger.info("Epoch {} | Train Loss: {:.4f} | Val Loss: {:.4f}".format(epoch + 1, train_loss, val_loss))
        writer.add_scalar("Loss/val", val_loss, global_step=epoch)

        # Save checkpoint
        if val_loss < best_loss:
            best_loss = val_loss

            logger.info("Saving model with loss: {:.4f}".format(best_loss))
            torch.save(model.state_dict(), f"./checkpoint/checkpoint.pth")

    model.load_state_dict(torch.load(f"./checkpoint/checkpoint.pth"))

    # Save model
    if not os.path.exists(args.output_dir + f"{args.job_id}"):
        os.makedirs(args.output_dir + f"{args.job_id}")
    torch.save(model.state_dict(), args.output_dir + f"{args.job_id}/model.pth")
    tokenizer.save_pretrained(args.output_dir + f"{args.job_id}/tokenizer")

    writer.close()


if __name__ == "__main__":
    main()
