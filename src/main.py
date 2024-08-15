import argparse
import json
import os

import numpy as np
import tokenizers
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from transformers import AutoTokenizer, get_cosine_schedule_with_warmup, set_seed

from src.dataset.create_data import create_data
from src.dataset.dataset import CustomDataset
from src.exp.exp import eval, train
from src.model.model import Model
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
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--train_batch_size", type=int, default=2)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)

    # Model arguments
    parser.add_argument("--dtype", type=str, default="bfloat16")
    parser.add_argument("--lm_name", type=str, default="t5-base")
    parser.add_argument("--decoder_max_length", type=int, default=64)
    parser.add_argument("--d_model", type=int, default=768)
    parser.add_argument("--n_heads", type=int, default=12)
    parser.add_argument("--d_ff", type=int, default=3072)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_layers", type=int, default=12)
    parser.add_argument("--n_locations", type=int, default=100)

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

    train_dataset = CustomDataset(st_maps, labels, tokenizer, args.decoder_max_length, train_flag=True)
    val_dataset = CustomDataset(st_maps, labels, tokenizer, args.decoder_max_length, train_flag=False)

    train_loader = DataLoader(train_dataset, batch_size=args.train_batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=args.eval_batch_size, shuffle=False)

    # Train and evaluate
    logger.info("Training model")
    writer = SummaryWriter(log_dir=args.log_dir + f"{args.job_id}")

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    schduler = get_cosine_schedule_with_warmup(
        optimizer, num_warmup_steps=len(train_loader), num_training_steps=len(train_loader) * args.num_epochs
    )

    best_score = 0.0
    for epoch in range(args.num_epochs):
        train_loss = train(epoch, model, train_loader, optimizer, schduler, writer, device, dtype)
        score, generated_text = eval(model, val_loader, tokenizer, args.decoder_max_length, device, dtype)

        # Log results
        logger.info(
            "Epoch {} | Loss: {:.4f} | ROUGE-1: {:.4f} | ROUGE-2: {:.4f} |".format(
                epoch + 1, train_loss, score["rouge1"], score["rouge2"]
            )
        )
        writer.add_scalar("Val_rouge/rouge-1", score["rouge1"], epoch)
        writer.add_scalar("Val_rouge/rouge-2", score["rouge2"], epoch)
        writer.add_scalar("Val_rouge/rouge-L", score["rougeL"], epoch)
        writer.add_scalar("Val_rouge/rouge-Lsum", score["rougeLsum"], epoch)

        # Save checkpoint
        if score["rouge2"] > best_score:
            best_score = score["rouge2"]

            logger.info("Saving model with score: {:.4f}".format(best_score))
            if not os.path.exists("./checkpoint"):
                os.makedirs("./checkpoint")
            torch.save(model.state_dict(), f"./checkpoint/checkpoint.pth")

            if not os.path.exists(args.output_dir + f"{args.job_id}"):
                os.makedirs(args.output_dir + f"{args.job_id}")
            with open(args.output_dir + f"{args.job_id}/generated_text.json", "w") as f:
                json.dump(generated_text, f)

    model.load_state_dict(torch.load(f"./checkpoint/checkpoint.pth"))

    # Save model
    torch.save(model.state_dict(), args.output_dir + f"{args.job_id}/model.pth")
    tokenizer.save_pretrained(args.output_dir + f"{args.job_id}/tokenizer")

    writer.close()


if __name__ == "__main__":
    main()
