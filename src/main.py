import argparse
import logging
import sys
import warnings

import transformers
from transformers import set_seed

from src.exp.exp_main import Exp_main

warnings.filterwarnings("ignore")
transformers.logging.set_verbosity_error()


def main():
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=2024)
    parser.add_argument("--data_dir", type=str, default="./data/")
    parser.add_argument("--output_dir", type=str, default="./outputs/")
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--job_id", type=int, default=0)

    # Data arguments
    parser.add_argument("--time_range", type=int, default=24)
    parser.add_argument("--max_fluc_range", type=int, default=10)
    parser.add_argument("--n_data", type=int, default=5000)
    parser.add_argument("--map_size", type=int, default=10)

    # Experiment arguments
    parser.add_argument("--num_epochs", type=int, default=400)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Model arguments
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--lm_name", type=str, default="t5-base")
    parser.add_argument("--decoder_max_length", type=int, default=80)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_layers", type=int, default=2)
    parser.add_argument("--time_steps", type=int, default=24)
    parser.add_argument("--n_locations", type=int, default=100)

    args = parser.parse_args()

    # Assert
    assert args.dtype in [
        "bfloat16",
        "float32",
    ], f"dtype should be either 'bfloat16' or 'float32', but got {args.dtype}"
    assert (
        args.n_locations == args.map_size**2
    ), f"n_locations should be equal to map_size ** 2, but got {args.n_locations}"

    # Set logger
    logging.basicConfig(
        format="%(asctime)s - %(message)s",
        stream=sys.stdout,
        level=logging.INFO,
        datefmt="%m/%d %H:%M",
    )
    logger = logging.getLogger(__name__)

    # Log arguments
    logger.info(f"job_id: {args.job_id}")

    # Set seed
    set_seed(args.seed)

    # Train and evaluate
    exp = Exp_main(args, logger)
    exp.train()
    exp.test()


if __name__ == "__main__":
    main()
