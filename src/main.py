import argparse
import logging
import warnings

import transformers
from transformers import set_seed

from src.exp.exp_main import Exp_main
from src.utils.main_utils import assert_arguments, log_arguments

warnings.filterwarnings("ignore")
logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    datefmt="%m/%d %H:%M",
)
logger = logging.getLogger(__name__)
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
    parser.add_argument("--num_epochs", type=int, default=200)
    parser.add_argument("--train_batch_size", type=int, default=8)
    parser.add_argument("--eval_batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)

    # Model arguments
    parser.add_argument("--dtype", type=str, default="float32")
    parser.add_argument("--lm_name", type=str, default="t5-base")
    parser.add_argument("--decoder_max_length", type=int, default=32)
    parser.add_argument("--d_model", type=int, default=512)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--d_ff", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--n_layers", type=int, default=1)
    parser.add_argument("--n_locations", type=int, default=100)

    args = parser.parse_args()
    assert_arguments(args)
    log_arguments(args)

    # Set seed
    set_seed(args.seed)

    # Train and evaluate
    exp = Exp_main(args)
    exp.train()


if __name__ == "__main__":
    main()
