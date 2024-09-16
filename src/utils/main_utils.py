import logging
from argparse import Namespace

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    datefmt="%m/%d %H:%M:%S",
)
logger = logging.getLogger(__name__)


def assert_arguments(args: Namespace):
    assert args.dtype in [
        "bfloat16",
        "float32",
    ], f"dtype should be either 'bfloat16' or 'float32', but got {args.dtype}"
    assert (
        args.n_locations == args.map_size**2
    ), f"n_locations should be equal to map_size ** 2, but got {args.n_locations}"


def log_arguments(args: Namespace):
    logger.info(f"job_id: {args.job_id}")
    logger.info(f"lr: {args.lr}")
    logger.info(f"train_batch_size: {args.train_batch_size}")
    logger.info(f"dtype: {args.dtype}")
    logger.info(f"decoder_max_length: {args.decoder_max_length}")
