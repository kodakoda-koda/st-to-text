import logging

import transformers


def assert_arguments(args):
    assert args.dtype in [
        "bfloat16",
        "float32",
    ], f"dtype should be either 'bfloat16' or 'float32', but got {args.dtype}"
    assert (
        args.n_locations == args.map_size**2
    ), f"n_locations should be equal to map_size ** 2, but got {args.n_locations}"


def log_arguments(args, logger):
    logger.info(f"job_id: {args.job_id}")
    logger.info(f"lr: {args.lr}")
    logger.info(f"train_batch_size: {args.train_batch_size}")
    logger.info(f"dtype: {args.dtype}")
    logger.info(f"decoder_max_length: {args.decoder_max_length}")
    logger.info(f"custom_loss: {args.use_custom_loss}")


def set_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M")

    transformers.logging.set_verbosity_error()
    return logger
