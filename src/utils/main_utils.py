import logging

import transformers


def log_arguments(args, logger):
    logger.info(f"job_id: {args.job_id}")
    logger.info(f"lr: {args.lr}")
    logger.info(f"train_batch_size: {args.train_batch_size}")
    logger.info(f"dtype: {args.dtype}")
    logger.info(f"decoder_max_length: {args.decoder_max_length}")


def set_logger():
    logger = logging.getLogger(__name__)
    logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M")

    transformers.logging.set_verbosity_error()
    return logger
