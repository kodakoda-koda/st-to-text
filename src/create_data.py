import argparse
import json
import logging
import os
import random

import numpy as np

from src.utils.data_utils import fluctuate, label_text

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def create_data():
    # Parse arguments
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--range", type=int, default=30)
    argparser.add_argument("--change_range", type=int, default=10)
    argparser.add_argument("--n_data", type=int, default=500)
    argparser.add_argument("--map_size", type=int, default=10)
    argparser.add_argument("--output_dir", type=str, default="./data/")
    args = argparser.parse_args()

    logger.info(f"Create {args.n_data} data")

    # Create data
    st_map = np.zeros((args.n_data, args.range, args.map_size, args.map_size))
    labels = []

    for i in range(args.n_data):
        # Randomly generate a spot and change
        change_range = np.random.randint(1, args.change_range)
        change_index = np.random.randint(3, args.range - change_range - 3)
        start_value = np.random.uniform(0.2, 0.8)
        change_list = ["increase", "decrease", "peak", "dip", "flat"]
        spot = (np.random.randint(1, args.map_size - 1), np.random.randint(1, args.map_size - 1))
        spot_change = random.choice(change_list)
        other_change = random.choice(change_list)

        if spot_change == other_change:
            spot_value = fluctuate(args.range, change_range, change_index, start_value, spot_change)
            other_value = spot_value.copy()
        else:
            spot_value = fluctuate(args.range, change_range, change_index, start_value, spot_change)
            other_value = fluctuate(args.range, change_range, change_index, start_value, other_change)

        # Replace the spot with the spot value and the surrounding with the other value
        for j in range(args.map_size):
            for k in range(args.map_size):
                if (j, k) == spot:
                    st_map[i, :, j, k] = spot_value
                elif j == spot[0] and (k == spot[1] - 1 or k == spot[1] + 1):
                    st_map[i, :, j, k] = (spot_value + other_value) / 2
                elif k == spot[1] and (j == spot[0] - 1 or j == spot[0] + 1):
                    st_map[i, :, j, k] = (spot_value + other_value) / 2
                else:
                    st_map[i, :, j, k] = other_value
        noise = np.random.randn(args.range, args.map_size, args.map_size) * 0.02
        st_map[i] += noise

        # Create label text
        labels.append(label_text(spot, spot_change, other_change))

    logger.info("Data created")

    # Save data
    np.save(args.output_dir + "st_map.npy", st_map)
    with open(args.output_dir + "labels.json", "w") as f:
        json.dump(labels, f)

    logger.info("Data saved")


if __name__ == "__main__":
    create_data()
