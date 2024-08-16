import argparse
import json
import logging
import os
import random

import numpy as np

from src.utils.data_utils import fluctuate, label_text

logging.basicConfig(format="%(asctime)s: %(message)s", level=logging.INFO, datefmt="%Y-%m-%d %H:%M:%S")
logger = logging.getLogger(__name__)


def create_data(time_range: int, max_fluc_range: int, n_data: int, map_size: int, data_dir: str):
    logger.info(f"Create {n_data} data")

    # Create data
    st_maps = np.zeros((n_data, time_range, map_size, map_size))
    labels = []

    for i in range(n_data):
        # Randomly generate a spot and change
        fluc_range = np.random.randint(1, max_fluc_range)
        fluc_index = np.random.randint(3, time_range - fluc_range - 3)
        start_value = np.random.uniform(0.2, 0.8)
        fluc_list = ["increase", "decrease", "peak", "dip", "flat"]
        spot = (np.random.randint(1, map_size - 1), np.random.randint(1, map_size - 1))
        spot_change = random.choice(fluc_list)
        other_change = random.choice(fluc_list)

        if spot_change == other_change:
            spot_value = fluctuate(time_range, fluc_range, fluc_index, start_value, spot_change)
            other_value = spot_value.copy()
        else:
            spot_value = fluctuate(time_range, fluc_range, fluc_index, start_value, spot_change)
            other_value = fluctuate(time_range, fluc_range, fluc_index, start_value, spot_change)

        # Replace the spot with the spot value and the surrounding with the other value
        for j in range(map_size):
            for k in range(map_size):
                if (j, k) == spot:
                    st_maps[i, :, j, k] = spot_value
                elif j == spot[0] and (k == spot[1] - 1 or k == spot[1] + 1):
                    st_maps[i, :, j, k] = (spot_value + other_value) / 2
                elif k == spot[1] and (j == spot[0] - 1 or j == spot[0] + 1):
                    st_maps[i, :, j, k] = (spot_value + other_value) / 2
                else:
                    st_maps[i, :, j, k] = other_value
        noise = np.random.randn(time_range, map_size, map_size) * 0.02
        st_maps[i] += noise

        # Create label text
        labels.append(label_text(spot, spot_change, other_change))

    logger.info("Data created")

    # Save data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    np.save(data_dir + "st_maps.npy", st_maps)
    with open(data_dir + "labels.json", "w") as f:
        json.dump(labels, f)

    logger.info("Data saved")
