import json
import logging
import os
import random

import numpy as np

from src.utils.data_utils import fluctuate, label_text

logging.basicConfig(
    format="%(asctime)s - %(message)s",
    level=logging.INFO,
    datefmt="%m/%d %H:%M",
)
logger = logging.getLogger(__name__)


def create_data(time_range: int, max_fluc_range: int, n_data: int, map_size: int, data_dir: str):
    logger.info(f"Create {n_data} data")

    # Create data
    st_maps = np.zeros((n_data, time_range, map_size, map_size))
    labels = []
    coords = []
    coords_labels = []

    for i in range(n_data):
        # Randomly generate a spot and change
        fluc_list = ["increase", "decrease", "peak", "trough", "flat"]
        spot = [np.random.randint(1, map_size - 1), np.random.randint(1, map_size - 1)]
        spot_fluc = random.choice(fluc_list)
        other_fluc = random.choice(fluc_list)

        spot_values, spot_ind = fluctuate(spot_fluc)
        other_values, other_ind = fluctuate(other_fluc)

        # Replace the spot with the spot value and the surrounding with the other value
        for j in range(map_size):
            for k in range(map_size):
                if [j, k] == spot:
                    st_maps[i, :, j, k] = spot_values
                elif j == spot[0] and (k == spot[1] - 1 or k == spot[1] + 1):
                    st_maps[i, :, j, k] = (spot_values + other_values) / 2
                elif k == spot[1] and (j == spot[0] - 1 or j == spot[0] + 1):
                    st_maps[i, :, j, k] = (spot_values + other_values) / 2
                else:
                    st_maps[i, :, j, k] = other_values
        noise = np.random.randn(time_range, map_size, map_size) * 0.02
        st_maps[i] += noise

        # Create coordinates
        coords.append([[[j, k] for k in range(map_size)] for j in range(map_size)])

        # Create label text
        labels.append(label_text(spot, spot_fluc, spot_ind, other_fluc, other_ind))
        coords_labels.append(spot)

    logger.info("Data created")

    # Save data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    data = {"st_maps": st_maps.tolist(), "coords": coords, "labels": labels, "coords_labels": coords_labels}
    with open(data_dir + "data.json", "w") as f:
        json.dump(data, f)

    logger.info("Data saved")
