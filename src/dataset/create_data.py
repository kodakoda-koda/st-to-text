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
        fluc_list = ["increase", "decrease", "peak", "bottom", "flat"]
        n_spot = np.random.randint(1, 4)
        spot_list = [[np.random.randint(1, map_size - 1), np.random.randint(1, map_size - 1)] for _ in range(n_spot)]
        spot_set = set([tuple(x) for x in spot_list])
        spot_list = [list(x) for x in spot_set]
        spot_list = sorted(spot_list)
        spot_fluc_list = [random.choice(fluc_list) for _ in range(len(spot_list))]
        other_fluc = random.choice(fluc_list)

        spot_values_list = []
        spot_ind_list = []
        for j in range(len(spot_list)):
            spot_values, spot_ind = fluctuate(spot_fluc_list[j])
            spot_values_list.append(spot_values)
            spot_ind_list.append(spot_ind)
        other_values, other_ind = fluctuate(other_fluc)

        # Replace the spot with the spot value and the surrounding with the other value
        for j in range(map_size):
            for k in range(map_size):
                if [j, k] in spot_list:
                    st_maps[i, :, j, k] = spot_values_list[spot_list.index([j, k])]
                else:
                    st_maps[i, :, j, k] = other_values
        noise = np.random.randn(time_range, map_size, map_size) * 0.02
        st_maps[i] += noise

        # Create coordinates
        coords.append([[[j, k] for k in range(map_size)] for j in range(map_size)])

        # Create label text
        labels.append(label_text(spot_list, spot_fluc_list, spot_ind_list, other_fluc, other_ind))
        coords_labels.append(spot_list)

    data = {"st_maps": st_maps.tolist(), "coords": coords, "labels": labels, "coords_labels": coords_labels}
    logger.info("Data created")

    # Save data
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    json.dump(data, open(data_dir + "data.json", "w"))

    logger.info("Data saved")
