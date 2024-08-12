import argparse
import random

import numpy as np

from src.utils import dec_func, dip_func, flat_func, inc_func, peak_func, to_text


def create_data():
    argparser = argparse.ArgumentParser()
    argparser.add_argument("--range", type=int, default=30)
    argparser.add_argument("--change_range", type=int, default=10)
    argparser.add_argument("--n_data", type=int, default=500)
    argparser.add_argument("--map_size", type=int, default=10)

    args = argparser.parse_args()

    st_map = np.zeros((args.n_data, args.range, args.map_size, args.map_size))
    labels = []

    for i in range(args.n_data):
        change_range = np.random.randint(1, args.change_range)
        change_index = np.random.randint(3, args.range - change_range - 3)
        start_value = np.random.randint(0.2, 0.8)
        change_dict = {
            "increase": inc_func,
            "decrease": dec_func,
            "peak": peak_func,
            "dip": dip_func,
            "flat": flat_func,
        }

        spot_change = random.choice(list(change_dict.keys()))
        other_change = random.choice(list(change_dict.keys()))

        if spot_change == other_change:
            spot_value = change_dict[spot_change](start_value, change_range, change_index)
            other_value = spot_value.copy()
        else:
            spot_value = change_dict[spot_change](start_value, change_range, change_index)
            other_value = change_dict[other_change](start_value, change_range, change_index)

        spot = (np.random.randint(1, args.map_size - 1), np.random.randint(1, args.map_size - 1))

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

        labels.append(to_text(spot, spot_change, other_change))

    np.save("data/st_map.npy", st_map)
    with open("data/labels.json", "w") as f:
        f.dump(labels)


if __name__ == "__main__":
    create_data()
