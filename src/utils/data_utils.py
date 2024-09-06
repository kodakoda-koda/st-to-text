from typing import List

import numpy as np


def fluctuate(range_: int, change_range: int, change_index: int, start_value: float, fluctuation: str) -> np.ndarray:
    max_value = np.random.uniform(start_value + 0.2, 1)
    min_value = np.random.uniform(0, start_value - 0.2)

    values = []
    for i in range(range_):
        if i < change_index:
            values.append(start_value)

        elif i < change_index + change_range // 2 + 1:
            if fluctuation == "increase":
                diff = (max_value - start_value) / (change_range + 1)
            elif fluctuation == "decrease":
                diff = (min_value - start_value) / (change_range + 1)
            elif fluctuation == "peak":
                diff = (max_value - start_value) / (change_range // 2 + 1)
            elif fluctuation == "dip":
                diff = (min_value - start_value) / (change_range // 2 + 1)
            else:
                diff = 0
            values.append(values[i - 1] + diff)

        elif i < change_index + change_range:
            if fluctuation == "increase":
                diff = (max_value - start_value) / (change_range + 1)
            elif fluctuation == "decrease":
                diff = (min_value - start_value) / (change_range + 1)
            elif fluctuation == "peak":
                diff = (start_value - max_value) / (change_range // 2 + 1)
            elif fluctuation == "dip":
                diff = (start_value - min_value) / (change_range // 2 + 1)
            else:
                diff = 0
            values.append(values[i - 1] + diff)

        else:
            if fluctuation == "increase":
                values.append(max_value)
            elif fluctuation == "decrease":
                values.append(min_value)
            else:
                values.append(start_value)

    return np.array(values)


def label_text(spot: List[int], spot_change: str, other_change: str):
    return f"location {spot} shows a {spot_change}, while other locations show a {other_change}."
