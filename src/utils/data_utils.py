from typing import List

import numpy as np


def fluctuate(fluc: str):
    if fluc == "peak":
        start_val = np.random.uniform(0.2, 0.8)
        end_val = np.random.uniform(0.2, 0.8)
        fluc_val = np.random.uniform(max(start_val, end_val) + 0.2, 1.0)
        fluc_ind = np.random.randint(3, 21)
        start_ind = np.random.randint(0, fluc_ind - 2)
        end_ind = np.random.randint(fluc_ind + 1, 24)
        ind = {"start_ind": start_ind, "fluc_ind": fluc_ind, "end_ind": end_ind}
    elif fluc == "trough":
        start_val = np.random.uniform(0.2, 0.8)
        end_val = np.random.uniform(0.2, 0.8)
        fluc_val = np.random.uniform(0, min(start_val, end_val) - 0.2)
        fluc_ind = np.random.randint(3, 21)
        start_ind = np.random.randint(0, fluc_ind - 2)
        end_ind = np.random.randint(fluc_ind + 1, 24)
        ind = {"start_ind": start_ind, "fluc_ind": fluc_ind, "end_ind": end_ind}
    elif fluc == "flat":
        start_val = np.random.uniform(0.2, 0.8)
        ind = {"start_ind": None, "fluc_ind": None, "end_ind": None}
    elif fluc == "increase":
        start_val = np.random.uniform(0.2, 0.8)
        end_val = np.random.uniform(start_val + 0.2, 1.0)
        start_ind = np.random.randint(0, 20)
        end_ind = np.random.randint(start_ind + 2, 24)
        ind = {"start_ind": start_ind, "fluc_ind": None, "end_ind": end_ind}
    elif fluc == "decrease":
        start_val = np.random.uniform(0.2, 0.8)
        end_val = np.random.uniform(0.0, start_val - 0.2)
        start_ind = np.random.randint(0, 20)
        end_ind = np.random.randint(start_ind + 2, 24)
        ind = {"start_ind": start_ind, "fluc_ind": None, "end_ind": end_ind}
    else:
        raise ValueError("Invalid fluctuation type")

    values = []
    if fluc == "flat":
        for i in range(24):
            values.append(start_val)
    else:
        for i in range(24):
            if i < start_ind:
                values.append(start_val)
            elif i > end_ind:
                values.append(end_val)
            else:
                if fluc == "peak" or fluc == "trough":
                    if i < fluc_ind:
                        values.append(start_val + (fluc_val - start_val) * (i - start_ind) / (fluc_ind - start_ind))
                    else:
                        values.append(fluc_val + (end_val - fluc_val) * (i - fluc_ind) / (end_ind - fluc_ind))
                elif fluc == "increase" or fluc == "decrease":
                    values.append(start_val + (end_val - start_val) * (i - start_ind) / (end_ind - start_ind))

    return np.array(values), ind


def label_text(spot, spot_fluc, spot_ind, other_fluc, other_ind):
    if spot_fluc == other_fluc and spot_ind == other_ind:
        if spot_fluc == "flat":
            return f"all spots are flat"
        elif spot_fluc == "peak" or spot_fluc == "trough":
            return f"all spots reach a {spot_fluc} at {spot_ind['fluc_ind']}:00"
        elif spot_fluc == "increase" or spot_fluc == "decrease":
            return f"all spots {spot_fluc} from {spot_ind['start_ind']}:00 to {spot_ind['end_ind']}:00"
        else:
            raise ValueError("Invalid fluctuation type")

    if spot_fluc == "flat":
        spot_text = f"Spot {spot} is flat"
    elif spot_fluc == "peak" or spot_fluc == "trough":
        spot_text = f"Spot {spot} reaches a {spot_fluc} at {spot_ind['fluc_ind']}:00"
    elif spot_fluc == "increase" or spot_fluc == "decrease":
        spot_text = f"Spot {spot} {spot_fluc}s from {spot_ind['start_ind']}:00 to {spot_ind['end_ind']}:00"
    else:
        raise ValueError("Invalid fluctuation type")

    if other_fluc == "flat":
        other_text = f"other spots are flat."
    elif other_fluc == "peak" or other_fluc == "trough":
        other_text = f"other spots reach a {other_fluc} at {other_ind['fluc_ind']}:00"
    elif other_fluc == "increase" or other_fluc == "decrease":
        other_text = f"other spots {other_fluc} from {other_ind['start_ind']}:00 to {other_ind['end_ind']}:00"
    else:
        raise ValueError("Invalid fluctuation type")

    if spot_fluc == other_fluc:
        conj = "and"
    else:
        conj = "while"

    return f"{spot_text}, {conj} {other_text}"
