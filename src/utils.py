import numpy as np


def inc_func(range_, change_range, change_index, start_value):
    max_value = np.random.uniform(start_value + 0.2, 1)

    inc = []
    for i in range(range_):
        if i < change_index:
            inc.append(start_value)
        elif i < change_index + change_range:
            inc.append(inc[i - 1] + (max_value - start_value) / (change_range + 1))
        else:
            inc.append(max_value)
    return np.array(inc)


def dec_func(range_, change_range, change_index, start_value):
    min_value = np.random.uniform(0, start_value - 0.2)

    dec = []
    for i in range(range_):
        if i < change_index:
            dec.append(start_value)
        elif i < change_index + change_range:
            dec.append(dec[i - 1] - (start_value - min_value) / (change_range + 1))
        else:
            dec.append(min_value)
    return np.array(dec)


def peak_func(range_, change_range, change_index, start_value):
    max_value = np.random.uniform(start_value + 0.2, 1)

    peak = []
    for i in range(range_):
        if i < change_index:
            peak.append(start_value)
        elif i < change_index + change_range // 2 + 1:
            peak.append(peak[i - 1] + (max_value - start_value) / (change_range // 2 + 1))
        elif i < change_index + change_range:
            peak.append(peak[i - 1] - (max_value - start_value) / (change_range // 2 + 1))
        else:
            peak.append(start_value)
    return np.array(peak)


def dip_func(range_, change_range, change_index, start_value):
    min_value = np.random.uniform(0, start_value - 0.2)

    dip = []
    for i in range(range_):
        if i < change_index:
            dip.append(start_value)
        elif i < change_index + change_range // 2 + 1:
            dip.append(dip[i - 1] - (start_value - min_value) / (change_range // 2 + 1))
        elif i < change_index + change_range:
            dip.append(dip[i - 1] + (start_value - min_value) / (change_range // 2 + 1))
        else:
            dip.append(start_value)
    return np.array(dip)


def flat_func(range_, change_range, change_index, start_value):
    flat = []
    for i in range(range_):
        flat.append(start_value)
    return np.array(flat)


def to_text(spot, spot_change, other_change):
    if spot_change == other_change:
        return f"all locations show a {spot_change}."
    else:
        return f"location {spot} shows a {spot_change}, while other locations show a {other_change}."
