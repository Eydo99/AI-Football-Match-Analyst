import numpy as np


def safe_extract(location, index):
    if (location is None) or (not isinstance(location, (list, tuple,np.ndarray)) )or (len(location) <= index):
        return np.nan
    return location[index]