import numpy as np
from config import NUM_CLIENTS
from partitioning.common import _largest_remainder, _allocate_by_class

def split_iid_original(y, seed=42):
    y = np.asarray(y, dtype=np.int64).ravel()
    sizes = np.full(NUM_CLIENTS, len(y) // NUM_CLIENTS, dtype=np.int64)
    sizes[:len(y) % NUM_CLIENTS] += 1
    positives = _largest_remainder(sizes, int(np.sum(y == 1)))
    return _allocate_by_class(y, sizes, positives, seed + 201)
