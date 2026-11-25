import numpy as np
from config import NUM_CLIENTS, MIN_POS_PER_CLIENT, SEVERE_SIZE_MULTIPLIERS
from partitioning.common import _largest_remainder, _allocate_by_class

def split_severe_original(y, seed=42):
    y = np.asarray(y, dtype=np.int64).ravel()
    sizes = _largest_remainder(SEVERE_SIZE_MULTIPLIERS, len(y))
    positives = np.full(NUM_CLIENTS, MIN_POS_PER_CLIENT, dtype=np.int64)
    remaining_pos = int(np.sum(y == 1) - positives.sum())
    if remaining_pos < 0 or np.any(sizes <= MIN_POS_PER_CLIENT):
        raise ValueError('Severe partition not feasible')
    largest_bonus = int(np.rint(0.75 * remaining_pos))
    positives[0] += largest_bonus
    positives[1] += remaining_pos - largest_bonus
    permutation = np.random.default_rng(seed + 301).permutation(NUM_CLIENTS)
    sizes = sizes[permutation]
    positives = positives[permutation]
    if np.any(positives >= sizes):
        raise ValueError('Severe positive allocation exceeds client capacity')
    return _allocate_by_class(y, sizes, positives, seed + 302)
