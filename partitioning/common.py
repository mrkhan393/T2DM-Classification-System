import numpy as np
from config import NUM_CLIENTS

def _largest_remainder(weights, total):
    weights = np.asarray(weights, dtype=float)
    if weights.ndim != 1 or np.any(weights < 0) or weights.sum() <= 0:
        raise ValueError('Allocation weights must be nonnegative and nonzero')
    fractional = weights / weights.sum() * int(total)
    counts = np.floor(fractional).astype(np.int64)
    remainder = int(total - counts.sum())
    if remainder:
        order = np.argsort(-(fractional - counts), kind='stable')
        counts[order[:remainder]] += 1
    assert int(counts.sum()) == int(total)
    return counts

def _validate_allocation(client_indices, size):
    combined = np.concatenate(client_indices)
    if not np.array_equal(np.sort(combined), np.arange(size)):
        raise AssertionError('Original training records were duplicated or omitted')

def _allocate_by_class(y, sizes, positives, seed):
    sizes = np.asarray(sizes, dtype=np.int64)
    positives = np.asarray(positives, dtype=np.int64)
    if len(sizes) != NUM_CLIENTS or len(positives) != NUM_CLIENTS or int(sizes.sum()) != len(y) or (int(positives.sum()) != int(np.sum(y == 1))) or np.any(positives < 1) or np.any(positives >= sizes):
        raise ValueError('Infeasible client allocation')
    rng = np.random.default_rng(seed)
    pos_idx = rng.permutation(np.flatnonzero(y == 1))
    neg_idx = rng.permutation(np.flatnonzero(y == 0))
    pos_ptr = neg_ptr = 0
    clients = []
    for count, npos in zip(sizes, positives):
        count, npos = (int(count), int(npos))
        nneg = count - npos
        idx = np.r_[pos_idx[pos_ptr:pos_ptr + npos], neg_idx[neg_ptr:neg_ptr + nneg]]
        pos_ptr += npos
        neg_ptr += nneg
        rng.shuffle(idx)
        clients.append(idx.astype(np.int64))
    _validate_allocation(clients, len(y))
    return clients
