import numpy as np
from config import NUM_CLIENTS, MIN_POS_PER_CLIENT
from partitioning.common import _validate_allocation

def split_mild_original(y, seed=42):
    y = np.asarray(y, dtype=np.int64).ravel()
    rng = np.random.default_rng(seed)
    pos_idx = np.where(y == 1)[0]
    neg_idx = np.where(y == 0)[0]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)
    if len(pos_idx) < NUM_CLIENTS * MIN_POS_PER_CLIENT:
        raise ValueError('Not enough real positives to guarantee 700/client')
    sizes = np.full(NUM_CLIENTS, len(y) // NUM_CLIENTS, dtype=np.int64)
    sizes[:len(y) % NUM_CLIENTS] += 1
    positives = np.full(NUM_CLIENTS, MIN_POS_PER_CLIENT, dtype=np.int64)
    remaining_pos = len(pos_idx) - int(positives.sum())
    weights = rng.dirichlet(np.ones(NUM_CLIENTS) * 2.0)
    extra = rng.multinomial(remaining_pos, weights)
    positives += extra
    negatives = sizes - positives
    if (negatives < 0).any():
        raise ValueError('Allocated more positives than client size')
    clients = []
    p_ptr = n_ptr = 0
    for p, n in zip(positives, negatives):
        p, n = (int(p), int(n))
        idx = np.concatenate([pos_idx[p_ptr:p_ptr + p], neg_idx[n_ptr:n_ptr + n]])
        p_ptr += p
        n_ptr += n
        rng.shuffle(idx)
        clients.append(idx.astype(np.int64))
    _validate_allocation(clients, len(y))
    return clients
