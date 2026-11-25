import numpy as np
from config import NUM_CLIENTS, NUM_SELECTED_CLIENTS, CLIENT_SELECTION_MODE

def weighted_fedavg(local_states, local_counts, selected):
    total = float(sum((local_counts[i] for i in selected)))
    merged = {}
    for name in local_states[0]:
        merged[name] = sum((local_states[i][name] * (local_counts[i] / total) for i in selected))
    return merged

def selection_score(metrics):
    return 0.5 * metrics['f1'] + 0.5 * metrics['pr_auc']

def choose_client_updates(scores, historical_counts, round_number):
    scores = np.asarray(scores, dtype=np.float64)
    if scores.shape != (NUM_CLIENTS,):
        raise ValueError(f'Expected exactly {NUM_CLIENTS} scores, got {scores.shape}')
    order = np.argsort(-scores, kind='stable')
    if CLIENT_SELECTION_MODE == 'performance_top5':
        return order[:NUM_SELECTED_CLIENTS]
    if CLIENT_SELECTION_MODE != 'top4_rotation1':
        raise ValueError('Unknown selection mode: ' + str(CLIENT_SELECTION_MODE))
    top = list(order[:NUM_SELECTED_CLIENTS - 1])
    candidates = [i for i in range(NUM_CLIENTS) if i not in top]
    rotating_rank = lambda i: (i - (round_number - 1)) % NUM_CLIENTS
    fair_slot = min(candidates, key=lambda i: (historical_counts[i], rotating_rank(i)))
    return np.asarray(top + [fair_slot], dtype=int)
