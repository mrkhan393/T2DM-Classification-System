import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

class WeightedFocalClassificationLoss(nn.Module):

    def __init__(self, pos_weight=1.0, gamma=2.0):
        super().__init__()
        if not np.isfinite(pos_weight) or pos_weight <= 0:
            raise ValueError('pos_weight must be finite and > 0')
        if gamma < 0:
            raise ValueError('gamma must be >= 0')
        self.gamma = float(gamma)
        self.register_buffer('pos_weight', torch.tensor(float(pos_weight), dtype=torch.float32))

    def forward(self, logits, targets):
        logits = logits.reshape(-1)
        targets = targets.to(dtype=logits.dtype).reshape(-1)
        if logits.numel() != targets.numel():
            raise ValueError('logit/target length mismatch')
        bce = F.binary_cross_entropy_with_logits(logits, targets, reduction='none')
        pt = torch.exp(-bce)
        class_w = torch.where(targets > 0.5, self.pos_weight.to(bce.dtype), torch.ones_like(bce))
        return (class_w * (1.0 - pt).pow(self.gamma) * bce).mean()

def positive_class_weight(y_training, max_weight=None):
    counts = np.bincount(np.asarray(y_training, dtype=int).reshape(-1), minlength=2)
    if (counts[:2] == 0).any():
        raise ValueError(f'Training subset needs both classes: {counts}')
    weight = float(counts[0] / counts[1])
    return min(weight, float(max_weight)) if max_weight is not None else weight

def supervised_contrastive_loss(embeddings, labels, temperature=0.1):
    y = labels.reshape(-1).long()
    n = embeddings.size(0)
    if n < 2:
        return embeddings.sum() * 0.0
    sim = embeddings @ embeddings.T / temperature
    sim = sim - sim.max(dim=1, keepdim=True).values.detach()
    not_self = ~torch.eye(n, dtype=torch.bool, device=embeddings.device)
    positives = y[:, None].eq(y[None, :]) & not_self
    pos_count = positives.sum(1)
    valid = pos_count > 0
    if not valid.any():
        return embeddings.sum() * 0.0
    log_prob = sim - torch.logsumexp(sim.masked_fill(~not_self, -torch.inf), dim=1, keepdim=True)
    anchor_loss = -(log_prob.masked_fill(~positives, 0.0).sum(1) / pos_count.clamp(min=1))
    per_class = [anchor_loss[valid & y.eq(label)].mean() for label in (0, 1) if (valid & y.eq(label)).any()]
    return torch.stack(per_class).mean()
