import time
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from config import input_dim, LOCAL_LR, ALPHA, GRAD_CLIP_NORM
from models.model import LSTMContrastiveClassifier
from models.losses import WeightedFocalClassificationLoss, positive_class_weight, supervised_contrastive_loss
from evaluation.metrics import model_probs, compute_metrics
from utils.device import seed_everything, synchronize_device

def stratified_subset(X, y, max_items, seed):
    if len(y) <= max_items:
        return (X, y)
    indices, _ = train_test_split(np.arange(len(y)), train_size=max_items, stratify=y, random_state=seed)
    return (X[indices], y[indices])

def train_one_client(global_state_cpu, X_tr, y_tr, X_val, y_val, pos_cap, gamma, epochs, batch_size, training_seed, device):
    seed_everything(training_seed, device)
    client_model = LSTMContrastiveClassifier(input_dim).to(device)
    client_model.load_state_dict(global_state_cpu)
    optimizer = torch.optim.Adam(client_model.parameters(), lr=LOCAL_LR)
    pos_weight = positive_class_weight(y_tr, pos_cap)
    loss_fn = WeightedFocalClassificationLoss(pos_weight, gamma).to(device)
    generator = torch.Generator().manual_seed(training_seed)
    loader = DataLoader(TensorDataset(torch.as_tensor(X_tr, dtype=torch.float32), torch.as_tensor(y_tr, dtype=torch.float32)), batch_size=batch_size, shuffle=True, generator=generator, num_workers=0, pin_memory=device.type == 'cuda')
    loss_total = loss_classif = loss_contrast = 0.0
    count = 0
    synchronize_device(device)
    start_time = time.perf_counter()
    for epoch in range(epochs):
        client_model.train()
        for xb, yb in loader:
            xb, yb = (xb.to(device), yb.to(device))
            optimizer.zero_grad(set_to_none=True)
            emb, logits = client_model(xb, return_embedding=True)
            c_loss = loss_fn(logits, yb)
            sup_loss = supervised_contrastive_loss(emb, yb)
            loss = c_loss + ALPHA * sup_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(client_model.parameters(), GRAD_CLIP_NORM)
            optimizer.step()
            b = len(yb)
            loss_total += loss.item() * b
            loss_classif += c_loss.item() * b
            loss_contrast += sup_loss.item() * b
            count += b
    probs = model_probs(client_model, X_val, batch_size=batch_size)
    metrics = compute_metrics(y_val, probs)
    eps = 1e-07
    val_loss = float(-np.mean(np.asarray(y_val) * np.log(np.clip(probs, eps, 1 - eps)) + (1 - np.asarray(y_val)) * np.log(np.clip(1 - probs, eps, 1 - eps))))
    metrics['client_val_loss'] = val_loss
    local_state_cpu = {k: v.detach().cpu().clone() for k, v in client_model.state_dict().items()}
    synchronize_device(device)
    duration = time.perf_counter() - start_time
    del client_model, optimizer
    return (local_state_cpu, metrics, {'total': loss_total / count, 'classification': loss_classif / count, 'contrastive': loss_contrast / count, 'pos_weight': float(pos_weight), 'duration': duration})
