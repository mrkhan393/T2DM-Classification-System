import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, average_precision_score, confusion_matrix, cohen_kappa_score, matthews_corrcoef, brier_score_loss

def model_probs(model, X, batch_size=1024):
    model.eval()
    model_device = next(model.parameters()).device
    probs = []
    with torch.inference_mode():
        for start in range(0, len(X), batch_size):
            inp = torch.as_tensor(X[start:start + batch_size], dtype=torch.float32, device=model_device)
            probs.append(torch.sigmoid(model(inp)).detach().cpu().numpy())
    return np.concatenate(probs)

def evaluate_model(model, X, y, batch_size=1024):
    probabilities = model_probs(model, X, batch_size=batch_size)
    return (compute_metrics(y, probabilities), probabilities)

def compute_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true, dtype=np.int64)
    y_prob = np.asarray(y_prob, dtype=float)
    y_pred = (y_prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    TP, FN, FP, TN = cm.ravel()
    metrics = {'accuracy': accuracy_score(y_true, y_pred), 'precision': precision_score(y_true, y_pred, zero_division=0), 'recall': recall_score(y_true, y_pred, zero_division=0), 'specificity': TN / (TN + FP + 1e-08), 'f1': f1_score(y_true, y_pred, zero_division=0), 'roc_auc': roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float('nan'), 'pr_auc': average_precision_score(y_true, y_prob) if len(np.unique(y_true)) == 2 else float('nan'), 'mcc': matthews_corrcoef(y_true, y_pred), 'kappa': cohen_kappa_score(y_true, y_pred), 'npv': TN / (TN + FN + 1e-08), 'brier': brier_score_loss(y_true, y_prob), 'fp_per_1000': FP / (FP + TN + 1e-08) * 1000, 'confusion_matrix': [int(TP), int(FP), int(TN), int(FN)]}
    return metrics
