import torch
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, confusion_matrix,
    cohen_kappa_score, matthews_corrcoef, brier_score_loss
)

def compute_metrics(y_true, y_prob):
    y_pred = (y_prob > 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred, labels=[1, 0])
    TP, FN, FP, TN = cm.ravel()
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "specificity": TN / (TN + FP + 1e-8),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "pr_auc": average_precision_score(y_true, y_prob),
        "mcc": matthews_corrcoef(y_true, y_pred),
        "kappa": cohen_kappa_score(y_true, y_pred),
        "npv": TN / (TN + FN + 1e-8),
        "brier": brier_score_loss(y_true, y_prob),
        "fp_per_1000": (FP / (FP + TN + 1e-8)) * 1000,
        "confusion_matrix": [int(TP), int(FP), int(TN), int(FN)]
    }
    return metrics

def evaluate_model(model, X, y, batch_size=256):
    model.eval()
    all_probs = []

    dataset = TensorDataset(torch.tensor(X, dtype=torch.float32),
                            torch.tensor(y, dtype=torch.float32))
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
    model_device = next(model.parameters()).device

    with torch.no_grad():
        for xb, _ in loader:
            xb = xb.to(model_device)
            _, logits = model(xb, return_embedding=True)
            probs = torch.sigmoid(logits).cpu()
            all_probs.append(probs)

    all_probs = torch.cat(all_probs, dim=0).numpy()
    metrics = compute_metrics(y, all_probs)
    return metrics, all_probs