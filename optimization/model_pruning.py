import os
import copy
import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
from utils.metrics import evaluate_model

def get_pruning_targets(model):
    targets = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            targets.append((m, 'weight'))
        if isinstance(m, nn.LSTM):
            targets.extend([(m, 'weight_ih_l0'), (m, 'weight_hh_l0')])
    return targets

def prune_global_model(model, epoch, start_ep=30, end_ep=70, final_sparsity=0.5):
    pruning_targets = get_pruning_targets(model)
    if start_ep <= epoch <= end_ep:
        step = epoch - start_ep + 1
        total_steps = end_ep - start_ep + 1
        target_sparsity = final_sparsity * (step / total_steps)
        prune.global_unstructured(
            pruning_targets,
            pruning_method=prune.L1Unstructured,
            amount=target_sparsity
        )
    return model

def remove_pruning(model):
    for name, module in model.named_modules():
        if isinstance(module, (nn.Linear, nn.LSTM)):
            for attr in ['weight', 'bias']:
                try:
                    prune.remove(module, attr)
                except Exception:
                    pass
            for weight_name in ['weight_ih_l0', 'weight_hh_l0',
                                'weight_ih_l0_reverse', 'weight_hh_l0_reverse',
                                'weight_ih_l1', 'weight_hh_l1',
                                'weight_ih_l1_reverse', 'weight_hh_l1_reverse']:
                try:
                    prune.remove(module, weight_name)
                except Exception:
                    pass

# Alias to avoid crash from original script naming mismatch
remove_pruning_masks = remove_pruning

def quantize_model(model, save_path):
    model_int8 = torch.ao.quantization.quantize_dynamic(
        copy.deepcopy(model).to("cpu").eval(),
        {nn.Linear, nn.LSTM},
        dtype=torch.qint8
    )
    torch.save(model_int8.state_dict(), save_path)
    return model_int8

def summarize_model_with_metrics(model, label, X_test, y_test, save_path):
    model_cpu = copy.deepcopy(model).to("cpu").eval()
    total_params = sum(p.numel() for p in model_cpu.parameters())
    nonzero_params = sum(torch.count_nonzero(p).item() for p in model_cpu.parameters())
    sparsity = 100 * (1 - nonzero_params / total_params)
    torch.save(model_cpu.state_dict(), save_path)
    size_mb = os.path.getsize(save_path) / (1024 * 1024)
    metrics, _ = evaluate_model(model_cpu, X_test, y_test)
    return {
        "label": label, "total": total_params, "nonzero": nonzero_params,
        "sparsity": sparsity, "size_mb": size_mb, "metrics": metrics
    }