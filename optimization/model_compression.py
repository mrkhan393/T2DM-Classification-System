import os, time, copy, platform, warnings
import numpy as np
import pandas as pd
import torch
from config import FINAL_SPARSITY, RUN_COMPRESSION_EVAL, RUN_INFERENCE_BENCHMARK, features
from utils.device import synchronize_device
from evaluation.metrics import evaluate_model, model_probs, compute_metrics
import torch.nn as nn
import torch.nn.utils.prune as prune

def run_compression(data, result, device):
    manifest, client_splits, client_real_train, client_real_training_counts, client_meta, X_validation, y_validation, X_test_scaled, y_test, X_test_balanced_scaled, y_test_balanced, holdout_indices, input_fingerprint = data
    global_model, best_global_round, SAVE_DIR, global_csv = result
    if RUN_COMPRESSION_EVAL:
        from pathlib import Path

        def get_pruning_targets(model):
            targets = []
            for module in model.modules():
                if isinstance(module, nn.Linear):
                    targets.append((module, 'weight'))
                elif isinstance(module, nn.LSTM):
                    for pname in ['weight_ih_l0', 'weight_hh_l0', 'weight_ih_l0_reverse', 'weight_hh_l0_reverse']:
                        if hasattr(module, pname):
                            targets.append((module, pname))
            return targets

        def make_pruned_copy(source_model, amount=0.5):
            result = copy.deepcopy(source_model).cpu().eval()
            targets = get_pruning_targets(result)
            prune.global_unstructured(targets, pruning_method=prune.L1Unstructured, amount=amount)
            for module, pname in targets:
                prune.remove(module, pname)
            return result

        def quantize_copy(source_model):
            if platform.system() == 'Darwin' and platform.machine().lower() in ('arm64', 'aarch64'):
                if 'qnnpack' in torch.backends.quantized.supported_engines:
                    torch.backends.quantized.engine = 'qnnpack'
            elif 'x86' in torch.backends.quantized.supported_engines:
                torch.backends.quantized.engine = 'x86'
            failure = None
            for module_types, scope in [({nn.Linear, nn.LSTM}, 'LSTM+Linear'), ({nn.Linear}, 'Linear-only')]:
                try:
                    with warnings.catch_warnings():
                        warnings.filterwarnings('ignore', message='torch.ao.quantization is deprecated')
                        candidate = torch.ao.quantization.quantize_dynamic(copy.deepcopy(source_model).cpu().eval(), module_types, dtype=torch.qint8)
                    with torch.inference_mode():
                        probe_result = candidate(torch.zeros(2, len(features), 1, dtype=torch.float32))
                    if not torch.isfinite(probe_result).all():
                        raise FloatingPointError('Non-finite quantized output')
                    return (candidate, scope)
                except Exception as exc:
                    failure = exc
                    warnings.warn(f'CPU dynamic quantization ({scope}) unsupported: {exc}')
            raise RuntimeError('No supported CPU INT8 quantization route is available') from failure

        def summarize_variant(model, name, filepath, X, y, packed=False):
            model = model.cpu().eval()
            torch.save(model.state_dict(), filepath)
            prob = model_probs(model, X)
            metrics = compute_metrics(y, prob)
            if packed:
                params, nzero, sparsity = ('N/A (packed int8)', 'N/A (packed int8)', 'N/A (packed int8)')
            else:
                params = sum((p.numel() for p in model.parameters()))
                nzero = sum((torch.count_nonzero(p).item() for p in model.parameters()))
                sparsity = 100.0 * (1 - nzero / params)
            return {'label': name, 'total': params, 'nonzero': nzero, 'sparsity': sparsity, 'size_mb': os.path.getsize(filepath) / 1024 ** 2, 'metrics': metrics, 'probabilities': prob}
        original_model = copy.deepcopy(global_model).cpu().eval()
        pruned_model = make_pruned_copy(original_model, FINAL_SPARSITY)
        quantized_model, quant_scope = quantize_copy(original_model)
        pruned_quantized_model, pruned_quant_scope = quantize_copy(pruned_model)
        print('CPU quantization backend:', torch.backends.quantized.engine)
        print('Quantization scopes:', quant_scope, '| pruned:', pruned_quant_scope)
        variants = [(original_model, 'Original', 'final_original_model.pth', False), (pruned_model, 'Pruned', 'final_pruned_model.pth', False), (quantized_model, 'Quantized', 'final_quantized_model.pth', True), (pruned_quantized_model, 'Pruned+Quant', 'final_pruned_quantized_model.pth', True)]
        summary = {}
        for model, label, filename, packed in variants:
            summary[label] = summarize_variant(model, label, os.path.join(SAVE_DIR, filename), X_test_scaled, y_test, packed=packed)
        rows = [['Total Params'] + [summary[label]['total'] for _, label, _, _ in variants], ['Non-Zero Params'] + [summary[label]['nonzero'] for _, label, _, _ in variants], ['Sparsity (%)'] + [summary[label]['sparsity'] for _, label, _, _ in variants], ['Size (MB)'] + [summary[label]['size_mb'] for _, label, _, _ in variants]]
        for name in summary['Original']['metrics']:
            if name != 'confusion_matrix':
                rows.append([name] + [summary[label]['metrics'][name] for _, label, _, _ in variants])
        comparison_df = pd.DataFrame(rows, columns=['Metric', 'Original', 'Pruned', 'Quantized', 'Pruned+Quant'])
        comparison_df.to_csv(os.path.join(SAVE_DIR, 'global_model_comparison.csv'), index=False)
        print(comparison_df.to_string(index=False))
        final_test_metrics = summary['Original']['metrics']
        final_test_probs = summary['Original']['probabilities']
        balanced_metrics, balanced_test_probs = evaluate_model(original_model, X_test_balanced_scaled, y_test_balanced)

        def as_one_row(metrics, name, outfile):
            d = {'evaluation_split': name, **{k: str(v) if isinstance(v, (list, tuple)) else v for k, v in metrics.items()}}
            pd.DataFrame([d]).to_csv(os.path.join(SAVE_DIR, outfile), index=False)
        as_one_row(final_test_metrics, 'untouched_natural_prevalence_test', 'final_natural_test_metrics.csv')
        as_one_row(balanced_metrics, 'supplementary_balanced_486_per_class', 'final_balanced_diagnostic_metrics.csv')
        pd.DataFrame({'true_label': y_test, 'predicted_probability': final_test_probs}).to_csv(os.path.join(SAVE_DIR, 'final_natural_test_predictions.csv'), index=False)
        pd.DataFrame({'true_label': y_test_balanced, 'predicted_probability': balanced_test_probs}).to_csv(os.path.join(SAVE_DIR, 'final_balanced_diagnostic_predictions.csv'), index=False)
        pd.DataFrame({'true_label': y_test, **{label: summary[label]['probabilities'] for label in summary}}).to_csv(os.path.join(SAVE_DIR, 'model_variant_test_probabilities.csv'), index=False)
        print('Natural test results:', final_test_metrics)
        print('Balanced diagnostic results:', balanced_metrics)
        pd.DataFrame([{'variant': 'Original', 'quantization_scope': 'none', 'quantization_backend': 'none'}, {'variant': 'Pruned', 'quantization_scope': 'none', 'quantization_backend': 'none'}, {'variant': 'Quantized', 'quantization_scope': quant_scope, 'quantization_backend': torch.backends.quantized.engine}, {'variant': 'Pruned+Quant', 'quantization_scope': pruned_quant_scope, 'quantization_backend': torch.backends.quantized.engine}]).to_csv(os.path.join(SAVE_DIR, 'quantization_backend_details.csv'), index=False)

        def inference_latency_profile(model, variant, target, batch_size, warmup=3, trials=12):
            model = model.to(target).eval()
            xb = torch.zeros(batch_size, len(features), 1, dtype=torch.float32, device=target)
            with torch.inference_mode():
                for _ in range(warmup):
                    _ = model(xb)
                synchronize_device(target)
                times = []
                for _ in range(trials):
                    synchronize_device(target)
                    t0 = time.perf_counter()
                    _ = model(xb)
                    synchronize_device(target)
                    times.append((time.perf_counter() - t0) * 1000.0)
            median_ms = float(np.median(times))
            return {'variant': variant, 'inference_device': str(target), 'batch_size': batch_size, 'warmup_runs': warmup, 'timed_runs': trials, 'median_batch_ms': median_ms, 'p95_batch_ms': float(np.percentile(times, 95)), 'median_ms_per_observation': median_ms / batch_size, 'throughput_obs_per_sec': 1000 * batch_size / median_ms}
        if RUN_INFERENCE_BENCHMARK:
            latency_rows = []
            for model, label, filename, packed in variants:
                for batch_size in (1, 256):
                    latency_rows.append(inference_latency_profile(model, label, torch.device('cpu'), batch_size, trials=15 if batch_size == 1 else 6))
            if device.type == 'mps':
                original_mps = copy.deepcopy(original_model).to(device).eval()
                for batch_size in (1, 256):
                    latency_rows.append(inference_latency_profile(original_mps, 'Original-MPS', device, batch_size, trials=15 if batch_size == 1 else 6))
                del original_mps
            pd.DataFrame(latency_rows).to_csv(os.path.join(SAVE_DIR, 'inference_latency_benchmark.csv'), index=False)
            print(pd.DataFrame(latency_rows).to_string(index=False))
        else:
            pd.DataFrame(columns=['variant', 'inference_device', 'batch_size', 'warmup_runs', 'timed_runs', 'median_batch_ms', 'p95_batch_ms', 'median_ms_per_observation', 'throughput_obs_per_sec']).to_csv(os.path.join(SAVE_DIR, 'inference_latency_benchmark.csv'), index=False)
    else:
        print('Compression and inference benchmarking skipped for dataset selection.')
    if RUN_COMPRESSION_EVAL:
        for name, model in [('Original', original_model), ('Pruned', pruned_model)]:
            total = sum((param.numel() for param in model.parameters()))
            nonzero = sum((torch.count_nonzero(param).item() for param in model.parameters()))
            print(name, 'total', total, 'nonzero', nonzero, 'overall sparsity (%)', round(100 * (1 - nonzero / total), 3))
    else:
        print('Pruning/quantization can be enabled later without retraining (best checkpoint saved).')
