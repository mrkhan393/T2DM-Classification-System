import os, csv, time, json, hashlib, platform
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from config import ALPHA, BATCH_SIZE, CLIENT_SELECTION_MODE, EXPERIMENT_SEED, GLOBAL_ROUNDS, GRAD_CLIP_NORM, LOCAL_EPOCHS, LOCAL_LR, NUM_CLIENTS, NUM_SELECTED_CLIENTS, OUTPUT_PATH, PARTITION_ROOT, SCENARIO, SELECTED_GAMMA, SELECTED_WEIGHT_CAP, TRAIN_MONITOR_SIZE, input_dim
from utils.device import synchronize_device
from evaluation.metrics import evaluate_model, model_probs, compute_metrics
from utils.device import CPU_THREADS, seed_everything
from models.model import LSTMContrastiveClassifier
from federated.training import train_one_client, stratified_subset
from federated.aggregation import weighted_fedavg, selection_score, choose_client_updates
from utils.checkpoint import atomic_json_save, trim_csv_to_round

def run_federated(data, device):
    manifest, client_splits, client_real_train, client_real_training_counts, client_meta, X_validation, y_validation, X_test_scaled, y_test, X_test_balanced_scaled, y_test_balanced, holdout_indices, input_fingerprint = data
    SAVE_DIR = str(OUTPUT_PATH.resolve())
    OUTPUT_PATH.mkdir(parents=True, exist_ok=True)
    client_csv = os.path.join(SAVE_DIR, 'client_metrics.csv')
    global_csv = os.path.join(SAVE_DIR, 'global_validation_metrics.csv')
    selection_csv = os.path.join(SAVE_DIR, 'selection_log.csv')
    tune_csv = os.path.join(SAVE_DIR, 'focal_hyperparameter_search.csv')
    CHECKPOINT_PATH = OUTPUT_PATH / 'resume_checkpoint.pt'
    CONFIG_PATH = OUTPUT_PATH / 'run_config.json'
    client_headers = ['round', 'client_id', 'num_samples', 'pos_samples', 'neg_samples', 'data_ratio_percent', 'local_train_samples', 'local_train_real_samples', 'local_train_synthetic_samples', 'fedavg_weight_samples', 'local_val_samples', 'local_train_pos_weight', 'focal_gamma', 'positive_weight_cap', 'was_selected', 'selection_rank', 'selection_score', 'client_total_loss', 'client_classification_loss', 'client_contrastive_loss', 'client_val_loss', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc', 'pr_auc', 'mcc', 'kappa', 'npv', 'brier', 'fp_per_1000', 'confusion_matrix', 'client_runtime_sec', 'local_epochs']
    global_headers = ['round', 'evaluation_split', 'global_total_loss', 'global_classification_loss', 'global_contrastive_loss', 'global_pos_weight', 'focal_gamma', 'positive_weight_cap', 'accuracy', 'precision', 'recall', 'specificity', 'f1', 'roc_auc', 'pr_auc', 'mcc', 'kappa', 'npv', 'brier', 'fp_per_1000', 'confusion_matrix', 'round_runtime_sec', 'num_selected_clients']
    selection_headers = ['round', 'selected_clients_list', 'dropped_clients_list', 'avg_selected_client_loss', 'avg_dropped_client_loss', 'selection_strategy']
    hardware_info = {'platform': platform.platform(), 'architecture': platform.machine(), 'python_version': platform.python_version(), 'torch_version': torch.__version__, 'training_device': str(device), 'mps_built': bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_built()), 'mps_available': bool(hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()), 'cpu_threads': CPU_THREADS, 'mps_unsupported_operation_fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK', ''), 'scenario': SCENARIO, 'input_fingerprint': input_fingerprint}
    print('Output directory:', SAVE_DIR)
    X_train_original_pool = np.concatenate([xy[0] for xy in client_real_train], axis=0)
    y_train_original_pool = np.concatenate([xy[1] for xy in client_real_train], axis=0)
    X_train_monitor, y_train_monitor = stratified_subset(X_train_original_pool, y_train_original_pool, TRAIN_MONITOR_SIZE, EXPERIMENT_SEED + 4000)
    del X_train_original_pool, y_train_original_pool
    run_configuration = {'scenario': SCENARIO, 'partition_root': str(PARTITION_ROOT), 'input_fingerprint': input_fingerprint, 'seed': EXPERIMENT_SEED, 'rounds': GLOBAL_ROUNDS, 'local_epochs': LOCAL_EPOCHS, 'batch_size': BATCH_SIZE, 'learning_rate': LOCAL_LR, 'contrastive_alpha': ALPHA, 'grad_clip': GRAD_CLIP_NORM, 'selected_clients_per_round': NUM_SELECTED_CLIENTS, 'selection_mode': CLIENT_SELECTION_MODE, 'focal_gamma': SELECTED_GAMMA, 'positive_weight_cap': 'uncapped', 'aggregation_weight': 'real_original_client_training_count', 'client_training_source': 'ADASYN5_saved_scaled', 'model_architecture': '2-layer BiLSTM 128/64 + 64 projection, sigmoid logit', 'test_policy': 'untouched natural-prevalence outer test after training only'}
    configuration_digest = hashlib.sha256(json.dumps(run_configuration, sort_keys=True).encode()).hexdigest()
    hardware_info.update({'client_selection_mode': CLIENT_SELECTION_MODE, 'real_weighted_fedavg': True})
    pd.DataFrame([hardware_info]).to_csv(os.path.join(SAVE_DIR, 'hardware_runtime_info.csv'), index=False)

    def read_checkpoint():
        try:
            return torch.load(CHECKPOINT_PATH, map_location='cpu', weights_only=True)
        except TypeError:
            return torch.load(CHECKPOINT_PATH, map_location='cpu')

    def configure_resume():
        nonlocal best_global_state_cpu, best_val_key, best_global_round
        nonlocal selection_history, train_val_gap_rows, validation_history
        if CONFIG_PATH.exists():
            cfg = json.loads(CONFIG_PATH.read_text())
            if cfg.get('digest') != configuration_digest:
                raise RuntimeError('Existing output has a DIFFERENT data/training fingerprint. Use another output folder.')
        elif CHECKPOINT_PATH.exists():
            raise RuntimeError('Resume checkpoint exists but matching run_config.json is missing.')
        else:
            if any(((OUTPUT_PATH / nm).exists() for nm in ('client_metrics.csv', 'global_validation_metrics.csv', 'selection_log.csv', 'final_natural_test_metrics.csv'))):
                raise RuntimeError('Output folder contains prior experiment logs, without a valid checkpoint. Use a new output folder.')
            atomic_json_save({'digest': configuration_digest, 'settings': run_configuration}, CONFIG_PATH)
        if CHECKPOINT_PATH.exists():
            cp = read_checkpoint()
            if cp.get('digest') != configuration_digest:
                raise RuntimeError('Checkpoint data/training fingerprint mismatch. Refusing to resume.')
            completed = int(cp['last_completed_round'])
            if completed > GLOBAL_ROUNDS or completed < 1:
                raise RuntimeError('Invalid checkpoint round.')
            global_model.load_state_dict(cp['global_model_state'])
            best_global_state_cpu = cp['best_global_state']
            best_val_key = tuple((float(x) for x in cp['best_val_key']))
            best_global_round = int(cp['best_global_round'])
            selection_history = np.asarray(cp['selection_history'], dtype=int)
            print(f'RESUME: {completed} rounds committed, continuing at round {completed + 1}')
        else:
            completed = 0
            print(f'NEW 50-round ADASYN run, seed={EXPERIMENT_SEED}')
        for path, headers in ((client_csv, client_headers), (global_csv, global_headers), (selection_csv, selection_headers)):
            trim_csv_to_round(path, completed, headers)
        gap_path = Path(SAVE_DIR) / 'training_validation_gap.csv'
        if completed:
            if not gap_path.exists():
                raise RuntimeError('Checkpoint exists but training_validation_gap.csv is missing.')
            gap = pd.read_csv(gap_path)
            train_val_gap_rows = gap[gap['round'] <= completed].to_dict('records')
            if len(train_val_gap_rows) != completed:
                raise RuntimeError('Training-validation gap file has an unexpected round count.')
            validation_history = pd.read_csv(global_csv).to_dict('records')
            assert len(validation_history) == completed, 'Committed global metrics do not match checkpoint.'
            assert len(pd.read_csv(client_csv)) == completed * NUM_CLIENTS
            assert len(pd.read_csv(selection_csv)) == completed
        else:
            train_val_gap_rows = []
            validation_history = []
        return completed + 1

    def save_round_checkpoint(completed_round):
        cp = {'digest': configuration_digest, 'last_completed_round': int(completed_round), 'global_model_state': {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}, 'best_global_state': best_global_state_cpu, 'best_val_key': tuple((float(x) for x in best_val_key)), 'best_global_round': int(best_global_round), 'selection_history': selection_history.astype(int).tolist()}
        tmp = Path(str(CHECKPOINT_PATH) + '.tmp')
        torch.save(cp, tmp)
        os.replace(tmp, CHECKPOINT_PATH)
    pd.DataFrame([{'positive_weight_cap': 'uncapped', 'focal_gamma': SELECTED_GAMMA, 'tuning_status': 'disabled_preselected_prior_no_ADASYN_run'}]).to_csv(tune_csv, index=False)
    pd.DataFrame([{'selected_positive_weight_cap': 'uncapped', 'selected_focal_gamma': SELECTED_GAMMA, 'selection_source': 'prior_completed_no_ADASYN_run', 'weight_basis': 'each_augmented_local_training_set', 'selection_mode': CLIENT_SELECTION_MODE}]).to_csv(os.path.join(SAVE_DIR, 'selected_focal_hyperparameters.csv'), index=False)
    pd.DataFrame([{'scenario': SCENARIO, 'original_train_count': int(sum(manifest['original_train_counts'])), 'original_training_subset_count': sum(client_real_training_counts), 'local_validation_count': len(y_validation), 'original_test_count': len(y_test), 'synthetic_positive_count': sum((m['synthetic_count'] for m in client_meta)), 'source_index_assignment': 'complete_nonoverlapping_train_and_validation', 'scale_status': 'already_scaled_in_saved_preparation; no_transform_applied', 'scaler_source': 'training_only_saved_per_scenario', 'input_fingerprint': input_fingerprint, 'validation_positive_rate': float(y_validation.mean()), 'test_positive_rate': float(y_test.mean())}]).to_csv(os.path.join(SAVE_DIR, 'data_split_audit.csv'), index=False)
    seed_everything(EXPERIMENT_SEED, device)
    global_model = LSTMContrastiveClassifier(input_dim).to(device)
    validation_history = []
    best_global_state_cpu = None
    best_val_key = (-float('inf'), -float('inf'))
    best_global_round = None
    selection_history = np.zeros(NUM_CLIENTS, dtype=int)
    train_val_gap_rows = []
    start_round = configure_resume()
    for rnd in range(start_round, GLOBAL_ROUNDS + 1):
        synchronize_device(device)
        round_start_time = time.perf_counter()
        global_state_cpu = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        local_states, client_metrics_list, client_sample_counts = ([], [], [])
        for cid, (X_tr, X_val, y_tr, y_val) in enumerate(client_splits):
            state, metrics, losses = train_one_client(global_state_cpu, X_tr, y_tr, X_val, y_val, SELECTED_WEIGHT_CAP, SELECTED_GAMMA, LOCAL_EPOCHS, BATCH_SIZE, EXPERIMENT_SEED + 10000 * rnd + cid, device)
            meta = client_meta[cid]
            row = {'round': rnd, 'client_id': cid + 1, 'num_samples': meta['original_count'], 'pos_samples': meta['original_pos'], 'neg_samples': meta['original_neg'], 'data_ratio_percent': 100.0 * meta['original_pos'] / meta['original_count'], 'local_train_samples': len(y_tr), 'local_train_real_samples': meta['real_train_count'], 'local_train_synthetic_samples': meta['synthetic_count'], 'fedavg_weight_samples': client_real_training_counts[cid], 'local_val_samples': len(y_val), 'local_train_pos_weight': losses['pos_weight'], 'positive_weight_cap': 'uncapped' if SELECTED_WEIGHT_CAP is None else SELECTED_WEIGHT_CAP, 'focal_gamma': SELECTED_GAMMA, 'was_selected': 0, 'selection_rank': '', 'selection_score': selection_score(metrics), 'client_total_loss': losses['total'], 'client_classification_loss': losses['classification'], 'client_contrastive_loss': losses['contrastive'], **metrics, 'client_runtime_sec': losses['duration'], 'local_epochs': LOCAL_EPOCHS}
            local_states.append(state)
            client_metrics_list.append(row)
            client_sample_counts.append(client_real_training_counts[cid])
        scores = np.asarray([row['selection_score'] for row in client_metrics_list])
        selected_clients_idx = choose_client_updates(scores, selection_history, rnd)
        selection_history[selected_clients_idx] += 1
        selected_set = set(selected_clients_idx.tolist())
        dropped_idx = [i for i in range(NUM_CLIENTS) if i not in selected_set]
        for rank, idx in enumerate(selected_clients_idx):
            client_metrics_list[idx]['was_selected'] = 1
            client_metrics_list[idx]['selection_rank'] = rank + 1
        with open(client_csv, 'a', newline='') as handle:
            writer = csv.writer(handle)
            for row in client_metrics_list:
                writer.writerow([row.get(h, '') for h in client_headers])
        with open(selection_csv, 'a', newline='') as handle:
            csv.writer(handle).writerow([rnd, [int(i) + 1 for i in selected_clients_idx], [i + 1 for i in dropped_idx], np.mean([client_metrics_list[i]['client_total_loss'] for i in selected_clients_idx]), np.mean([client_metrics_list[i]['client_total_loss'] for i in dropped_idx]), CLIENT_SELECTION_MODE])
        merged = weighted_fedavg(local_states, client_sample_counts, selected_clients_idx)
        global_model.load_state_dict(merged)
        global_metrics, _ = evaluate_model(global_model, X_validation, y_validation)
        train_monitor_metrics = compute_metrics(y_train_monitor, model_probs(global_model, X_train_monitor))
        train_val_gap_rows.append({'round': rnd, 'training_subset_size': len(y_train_monitor), 'training_subset_f1': train_monitor_metrics['f1'], 'validation_f1': global_metrics['f1'], 'f1_train_minus_validation': train_monitor_metrics['f1'] - global_metrics['f1'], 'training_subset_pr_auc': train_monitor_metrics['pr_auc'], 'validation_pr_auc': global_metrics['pr_auc'], 'pr_auc_train_minus_validation': train_monitor_metrics['pr_auc'] - global_metrics['pr_auc'], 'training_subset_brier': train_monitor_metrics['brier'], 'validation_brier': global_metrics['brier']})
        synchronize_device(device)
        round_elapsed_sec = time.perf_counter() - round_start_time
        selected_weights = np.asarray([client_sample_counts[i] for i in selected_clients_idx], dtype=float)
        selected_rows = [client_metrics_list[i] for i in selected_clients_idx]
        row = {'round': rnd, 'evaluation_split': 'fixed_client_validation', 'global_total_loss': float(np.average([r['client_total_loss'] for r in selected_rows], weights=selected_weights)), 'global_classification_loss': float(np.average([r['client_classification_loss'] for r in selected_rows], weights=selected_weights)), 'global_contrastive_loss': float(np.average([r['client_contrastive_loss'] for r in selected_rows], weights=selected_weights)), 'global_pos_weight': float(np.average([r['local_train_pos_weight'] for r in selected_rows], weights=selected_weights)), 'positive_weight_cap': 'uncapped' if SELECTED_WEIGHT_CAP is None else SELECTED_WEIGHT_CAP, 'focal_gamma': SELECTED_GAMMA, **global_metrics, 'round_runtime_sec': round_elapsed_sec, 'num_selected_clients': len(selected_clients_idx)}
        validation_history.append(row)
        val_key = (float(global_metrics['f1']), float(global_metrics['pr_auc']))
        if val_key > best_val_key:
            best_val_key = val_key
            best_global_round = rnd
            best_global_state_cpu = {k: v.detach().cpu().clone() for k, v in global_model.state_dict().items()}
        with open(global_csv, 'a', newline='') as handle:
            csv.writer(handle).writerow([row.get(h, '') for h in global_headers])
        pd.DataFrame(train_val_gap_rows).to_csv(os.path.join(SAVE_DIR, 'training_validation_gap.csv'), index=False)
        save_round_checkpoint(rnd)
        print(f"Round {rnd:02d}/{GLOBAL_ROUNDS}: validation F1={global_metrics['f1']:.4f}, recall={global_metrics['recall']:.4f}, AP={global_metrics['pr_auc']:.4f}, selected clients={[int(i) + 1 for i in selected_clients_idx]}", flush=True)
    pd.DataFrame(train_val_gap_rows).to_csv(os.path.join(SAVE_DIR, 'training_validation_gap.csv'), index=False)
    assert best_global_state_cpu is not None
    global_model.load_state_dict(best_global_state_cpu)
    pd.DataFrame([{'selected_round': best_global_round, 'selected_validation_f1': best_val_key[0], 'selected_validation_pr_auc': best_val_key[1], 'criterion': 'max_validation_F1_tiebreak_PR_AUC'}]).to_csv(os.path.join(SAVE_DIR, 'best_global_checkpoint_selection.csv'), index=False)
    print('Best global checkpoint from round', best_global_round, 'val F1=', best_val_key[0])
    client_table = pd.read_csv(client_csv)
    participation = client_table.groupby('client_id').agg(times_selected=('was_selected', 'sum'), mean_validation_f1=('f1', 'mean'), mean_validation_pr_auc=('pr_auc', 'mean'), positive_samples=('pos_samples', 'first'), negative_samples=('neg_samples', 'first')).reset_index()
    participation['selection_frequency'] = participation['times_selected'] / GLOBAL_ROUNDS
    participation.to_csv(os.path.join(SAVE_DIR, 'client_participation_summary.csv'), index=False)
    print('Client participation summary:')
    print(participation.to_string(index=False))
    counts = participation['times_selected'].to_numpy(dtype=float)
    jain_index = counts.sum() ** 2 / (len(counts) * (counts @ counts)) if counts @ counts > 0 else float('nan')
    pd.DataFrame([{'client_selection_mode': CLIENT_SELECTION_MODE, 'rounds': GLOBAL_ROUNDS, 'clients': NUM_CLIENTS, 'selected_each_round': NUM_SELECTED_CLIENTS, 'min_client_selections': int(counts.min()), 'max_client_selections': int(counts.max()), 'never_selected_clients': int((counts == 0).sum()), 'jain_participation_index': jain_index}]).to_csv(os.path.join(SAVE_DIR, 'participation_fairness_summary.csv'), index=False)
    print('Jain participation index:', round(jain_index, 4))
    return (global_model, best_global_round, SAVE_DIR, global_csv)
