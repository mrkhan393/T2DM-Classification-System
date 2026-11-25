import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from config import CODE_DIR, FEATURE_ORDER, NUM_CLIENTS, MIN_POS_PER_CLIENT, LOCAL_VALIDATION_FRACTION, OUTER_TEST_FRACTION, SEVERE_SIZE_MULTIPLIERS
from partitioning.common import _validate_allocation
from partitioning.iid import split_iid_original
from partitioning.mild_non_iid import split_mild_original
from partitioning.severe_non_iid import split_severe_original

DEFAULT_CSV = CODE_DIR / "data" / "india_clean_file.csv"
DEFAULT_OUTPUT = CODE_DIR / "data" / "original_partitions"
TARGET_NAME = "final_diabetes"

def _model_shape(X):
    return np.asarray(X, dtype=np.float32).reshape(-1, len(FEATURE_ORDER), 1)

def _sha256_arrays(*arrays):
    digest = hashlib.sha256()
    for array in arrays:
        array = np.ascontiguousarray(array)
        digest.update(str(array.shape).encode())
        digest.update(str(array.dtype).encode())
        digest.update(memoryview(array).cast('B'))
    return digest.hexdigest()

def prepare_original_partitions(csv_path=DEFAULT_CSV, output_dir=DEFAULT_OUTPUT, seed=42, dry_run=False, save_raw_clients=True):
    csv_path = Path(csv_path).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve()
    if not csv_path.is_file():
        raise FileNotFoundError(f'Original cleaned CSV not found: {csv_path}')
    if not dry_run and root.exists() and any(root.iterdir()):
        raise FileExistsError(f'Output directory is not empty: {root}. Choose a new one.')
    df = pd.read_csv(csv_path)
    missing = [c for c in FEATURE_ORDER + [TARGET_NAME] if c not in df.columns]
    if missing:
        raise KeyError(f'CSV is missing required columns: {missing}')
    X = df[FEATURE_ORDER].to_numpy(dtype=np.float32)
    y = df[TARGET_NAME].to_numpy(dtype=np.int64)
    del df
    if len(X) == 0 or not np.isin(y, [0, 1]).all() or (not np.isfinite(X).all()):
        raise ValueError('CSV must contain finite predictors and binary 0/1 labels')
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=OUTER_TEST_FRACTION, stratify=y, random_state=seed)
    del X, y
    scenarios = {'iid': split_iid_original(y_train, seed), 'mild_non_iid': split_mild_original(y_train, seed), 'severe_non_iid': split_severe_original(y_train, seed)}
    print('Complete original cohort:', len(y_train) + len(y_test))
    print('Outer original train counts (negative, positive):', np.bincount(y_train, minlength=2).tolist())
    print('Untouched global test counts (negative, positive):', np.bincount(y_test, minlength=2).tolist())
    if not dry_run:
        root.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(root / 'global_test_original_unscaled.npz', X_test=_model_shape(X_test), y_test=y_test, is_scaled=np.array(False))
    rows = []
    reports = []
    for scenario, original_indices in scenarios.items():
        _validate_allocation(original_indices, len(y_train))
        train_val = []
        for client_id, ids in enumerate(original_indices):
            tr, val = train_test_split(ids, test_size=LOCAL_VALIDATION_FRACTION, stratify=y_train[ids], random_state=seed + client_id)
            train_val.append((np.asarray(tr, dtype=np.int64), np.asarray(val, dtype=np.int64)))
        full_train_ids = np.concatenate([t for t, _ in train_val])
        full_val_ids = np.concatenate([v for _, v in train_val])
        _validate_allocation([full_train_ids, full_val_ids], len(y_train))
        assert np.intersect1d(full_train_ids, full_val_ids).size == 0
        scenario_dir = root / scenario
        if not dry_run:
            scenario_dir.mkdir(parents=True, exist_ok=True)
            scaler = StandardScaler().fit(X_train[full_train_ids])
            joblib.dump(scaler, scenario_dir / 'standard_scaler_real_training_only.joblib')
            x_test_scaled = scaler.transform(X_test).astype(np.float32)
            np.savez_compressed(scenario_dir / 'global_test_scaled.npz', X_test=_model_shape(x_test_scaled), y_test=y_test, is_scaled=np.array(True), source=np.array('original_untouched_global_test'), scaler_file=np.array('standard_scaler_real_training_only.joblib'))
        else:
            scaler = None
        scenario_rows = []
        for client_id, (ids, (tr, val)) in enumerate(zip(original_indices, train_val), start=1):
            y_client = y_train[ids]
            y_tr = y_train[tr]
            y_val = y_train[val]
            row = {'scenario': scenario, 'client_id': client_id, 'original_client_total': int(len(ids)), 'original_client_negative': int(np.sum(y_client == 0)), 'original_client_positive': int(np.sum(y_client == 1)), 'original_client_positive_rate': float(np.mean(y_client == 1)), 'real_training_samples': int(len(tr)), 'real_training_negative': int(np.sum(y_tr == 0)), 'real_training_positive': int(np.sum(y_tr == 1)), 'original_validation_samples': int(len(val)), 'original_validation_negative': int(np.sum(y_val == 0)), 'original_validation_positive': int(np.sum(y_val == 1)), 'synthetic_samples': 0}
            rows.append(row)
            scenario_rows.append(row)
            if not dry_run:
                if save_raw_clients:
                    np.savez_compressed(scenario_dir / f'client_{client_id:02d}_raw_original.npz', X_client=_model_shape(X_train[ids]), y_client=y_client, source_index_client=ids, is_scaled=np.array(False), is_synthetic=np.zeros(len(ids), dtype=bool))
                np.savez_compressed(scenario_dir / f'client_{client_id:02d}_original.npz', X_train=_model_shape(scaler.transform(X_train[tr])), y_train=y_tr, source_index_train=tr, X_val=_model_shape(scaler.transform(X_train[val])), y_val=y_val, source_index_val=val, source_index_client=ids, is_scaled=np.array(True), is_synthetic=np.zeros(len(tr), dtype=bool))
        scenario_df = pd.DataFrame(scenario_rows)
        if not dry_run:
            scenario_df.to_csv(scenario_dir / 'client_distribution_summary.csv', index=False)
            (scenario_dir / 'scenario_info.json').write_text(json.dumps({'scenario': scenario, 'data_source': 'original_only', 'resampling': 'none', 'all_original_training_observations_assigned_once': True, 'n_clients': NUM_CLIENTS, 'original_client_size_range': [int(scenario_df.original_client_total.min()), int(scenario_df.original_client_total.max())], 'original_client_positive_rate_range': [float(scenario_df.original_client_positive_rate.min()), float(scenario_df.original_client_positive_rate.max())], 'severe_distribution_note': 'Combined quantity and label heterogeneity' if scenario == 'severe_non_iid' else None, 'iid_distribution_note': 'Stratified label-IID approximation; full feature-IID not established' if scenario == 'iid' else None, 'scaler': 'standard_scaler_real_training_only.joblib', 'test': 'global_test_scaled.npz'}, indent=2), encoding='utf-8')
        print(f'\n{scenario.upper()}')
        print(scenario_df[['client_id', 'original_client_total', 'original_client_positive', 'original_client_positive_rate', 'real_training_samples', 'original_validation_samples']].to_string(index=False, formatters={'original_client_positive_rate': lambda r: f'{100 * r:.4f}%'}))
        reports.append({'scenario': scenario, 'original_source_records': len(y_train), 'training_records': len(full_train_ids), 'validation_records': len(full_val_ids), 'global_test_records': len(y_test), 'all_original_observations_assigned_once': True, 'training_validation_disjoint': True, 'synthetic_records': 0, 'verified': 'PASS'})
    summary = pd.DataFrame(rows)
    verification = pd.DataFrame(reports)
    if not dry_run:
        summary.to_csv(root / 'partition_summary.csv', index=False)
        verification.to_csv(root / 'partition_integrity_report.csv', index=False)
        (root / 'manifest.json').write_text(json.dumps({'created_utc': datetime.now(timezone.utc).isoformat(), 'source_csv': str(csv_path), 'feature_order': FEATURE_ORDER, 'target': TARGET_NAME, 'seed': int(seed), 'outer_test_fraction': OUTER_TEST_FRACTION, 'local_validation_fraction': LOCAL_VALIDATION_FRACTION, 'original_train_counts': np.bincount(y_train, minlength=2).tolist(), 'original_test_counts': np.bincount(y_test, minlength=2).tolist(), 'train_sha256': _sha256_arrays(X_train, y_train), 'test_sha256': _sha256_arrays(X_test, y_test), 'scenarios': list(scenarios.keys()), 'clients_per_scenario': NUM_CLIENTS, 'minimum_real_positives_per_client': MIN_POS_PER_CLIENT, 'mild_dirichlet_alpha': 2.0, 'severe_size_multipliers': list(SEVERE_SIZE_MULTIPLIERS), 'severe_residual_positive_allocation': [0.75, 0.25], 'preprocessing': 'One pooled training-only StandardScaler per scenario', 'sampling_method': 'none', 'synthetic_samples': 0, 'test_data': 'Original 5% test, never resampled'}, indent=2), encoding='utf-8')
    print('\n' + verification.to_string(index=False))
    if not dry_run:
        print(f'\nOriginal-only partitions saved to: {root}')
    return summary

def main():
    parser = argparse.ArgumentParser(description="Prepare original-only DiaFed client partitions")
    parser.add_argument('--csv', type=Path, default=DEFAULT_CSV, help='Original cleaned CSV path')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT, help='New directory for ORIGINAL-only saved data')
    parser.add_argument('--seed', type=int, default=42, help='Partition random seed; 42 reproduces historical experiments')
    parser.add_argument('--dry-run', action='store_true', help='Check and print original allocations without saving')
    parser.add_argument('--skip-raw-clients', action='store_true', help='Do not save additional unscaled full-client copies')
    args = parser.parse_args()
    prepare_original_partitions(csv_path=args.csv, output_dir=args.output, seed=args.seed, dry_run=args.dry_run, save_raw_clients=not args.skip_raw_clients)

if __name__ == "__main__":
    main()
