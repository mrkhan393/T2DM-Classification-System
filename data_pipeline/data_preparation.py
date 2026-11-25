import json, hashlib
import numpy as np
from config import PARTITION_ROOT, SCENARIO_DIR, SCENARIO, FEATURE_ORDER, NUM_CLIENTS, OUTPUT_PATH

def sha256_file(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

def load_data():
    if not SCENARIO_DIR.is_dir():
        raise FileNotFoundError(f'Missing prepared scenario directory: {SCENARIO_DIR}')
    if not (PARTITION_ROOT / 'manifest.json').is_file():
        raise FileNotFoundError(f"Missing partition manifest: {PARTITION_ROOT / 'manifest.json'}")
    if not (SCENARIO_DIR / 'scenario_info.json').is_file():
        raise FileNotFoundError('Missing scenario_info.json')
    print('Selected scenario:', SCENARIO)
    print('Prepared inputs:', SCENARIO_DIR)
    print('Output location:', OUTPUT_PATH)
    manifest = json.loads((PARTITION_ROOT / 'manifest.json').read_text())
    scenario_info = json.loads((SCENARIO_DIR / 'scenario_info.json').read_text())
    assert manifest['feature_order'] == FEATURE_ORDER, 'Feature order differs from the saved preparation script.'
    assert manifest['n_clients'] == NUM_CLIENTS, 'Expected 10 saved clients.'
    assert scenario_info['scenario'] == SCENARIO, 'Scenario metadata mismatch.'
    assert scenario_info['categorical_policy'] == 'none', 'Expected vanilla ADASYN; report other synthetic methods separately.'
    client_splits = []
    client_real_train = []
    client_real_training_counts = []
    client_aug_training_counts = []
    client_meta = []
    all_real_source_ids = []
    all_val_source_ids = []
    input_filenames = [PARTITION_ROOT / 'manifest.json', SCENARIO_DIR / 'scenario_info.json']
    for cid in range(1, NUM_CLIENTS + 1):
        original_path = SCENARIO_DIR / f'client_{cid:02d}_original.npz'
        adasyn_path = SCENARIO_DIR / f'client_{cid:02d}_adasyn.npz'
        input_filenames.extend([original_path, adasyn_path])
        with np.load(original_path, allow_pickle=False) as orig, np.load(adasyn_path, allow_pickle=False) as aug:
            X_real = orig['X_train'].astype(np.float32)
            y_real = orig['y_train'].astype(np.int64)
            X_aug = aug['X_train'].astype(np.float32)
            y_aug = aug['y_train'].astype(np.int64)
            X_val = aug['X_val'].astype(np.float32)
            y_val = aug['y_val'].astype(np.int64)
            source_train = aug['source_index_train'].astype(np.int64)
            source_val = aug['source_index_val'].astype(np.int64)
            original_source_client = orig['source_index_client'].astype(np.int64)
            synthetic_flags = aug['is_synthetic'].astype(bool)
            assert bool(orig['is_scaled'].item()) and bool(aug['is_scaled'].item()), 'Inputs are not marked as scaled.'
            assert X_aug.ndim == X_real.ndim == X_val.ndim == 3
            assert X_aug.shape[1:] == X_real.shape[1:] == X_val.shape[1:] == (16, 1)
            assert len(X_aug) == len(y_aug) == len(synthetic_flags) == len(source_train)
            assert len(X_real) == len(y_real) and len(X_val) == len(y_val) == len(source_val)
            assert np.array_equal(X_aug[:len(X_real)], X_real), 'ADASYN altered original training features.'
            assert np.array_equal(y_aug[:len(y_real)], y_real), 'ADASYN altered original labels.'
            assert np.array_equal(X_val, orig['X_val']) and np.array_equal(y_val, orig['y_val']), 'Original/ADASYN validation mismatch.'
            assert not synthetic_flags[:len(y_real)].any() and synthetic_flags[len(y_real):].all()
            assert np.all(y_aug[synthetic_flags] == 1), 'Found non-positive synthetic rows.'
            assert np.array_equal(source_train[:len(y_real)], orig['source_index_train'])
            assert np.all(source_train[synthetic_flags] == -1)
            assert np.array_equal(source_val, orig['source_index_val'])
            assert np.isfinite(X_aug).all() and np.isfinite(X_val).all()
            assert set(np.unique(y_aug)).issubset({0, 1}) and len(np.unique(y_val)) == 2
            assert np.intersect1d(source_train[~synthetic_flags], source_val).size == 0
            assert len(np.unique(original_source_client)) == len(original_source_client)
            assert len(original_source_client) == len(y_real) + len(y_val)
            assert len(synthetic_flags) - len(y_real) > 0
            all_real_source_ids.append(source_train[~synthetic_flags])
            all_val_source_ids.append(source_val)
            client_splits.append((X_aug, X_val, y_aug, y_val))
            client_real_train.append((X_real, y_real))
            client_real_training_counts.append(len(y_real))
            client_aug_training_counts.append(len(y_aug))
            client_meta.append({'original_count': len(original_source_client), 'original_pos': int(y_real.sum() + y_val.sum()), 'original_neg': int(len(original_source_client) - y_real.sum() - y_val.sum()), 'real_train_count': len(y_real), 'synthetic_count': int(synthetic_flags.sum()), 'augmented_pos_count': int(y_aug.sum())})
    original_source_ids = np.concatenate(all_real_source_ids + all_val_source_ids)
    expected_orig = int(sum(manifest['original_train_counts']))
    assert len(original_source_ids) == expected_orig
    assert np.array_equal(np.sort(original_source_ids), np.arange(expected_orig)), 'Missing or repeated original source indices.'
    assert sum((m['synthetic_count'] for m in client_meta)) == scenario_info['total_synthetic_positive']
    assert sum((m['original_count'] for m in client_meta)) == expected_orig
    test_path = SCENARIO_DIR / 'global_test_scaled.npz'
    diagnostic_path = SCENARIO_DIR / 'balanced_diagnostic_scaled.npz'
    input_filenames.extend([test_path, diagnostic_path])
    with np.load(test_path, allow_pickle=False) as obj:
        assert bool(obj['is_scaled'].item())
        X_test_scaled = obj['X_test'].astype(np.float32)
        y_test = obj['y_test'].astype(np.int64)
    with np.load(diagnostic_path, allow_pickle=False) as obj:
        assert bool(obj['is_scaled'].item())
        X_test_balanced_scaled = obj['X_test'].astype(np.float32)
        y_test_balanced = obj['y_test'].astype(np.int64)
        holdout_indices = obj['index_in_global_test'].astype(np.int64)
    assert X_test_scaled.shape[1:] == X_test_balanced_scaled.shape[1:] == (16, 1)
    assert len(y_test) == int(sum(manifest['original_test_counts']))
    assert np.array_equal(np.bincount(y_test, minlength=2), manifest['original_test_counts'])
    assert np.array_equal(X_test_balanced_scaled, X_test_scaled[holdout_indices])
    assert np.array_equal(y_test_balanced, y_test[holdout_indices])
    assert np.all(np.bincount(y_test_balanced, minlength=2) == [486, 486]), 'Expected 486/486 diagnostic subset.'
    assert np.isfinite(X_test_scaled).all()
    X_validation = np.concatenate([s[1] for s in client_splits], axis=0)
    y_validation = np.concatenate([s[3] for s in client_splits], axis=0)
    assert np.array_equal(np.bincount(y_validation, minlength=2) + np.sum([np.bincount(y, minlength=2) for x, y in client_real_train], axis=0), manifest['original_train_counts'])
    input_file_hashes = {str(p.relative_to(PARTITION_ROOT)): sha256_file(p) for p in input_filenames}
    input_fingerprint = hashlib.sha256(json.dumps(input_file_hashes, sort_keys=True).encode()).hexdigest()
    print('Verified original assignment:', f'{expected_orig:,}', 'real observations exactly once')
    print('Source-scaled 10 clients:', len(client_splits), '| Synthetic positives:', sum((m['synthetic_count'] for m in client_meta)))
    print('Original-only validation:', len(y_validation), '| Natural test:', len(y_test))
    print('Natural test prevalence:', f'{y_test.mean():.4%}', '| Balanced diagnostic:', len(y_test_balanced))
    print('Input fingerprint (SHA-256):', input_fingerprint[:20] + '...')
    aug_all = np.concatenate([s[2] for s in client_splits])
    original_train_counts = np.asarray(manifest['original_train_counts'], dtype=int)
    for label, n0, n1 in [('Original complete train+local validation', original_train_counts[0], original_train_counts[1]), ('ADASYN-augmented local training ONLY', int((aug_all == 0).sum()), int((aug_all == 1).sum())), ('Genuine pooled local validation', int((y_validation == 0).sum()), int((y_validation == 1).sum())), ('Untouched natural test', int((y_test == 0).sum()), int((y_test == 1).sum())), ('Balanced diagnostic subset', int((y_test_balanced == 0).sum()), int((y_test_balanced == 1).sum()))]:
        print(f'{label}: negative={n0:,}, positive={n1:,}, pos_rate={n1 / (n0 + n1):.4%}')
    del aug_all
    print(f'Prepared {NUM_CLIENTS} clients for {SCENARIO}:')
    for c, (Xtr, Xval, ytr, yval) in enumerate(client_splits, 1):
        meta = client_meta[c - 1]
        print(f"C{c:02d}: original={meta['original_count']:,}, real_train={meta['real_train_count']:,}, ADASYN_train={len(ytr):,}, synthetic={meta['synthetic_count']:,}, val={len(yval):,}, augmented_positive_rate={ytr.mean():.3%}")
    return (manifest, client_splits, client_real_train, client_real_training_counts, client_meta, X_validation, y_validation, X_test_scaled, y_test, X_test_balanced_scaled, y_test_balanced, holdout_indices, input_fingerprint)
