import pandas as pd
from config import GLOBAL_ROUNDS, NUM_CLIENTS, RUN_COMPRESSION_EVAL, SCENARIO

def audit_outputs(data, result):
    SAVE_DIR = result[2]
    y_test = data[8]
    from pathlib import Path
    expected = {'client_metrics.csv': GLOBAL_ROUNDS * NUM_CLIENTS, 'global_validation_metrics.csv': GLOBAL_ROUNDS, 'selection_log.csv': GLOBAL_ROUNDS, 'client_participation_summary.csv': NUM_CLIENTS, 'selected_focal_hyperparameters.csv': 1, 'data_split_audit.csv': 1, 'best_global_checkpoint_selection.csv': 1, 'hardware_runtime_info.csv': 1, 'participation_fairness_summary.csv': 1, 'training_validation_gap.csv': GLOBAL_ROUNDS, 'final_natural_test_metrics.csv': 1, 'final_balanced_diagnostic_metrics.csv': 1, 'focal_hyperparameter_search.csv': 1}
    if RUN_COMPRESSION_EVAL:
        expected.update({'quantization_backend_details.csv': 4})
    for filename, count in expected.items():
        path = Path(SAVE_DIR) / filename
        assert path.is_file(), f'Missing CSV: {path}'
        actual = len(pd.read_csv(path))
        assert actual == count, f'{filename}: {actual} rows, expected {count}'
        print(filename, ':', actual, 'rows')
    for filename in ['final_natural_test_predictions.csv', 'final_balanced_diagnostic_predictions.csv', 'final_original_model.pth', 'resume_checkpoint.pt', 'run_config.json']:
        file = Path(SAVE_DIR) / filename
        assert file.is_file() and file.stat().st_size > 0, f'Missing/empty: {file}'
    if RUN_COMPRESSION_EVAL:
        for filename in ['global_model_comparison.csv', 'model_variant_test_probabilities.csv', 'inference_latency_benchmark.csv']:
            file = Path(SAVE_DIR) / filename
            assert file.is_file() and file.stat().st_size > 0, f'Missing/empty: {file}'
    assert len(pd.read_csv(Path(SAVE_DIR) / 'final_natural_test_predictions.csv')) == len(y_test)
    print('All 50 rounds and final files verified. Scenario:', SCENARIO)
