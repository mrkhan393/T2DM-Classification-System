import os, copy
import pandas as pd
import torch
from config import EXPERIMENT_SEED, GLOBAL_ROUNDS, SCENARIO
from evaluation.metrics import evaluate_model

def run_final_evaluation(data, result):
    manifest, client_splits, client_real_train, client_real_training_counts, client_meta, X_validation, y_validation, X_test_scaled, y_test, X_test_balanced_scaled, y_test_balanced, holdout_indices, input_fingerprint = data
    global_model, best_global_round, SAVE_DIR, global_csv = result
    assert len(pd.read_csv(global_csv)) == GLOBAL_ROUNDS, 'Run must finish 50 rounds before testing.'
    original_model = copy.deepcopy(global_model).cpu().eval()
    torch.save(original_model.state_dict(), os.path.join(SAVE_DIR, 'final_original_model.pth'))
    test_metrics, test_probs = evaluate_model(original_model, X_test_scaled, y_test)
    diagnostic_metrics, diagnostic_probs = evaluate_model(original_model, X_test_balanced_scaled, y_test_balanced)

    def save_metric_row(metrics, split_name, filename):
        row = {'scenario': SCENARIO, 'seed': EXPERIMENT_SEED, 'selected_round': best_global_round, 'evaluation_split': split_name, **{k: str(v) if isinstance(v, (list, tuple)) else v for k, v in metrics.items()}}
        pd.DataFrame([row]).to_csv(os.path.join(SAVE_DIR, filename), index=False)
    save_metric_row(test_metrics, 'untouched_natural_prevalence_test', 'final_natural_test_metrics.csv')
    save_metric_row(diagnostic_metrics, 'supplementary_balanced_subset_from_natural_test', 'final_balanced_diagnostic_metrics.csv')
    pd.DataFrame({'true_label': y_test, 'predicted_probability': test_probs}).to_csv(os.path.join(SAVE_DIR, 'final_natural_test_predictions.csv'), index=False)
    pd.DataFrame({'true_label': y_test_balanced, 'predicted_probability': diagnostic_probs, 'index_in_natural_test': holdout_indices}).to_csv(os.path.join(SAVE_DIR, 'final_balanced_diagnostic_predictions.csv'), index=False)
    print('BEST VALIDATION ROUND:', best_global_round)
    print('FINAL NATURAL TEST:', test_metrics)
    print('SECONDARY BALANCED DIAGNOSTIC:', diagnostic_metrics)
    print('The test observations were not used during model selection or client ranking.')
