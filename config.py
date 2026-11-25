import os
from pathlib import Path
os.environ.setdefault('PYTORCH_ENABLE_MPS_FALLBACK', '1')
CODE_DIR = Path(__file__).resolve().parent
PARTITION_ROOT = Path(os.environ.get('DIAFED_PARTITION_ROOT', str(CODE_DIR / 'data')))
SCENARIO = os.environ.get('DIAFED_SCENARIO', 'mild_non_iid').strip()
assert SCENARIO in {'mild_non_iid', 'iid', 'severe_non_iid'}, SCENARIO
SCENARIO_DIR = PARTITION_ROOT / SCENARIO
SEED = 42
OUTPUT_PATH = Path(os.environ.get('DIAFED_OUTPUT_DIR', str(CODE_DIR / 'results' / f'ADASYN5_{SCENARIO}_seed{SEED}')))
FEATURE_ORDER = ['hv005', 'hv021', 'hv023', 'hv105', 'hv115', 'hv104', 'hv106', 'hml18', 'shb70', 'ha53', 'shb13', 'hv206', 'hv116', 'avg_sys', 'avg_dia', 'bmi']
features = FEATURE_ORDER[:]
NUM_CLIENTS = 10
GLOBAL_ROUNDS = 50
LOCAL_EPOCHS = 5
BATCH_SIZE = 256
LOCAL_LR = 0.0001
ALPHA = 0.5
GRAD_CLIP_NORM = 1.0
NUM_SELECTED_CLIENTS = 5
CLIENT_SELECTION_MODE = 'performance_top5'
EVAL_BATCH_SIZE = 1024
TRAIN_MONITOR_SIZE = 8192
FINAL_SPARSITY = 0.5
RUN_COMPRESSION_EVAL = False
RUN_INFERENCE_BENCHMARK = False
EXPERIMENT_SEED = SEED
RUN_FOCAL_TUNING = False
SELECTED_WEIGHT_CAP = None
SELECTED_GAMMA = 1.0
input_dim = 1

MIN_POS_PER_CLIENT = 700
LOCAL_VALIDATION_FRACTION = 0.20
OUTER_TEST_FRACTION = 0.05
SEVERE_SIZE_MULTIPLIERS = [0.35, 0.50, 0.65, 0.75, 0.90, 1.10, 1.25, 1.40, 1.50, 1.60]
