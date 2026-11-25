# DiaFed

Federated BiLSTM classification using saved 5% ADASYN client partitions.

## Local setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python main.py
```

Place the existing prepared dataset in `data/`, with `manifest.json` at its root. Each scenario folder (`mild_non_iid`, `iid`, `severe_non_iid`) must contain `scenario_info.json`, ten `client_01_original.npz` through `client_10_original.npz` files, ten matching `client_XX_adasyn.npz` files, `global_test_scaled.npz` and `balanced_diagnostic_scaled.npz`.

Alternatively use the existing dataset location:

```bash
DIAFED_PARTITION_ROOT="/path/to/prepared datasets" DIAFED_SCENARIO=mild_non_iid python main.py
```

The default is 50 rounds, 10 clients, 5 local epochs and top-five aggregation. Settings are in `config.py`. Device selection is automatic; override with `DIAFED_DEVICE=cpu`, `mps` or `cuda`. Results are stored in `results/` unless `DIAFED_OUTPUT_DIR` is set. Rerun with the same settings and output directory to resume. Optional compression is disabled by default.

The data loader does not regenerate ADASYN or fit a scaler. Prepared datasets and generated results are excluded from Git. Do not publish participant-level records or prediction files without appropriate authorization.

## Generate original-only client partitions

```bash
python -m data_pipeline.prepare_dataset --csv data/india_clean_file.csv --output data/original_partitions
```

Use `--dry-run` to inspect distributions without saving. All three scenarios are generated independently from the same stratified 95:5 outer split. The local split is 80:20. A separate scaler per scenario is fitted on pooled genuine local training records only. The three allocation algorithms live in `partitioning/iid.py`, `partitioning/mild_non_iid.py`, and `partitioning/severe_non_iid.py`, with shared allocation helpers in `partitioning/common.py`.

The supplied partitioning source explicitly excludes ADASYN. Its output is not directly compatible with the existing ADASYN training loader: the original ADASYN preparation step and balanced diagnostic subset are still required. Keep original-only outputs separate from the existing augmented datasets. The original-only output retains its original metadata schema.

## Source layout

| Package | Responsibility |
|---|---|
| `data_pipeline/` | Original dataset preparation and prepared-data loading |
| `partitioning/` | IID, mild non-IID, severe non-IID and shared allocation helpers |
| `models/` | Network architecture and training losses |
| `federated/` | Local training, round orchestration and aggregation |
| `evaluation/` | Metrics and final evaluation |
| `optimization/` | Optional model compression |
| `utils/` | Device selection, checkpoints and output audit |

Run commands from the repository root. `main.py` and `config.py` remain at the root.
