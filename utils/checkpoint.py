import os, json, csv
from pathlib import Path

def atomic_json_save(obj, target):
    tmp = Path(str(target) + '.tmp')
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True), encoding='utf-8')
    os.replace(tmp, target)

def trim_csv_to_round(path, keep_round, headers):
    path = Path(path)
    if not path.exists():
        if keep_round > 0:
            raise FileNotFoundError('Expected checkpointed training log: ' + str(path))
        with open(path, 'w', newline='') as f:
            csv.writer(f).writerow(headers)
        return
    with open(path, 'r', newline='') as f:
        reader = csv.DictReader(f)
        if reader.fieldnames != headers:
            raise RuntimeError(f'CSV schema changed in {path}; do not resume with different code.')
        retained = []
        for row in reader:
            try:
                rnd = int(row['round'])
            except (ValueError, KeyError, TypeError):
                continue
            if 1 <= rnd <= keep_round:
                retained.append(row)
    if keep_round and (not retained):
        raise RuntimeError('Checkpoint exists but no committed rows found: ' + str(path))
    temp = Path(str(path) + '.tmp')
    with open(temp, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        writer.writerows(retained)
    os.replace(temp, path)
