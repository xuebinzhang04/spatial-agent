#!/usr/bin/env python3
"""Consolidate completed MindCube samples and regenerate SpatialBench metric exports."""
import csv
import importlib.util
import json
import shutil
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / '05_Experiment/results/spatialbench_qwen3.8'
FINAL = BASE / 'final-20261005'
DEST = FINAL / 'benchmarks/qwen3.8-mindcube'

def dump(path, data):
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + '\n')

def main():
    sources = [ROOT / '05_Experiment/results/qwen3.8-mindcube', BASE / 'qwen3.8-mindcube']
    candidates = list(DEST.glob('Qwen3.8-27B/*samples_mindcube_full.jsonl'))
    candidates += [p for source in sources for p in source.glob('Qwen3.8-27B/*samples_mindcube_full.jsonl')]
    if len(candidates) != 1:
        raise ValueError(f'Expected one sample file, found {candidates}')
    sample = candidates[0]
    rows = [json.loads(line) for line in sample.open()]
    assert len(rows) == 21154 and {r['doc_id'] for r in rows} == set(range(21154))
    spec = importlib.util.spec_from_file_location('mindcube_utils', ROOT / '03_SpatialBench/lmms-eval/lmms_eval/tasks/mindcube/utils.py')
    utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(utils)
    # Independently verify persisted scores against the official answer parser.
    for row in rows:
        response = row['filtered_resps']
        assert isinstance(response, str)
        expected = float(utils.extract_answer(response) == utils.extract_answer(row['target']))
        assert row['overall_accuracy']['score'] == expected
        for category in ('among', 'around', 'rotation'):
            assert row[f'{category}_accuracy']['score'] == expected
    scores = {'overall_accuracy': utils.mindcube_aggregate_results([r['overall_accuracy'] for r in rows])}
    counts = Counter(r['among_accuracy']['type'] for r in rows)
    correct = Counter()
    for row in rows:
        correct[row['among_accuracy']['type']] += int(row['overall_accuracy']['score'])
    for category in ('among', 'around', 'rotation'):
        scores[f'{category}_accuracy'] = getattr(utils, f'mindcube_aggregate_{category}_results')([r[f'{category}_accuracy'] for r in rows])
    assert sum(counts.values()) == 21154 and set(counts) == {'among', 'around', 'rotation'}
    DEST.mkdir(parents=True, exist_ok=True)
    for source in reversed(sources):
        if source.exists():
            for p in source.rglob('*'):
                if p.is_file():
                    target = DEST / p.relative_to(source)
                    target.parent.mkdir(parents=True, exist_ok=True)
                    if target.exists() and target.stat().st_size and not p.stat().st_size:
                        continue
                    if target.exists() and target.stat().st_size and p.stat().st_size:
                        assert target.read_bytes() == p.read_bytes(), f'Conflicting artifacts: {p}'
                    else:
                        shutil.copy2(p, target)
    sample = DEST / 'Qwen3.8-27B' / sample.name
    log = ROOT / '05_Experiment/logs/qwen3.8-mindcube-tp4-344813.log'
    shutil.copy2(log, DEST / 'slurm-344813.log')
    report = {'model_name': 'Qwen3.8-27B', 'results': {'mindcube_full': {k + ',none': v for k, v in scores.items()}}, 'n-samples': {'mindcube_full': {'original': 21154, 'effective': 21154}}, 'group_subtasks': {}, 'higher_is_better': {'mindcube_full': {k: True for k in scores}}, 'reconstruction': {'reason': 'Original aggregate save failed with FileNotFoundError; reconstructed from complete persisted per-sample scores using task utils.py', 'sample_file': str(sample.relative_to(ROOT)), 'verification': '21154 unique contiguous doc_ids; all scores independently verified with official extract_answer', 'category_counts': dict(counts), 'category_correct': dict(correct), 'original_aggregate_available': False}}
    dump(DEST / 'Qwen3.8-27B/20261005_171203_results.json', report)
    dump(FINAL / 'raw_results/mindcube_results.json', report)
    metrics = []
    for p in sorted((FINAL / 'raw_results').glob('*_results.json')):
        data = json.loads(p.read_text())
        for task, values in data['results'].items():
            n = data.get('n-samples', {}).get(task)
            if n is None:
                children = data.get('group_subtasks', {}).get(task, [])
                assert children, task
                n = {key: sum(data['n-samples'][child][key] for child in children) for key in ('original', 'effective')}
            assert n['original'] == n['effective']
            for metric, value in values.items():
                if ',' not in metric or 'stderr' in metric:
                    continue
                name = metric.split(',')[0]
                values_flat = {name + '.' + k: v for k, v in value.items()} if isinstance(value, dict) else {name: value}
                for name, v in values_flat.items():
                    if not isinstance(v, (float, int)):
                        continue
                    count = counts[name.removesuffix('_accuracy')] if task == 'mindcube_full' and name != 'overall_accuracy' else n['original']
                    metrics.append(dict(benchmark=p.name.removesuffix('_results.json'), task=task, metric=name, value=v, original=count, effective=count, higher_is_better='yes'))
    dump(FINAL / 'metrics.json', metrics)
    with (FINAL / 'metrics.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(metrics[0]))
        writer.writeheader()
        writer.writerows(metrics)
    # Only remove the old trees after every file is safely copied and checked.
    for source in sources:
        if source.exists():
            for p in source.rglob('*'):
                if p.is_file() and p.stat().st_size:
                    assert p.read_bytes() == (DEST / p.relative_to(source)).read_bytes()
            shutil.rmtree(source)
    print(json.dumps({'scores': scores, 'counts': dict(counts), 'correct': dict(correct)}, indent=2))

if __name__ == '__main__':
    main()
