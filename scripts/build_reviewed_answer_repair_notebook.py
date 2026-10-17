"""Create a Colab/Kaggle notebook for the frozen reviewed v5 development bundle."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def cell(kind, source):
    result = {"cell_type": kind, "metadata": {}, "source": source.splitlines(keepends=True)}
    if kind == 'code':
        compile(source, '<notebook-cell>', 'exec')
        result.update(execution_count=None, outputs=[])
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=ROOT / 'artifacts/answer_repair/v5_reviewed/answer_repair_v5_reviewed_gpu.tar.gz')
    parser.add_argument('--out', type=Path, default=ROOT / 'notebooks/history/answer_repair/answer_repair_colab_v5_reviewed.ipynb')
    parser.add_argument('--experiment', choices=('v5', 'v6', 'v7'), default='v5')
    args = parser.parse_args()
    digest = hashlib.sha256(args.bundle.read_bytes()).hexdigest()
    name = args.bundle.name.removesuffix('.tar.gz')
    cells = [cell('markdown', '''# E v5：28 条审核后开发样本

选择 Colab T4 GPU 或 Kaggle GPU，按顺序执行。上传配套 `.tar.gz` 实验包。
FEVER/NQ 各 14 条，共 19 条错误核心候选和 9 条正确对照，均为训练内开发材料。
四组对照共 112 条最终预测，最多 140 次生成；同模型复核不是独立 verifier。
旧参考答案的 EM/F1 与人工语义判断分开保存，人工输出审核栏不会自动填充。
不挂载 Drive。运行结束后务必下载 ZIP；云端运行时释放会丢失本地文件。
'''), cell('code', f'''import json, os, sys, subprocess, hashlib, tarfile, shutil
from pathlib import Path
import torch
assert sys.version_info >= (3, 11), '需要 Python 3.11+'
assert torch.cuda.is_available(), '请选择 GPU 运行时'
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content')
EXPECTED_SHA256 = {digest!r}
BUNDLE_NAME = {name!r}
OUTPUT = WORK / (BUNDLE_NAME.removesuffix('_gpu') + '_outputs')
OUTPUT.mkdir(parents=True, exist_ok=True)
print('GPU:', torch.cuda.get_device_name(0), 'CUDA:', torch.version.cuda)
'''), cell('code', '''archive = WORK / (BUNDLE_NAME + '.tar.gz')
if not archive.exists() and Path('/kaggle/input').exists():
    candidates = list(Path('/kaggle/input').rglob(archive.name))
    assert len(candidates) == 1, '请添加唯一的实验包输入'
    archive = candidates[0]
elif not archive.exists():
    from google.colab import files
    uploaded = files.upload()
    assert archive.name in uploaded, '请选择配套实验包'
    archive.write_bytes(uploaded[archive.name])
assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_SHA256, '归档哈希不一致'
extract_root = WORK / ('e_reviewed_' + EXPECTED_SHA256[:12])
BUNDLE = extract_root / BUNDLE_NAME
if not BUNDLE.exists():
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(extract_root, filter='data')
checksums = json.loads((BUNDLE / 'bundle_checksums.json').read_text())
assert all(hashlib.sha256((BUNDLE / p).read_bytes()).hexdigest() == h for p,h in checksums.items())
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
print('包校验通过:', len(checksums), '个文件')
print({ds: r['groups'] for ds,r in manifest['datasets'].items()})
'''), cell('code', '''def run_logged(argv, name):
    with (OUTPUT / name).open('a', encoding='utf-8') as log:
        log.write('COMMAND ' + json.dumps(argv) + '\\n'); log.flush()
        process = subprocess.Popen(argv, cwd=BUNDLE, env={**os.environ, 'PYTHONUNBUFFERED': '1'},
                                   stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            status = process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if status: raise subprocess.CalledProcessError(status, argv)

run_logged([sys.executable, '-m', 'pip', 'install', '-r', 'requirements-answer-repair.txt'], 'install.log')
run_logged([sys.executable, 'scripts/check_answer_repair_environment.py', '--require-cuda',
            '--out', str(OUTPUT / 'environment.json')], 'environment.log')
run_logged([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], 'tests.log')
for path in ('manifest.json', 'bundle_checksums.json'):
    shutil.copy2(BUNDLE / path, OUTPUT / path)
RUNS = []
for ds in ('fever', 'nq'):
    for arm in manifest['protocol']:
        if arm.get('dataset', ds) != ds: continue
        strategy = arm['strategy']
        result = OUTPUT / f'{ds}_{strategy}.jsonl'
        argv = [sys.executable, '-u', 'scripts/run_answer_repair.py', '--dataset', ds,
                '--c1', str(BUNDLE / f'inputs/{ds}_reviewed_c1.jsonl'), '--all',
                '--strategy', strategy, '--max-new-tokens', str(arm['max_new_tokens']), '--out', str(result)]
        RUNS.append((ds, strategy, result, argv))
        run_logged(argv + ['--check-only'], f'{ds}_{strategy}_preflight.log')
'''), cell('markdown', '''## 真实开发试跑

下一格执行全部四种策略。候选失败时跳过复核，已完成样本会自动跳过。
每个子进程结束释放模型显存，模型文件通过云端缓存复用。
生成中断时保留已写入的行；恢复环境一致后重跑本格。
'''), cell('code', '''for ds, strategy, result, argv in RUNS:
    print('开始:', ds, strategy, flush=True)
    run_logged(argv, f'{ds}_{strategy}_run.log')
'''), cell('code', '''for ds, strategy, result, argv in RUNS:
    run_logged([sys.executable, 'scripts/summarize_reviewed_answer_repair.py',
                '--manifest', str(BUNDLE / 'manifest.json'),
                '--c1', str(BUNDLE / f'inputs/{ds}_reviewed_c1.jsonl'),
                '--results', str(result), '--out-prefix', str(OUTPUT / f'{ds}_{strategy}_reviewed')],
               f'{ds}_{strategy}_summary.log')
# 将无复核与有复核两组的候选作比较，避免把候选生成差异归因于复核。
comparison = {}
for ds in ('fever', 'nq') if any(a['strategy']=='guarded_reanswer' for a in manifest['protocol']) else ():
    load = lambda strategy: [json.loads(l) for l in (OUTPUT / f'{ds}_{strategy}.jsonl').read_text().splitlines()]
    plain, guarded = load('query_grounded'), load('guarded_reanswer')
    assert [r['sample_id'] for r in plain] == [r['sample_id'] for r in guarded]
    comparison[ds] = {'candidate_output_mismatch_ids': [a['sample_id'] for a,b in zip(plain,guarded)
        if a['raw_output'] != b['raw_output'] or a['validation_input'] != b['validation_input']]}
(OUTPUT / 'candidate_comparison.json').write_text(json.dumps(comparison, ensure_ascii=False, indent=2))
print('候选一致性检查:', comparison)
print('人工成对审核表已生成，所有结果审核栏仍为空。')
'''), cell('code', '''export = shutil.make_archive(str(WORK / (OUTPUT.name + '_export')), 'zip', OUTPUT)
print('结果包:', export)
print('SHA256:', hashlib.sha256(Path(export).read_bytes()).hexdigest())
if not Path('/kaggle/working').exists():
    from google.colab import files
    files.download(export)
''')]
    if args.experiment == 'v6':
        cells[0] = cell('markdown', """# E v6: extractive NQ and independent FEVER NLI

28 reviewed development samples plus the previous 100 development regression samples.
The two cohorts overlap on 5 IDs; do not combine them as 128 independent samples.
FEVER uses a pinned MNLI classifier, not Qwen. Its cost and accuracy are reported separately.
NQ uses the pinned Qwen NF4 model and selects exact evidence spans once.
All paired semantic review remains pending. This is development, not held-out evaluation.
""")
        regression_code = """# Fixed previous development cohort; additional regression checks, no tuning.
for arm in manifest['protocol']:
    ds, strategy = arm['dataset'], arm['strategy']
    result = OUTPUT / f'{ds}_{strategy}_dev50.jsonl'
    argv = [sys.executable, '-u', 'scripts/run_answer_repair.py', '--dataset', ds,
            '--c1', str(BUNDLE / f'inputs/{ds}_dev50_c1.jsonl'), '--all',
            '--strategy', strategy, '--max-new-tokens', str(arm['max_new_tokens']), '--out', str(result)]
    run_logged(argv + ['--check-only'], f'{ds}_{strategy}_dev50_preflight.log')
    run_logged(argv, f'{ds}_{strategy}_dev50_run.log')
"""
        cells.insert(-1, cell('code', regression_code))
        cells[4] = cell('markdown', '## Real GPU experiments\nRun both dataset-specific arms, then the fixed regression cohort.\n')
    if args.experiment == 'v7':
        cells[0] = cell('markdown', """# E v7: short answer + quote for NQ, subject-anchored NLI for FEVER

Protocol: docs/ANSWER_REPAIR_V7.md (frozen before this run). Three cohorts per dataset:
reviewed 28 and dev50 x2 are paired with v6r1; heldout (FEVER 267, NQ 146) is E-untouched
shared-training data and is run ONCE and reported regardless of outcome. No tuning here.
FEVER uses the pinned MNLI classifier (same windows and scores as v6; only the decision rule changed).
NQ uses the pinned Qwen NF4 model, one call, line output, no JSON stop.
All paired semantic review remains pending. Costs are reported per model family.
""")
        cohort_code = """# Fixed cohorts beyond the reviewed 28: dev50 (paired with v6r1) and the run-once held-out.
for cohort in ('dev50', 'heldout'):
    for arm in manifest['protocol']:
        ds, strategy = arm['dataset'], arm['strategy']
        result = OUTPUT / f'{ds}_{strategy}_{cohort}.jsonl'
        argv = [sys.executable, '-u', 'scripts/run_answer_repair.py', '--dataset', ds,
                '--c1', str(BUNDLE / f'inputs/{ds}_{cohort}_c1.jsonl'), '--all',
                '--strategy', strategy, '--max-new-tokens', str(arm['max_new_tokens']), '--out', str(result)]
        run_logged(argv + ['--check-only'], f'{ds}_{strategy}_{cohort}_preflight.log')
        run_logged(argv, f'{ds}_{strategy}_{cohort}_run.log')
"""
        cells.insert(-1, cell('code', cohort_code))
        cells[4] = cell('markdown', '## Real GPU experiments\nReviewed 28 first, then dev50 and the run-once held-out cohorts.\n')
    for i,c in enumerate(cells): c['id'] = f'e-reviewed-v5-{i:02d}'
    notebook = {'nbformat': 4, 'nbformat_minor': 5, 'cells': cells,
                'metadata': {'kernelspec': {'name': 'python3', 'display_name': 'Python 3'},
                             'language_info': {'name': 'python'}, 'accelerator': 'GPU',
                             'colab': {'name': args.out.name}}}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(notebook, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({'notebook': str(args.out), 'bundle_sha256': digest,
                      'code_cells_syntax_checked': sum(c['cell_type']=='code' for c in cells)}, indent=2))


if __name__ == '__main__':
    main()
