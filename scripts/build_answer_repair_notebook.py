"""Build a Colab/Kaggle notebook for the already-frozen E GPU bundle."""

from __future__ import annotations

import hashlib
import json
import argparse
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def markdown(text: str) -> dict:
    return {"cell_type": "markdown", "metadata": {}, "source": text.splitlines(keepends=True)}


def code(text: str) -> dict:
    return {"cell_type": "code", "metadata": {}, "execution_count": None,
            "outputs": [], "source": text.splitlines(keepends=True)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle", type=Path, default=ROOT / "artifacts/answer_repair/v4/answer_repair_v4_gpu.tar.gz")
    parser.add_argument("--out", type=Path, default=ROOT / "notebooks/history/answer_repair/answer_repair_colab_v4.ipynb")
    parser.add_argument("--strategy", choices=("verify_repair", "grounded_reanswer"), default="verify_repair",
                        help="E strategy for the optional 50-sample development comparison")
    args = parser.parse_args()
    archive = args.bundle
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    cells = [markdown("""# E: first T4 verification and answer repair

In Colab, choose **Runtime → Change runtime type → T4 GPU**, then run the cells in order.
The same notebook can be imported on Kaggle. Enable GPU and Internet, and add the experiment bundle as a private input.

Upload `answer_repair_v1_gpu.tar.gz`. It contains frozen code and 50 FEVER plus 50 NQ development samples.
First compare Verify with cited independent candidate generation on 10 FEVER and 10 NQ items, 40 calls in total.
Check every output, then run the optional development experiment.
Development samples exclude Diagnosis v1/v2 validation and test IDs. Smoke is a subset of the development set.
Reference answers are used only for offline scoring and are not passed to the repair model. A legal citation does not mean semantic support has been verified.
"""), code("""import json, os, sys, subprocess, hashlib, tarfile, shutil
from pathlib import Path
import torch

assert sys.version_info >= (3, 11), "Python 3.11+ required"
assert torch.cuda.is_available(), "Select a GPU runtime first; this PyTorch build cannot see CUDA"
print(json.dumps({"python": sys.version, "torch": torch.__version__,
                  "cuda": torch.version.cuda,
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, indent=2))
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content')
OUTPUT = WORK / 'answer_repair_v1_outputs'
OUTPUT.mkdir(parents=True, exist_ok=True)
"""), code(f"""# Reuse a file already uploaded from the Colab file panel; otherwise open the upload dialog.
EXPECTED_BUNDLE_SHA256 = {digest!r}
archive = WORK / 'answer_repair_v1_gpu.tar.gz'
if not archive.exists() and Path('/kaggle/input').exists():
    candidates = list(Path('/kaggle/input').rglob('answer_repair_v1_gpu.tar.gz'))
    assert len(candidates) == 1, 'Add exactly one experiment-bundle input on Kaggle'
    archive = candidates[0]
elif not archive.exists():
    from google.colab import files
    uploaded = files.upload()
    assert 'answer_repair_v1_gpu.tar.gz' in uploaded, 'Select answer_repair_v1_gpu.tar.gz'
    archive.write_bytes(uploaded['answer_repair_v1_gpu.tar.gz'])

assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_BUNDLE_SHA256, 'Experiment bundle hash mismatch'
extract_root = WORK / ('e_bundle_' + EXPECTED_BUNDLE_SHA256[:12])
BUNDLE = extract_root / 'answer_repair_v1_gpu'
if not BUNDLE.exists():
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(extract_root, filter='data')
checksums = json.loads((BUNDLE / 'bundle_checksums.json').read_text())
assert all(hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == value
           for name, value in checksums.items()), 'Code or data file check failed'
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
print('Bundle verified:', len(checksums), 'files')
print(json.dumps({{ds: {{'development': item['development'], 'smoke': item['smoke']}}
                  for ds, item in manifest['datasets'].items()}}, indent=2))
"""), code("""def run_logged(argv, log_name, cwd=BUNDLE):
    # Forward subprocess output explicitly so the notebook does not only show "Executing".
    env = {**os.environ, 'PYTHONUNBUFFERED': '1'}
    with (OUTPUT / log_name).open('a', encoding='utf-8') as log:
        log.write('COMMAND ' + json.dumps(argv) + '\\n')
        log.flush()
        process = subprocess.Popen(argv, cwd=cwd, env=env, stdout=subprocess.PIPE,
                                   stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True)
                log.write(line)
                log.flush()
            returncode = process.wait()
        except BaseException:
            process.terminate()
            try:
                process.wait(timeout=10)
            except subprocess.TimeoutExpired:
                process.kill()
                process.wait()
            raise
    if returncode:
        raise subprocess.CalledProcessError(returncode, argv)

# Use the host CUDA PyTorch. The environment check records the versions that are actually installed.
run_logged([sys.executable, '-m', 'pip', 'install', '-r',
            str(BUNDLE / 'requirements-answer-repair.txt')], 'install.log')
run_logged([sys.executable, 'scripts/check_answer_repair_environment.py',
            '--require-cuda', '--out', str(OUTPUT / 'environment.json')], 'environment.log')
run_logged([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], 'tests.log')
shutil.copy2(BUNDLE / 'manifest.json', OUTPUT / 'manifest.json')
shutil.copy2(BUNDLE / 'bundle_checksums.json', OUTPUT / 'bundle_checksums.json')
"""), code("""SMOKE_RUNS = []
for dataset in ('fever', 'nq'):
    for strategy in ('verify_repair', 'grounded_reanswer'):
        result = OUTPUT / f'{dataset}_smoke10_{strategy}.jsonl'
        argv = [sys.executable, '-u', 'scripts/run_answer_repair.py', '--dataset', dataset,
                '--c1', str(BUNDLE / f'inputs/{dataset}_smoke10_c1.jsonl'), '--all',
                '--strategy', strategy, '--out', str(result)]
        SMOKE_RUNS.append((dataset, strategy, result, argv))
        run_logged(argv + ['--check-only'], f'{dataset}_{strategy}_preflight.log')
"""), code("""# Real 7B NF4 calls. The first run downloads the model. Each item is appended, so the same environment can resume.
for dataset, strategy, result, argv in SMOKE_RUNS:
    run_logged(argv, f'{dataset}_{strategy}_run.log')
"""), code("""for dataset, strategy, result_path, argv in SMOKE_RUNS:
    print(json.dumps(json.loads(result_path.with_suffix('.summary.json').read_text()), indent=2, ensure_ascii=False))
    source_rows = {str(row['sample_id']): row for row in
                   map(json.loads, (BUNDLE / f'inputs/{dataset}_smoke10_c1.jsonl').read_text().splitlines())}
    review = []
    for result in map(json.loads, result_path.read_text().splitlines()):
        source = source_rows[result['sample_id']]
        row = {**result, 'reference_answer_or_label': source['gold_answer_or_label'],
               'input_evidence': source['evidence'],
               'manual_review': {'supports_final_answer': None, 'notes': ''}}
        review.append(row)
        print(json.dumps({k: result.get(k) for k in
                         ('sample_id', 'status', 'initial_answer', 'final_answer', 'citations', 'validation_error')},
                        ensure_ascii=False))
    review_file = result_path.with_suffix('.review.jsonl')
    review_file.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\\n' for row in review))
    print('Offline review template, not yet filled by a person:', review_file)
"""), markdown("""## Optional development experiment

First review the smoke JSON, citations, corrections, and regressions. After execution looks right, set
`RUN_DEVELOPMENT` to `True` in the next cell. It runs the selected E strategy and the Reanswer
control on 50 FEVER and 50 NQ items, at most 200 extra generations. These are development results and are not a final independent test.
"""), code("""RUN_DEVELOPMENT = False
SELECTED_E_STRATEGY = 'SELECTED_STRATEGY_PLACEHOLDER'
if RUN_DEVELOPMENT:
    for dataset in ('fever', 'nq'):
        for strategy in (SELECTED_E_STRATEGY, 'reanswer'):
            out = OUTPUT / f'{dataset}_dev50_{strategy}.jsonl'
            run_logged([sys.executable, '-u', 'scripts/run_answer_repair.py', '--dataset', dataset,
                        '--c1', str(BUNDLE / f'inputs/{dataset}_dev50_c1.jsonl'), '--all',
                        '--strategy', strategy, '--out', str(out)], f'{dataset}_{strategy}.log')
else:
    print('Development experiment not started; review smoke first.')
"""), code("""# Export existing results and environment records. Model weights are not packaged.
export = shutil.make_archive(str(WORK / 'answer_repair_v1_outputs'), 'zip', OUTPUT)
print('Result archive:', export)
if not Path('/kaggle/working').exists():
    from google.colab import files
    files.download(export)
""")]
    notebook = {"nbformat": 4, "nbformat_minor": 5,
                "metadata": {"kernelspec": {"display_name": "Python 3", "name": "python3"},
                             "language_info": {"name": "python"}, "accelerator": "GPU",
                             "colab": {"name": "answer_repair_colab.ipynb"}}, "cells": cells}
    for index, cell in enumerate(cells):
        cell["id"] = f"e-v1-{index:02d}"
        bundle_name = archive.name.removesuffix('.tar.gz')
        source = ''.join(cell['source']).replace('answer_repair_v1_gpu', bundle_name)
        source = source.replace('answer_repair_v1_outputs', bundle_name.removesuffix('_gpu') + '_outputs')
        source = source.replace('SELECTED_STRATEGY_PLACEHOLDER', args.strategy)
        cell['source'] = source.splitlines(keepends=True)
        if cell["cell_type"] == "code":
            compile("".join(cell["source"]), f"cell-{index}", "exec")
    destination = args.out
    notebook['metadata']['colab']['name'] = destination.name
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(notebook, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"notebook": str(destination), "bundle_sha256": digest,
                      "code_cells_syntax_checked": sum(c["cell_type"] == "code" for c in cells)}, indent=2))


if __name__ == "__main__":
    main()
