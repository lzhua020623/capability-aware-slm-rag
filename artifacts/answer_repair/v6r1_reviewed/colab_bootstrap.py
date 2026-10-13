# E v6: frozen bundle, separate outputs, both development cohorts.
from pathlib import Path
import hashlib, json, os, subprocess, sys, tarfile, shutil
import torch
assert torch.cuda.is_available()
WORK = Path('/content')
ARCHIVE = WORK / 'answer_repair_v6r1_reviewed_gpu.tar.gz'
assert hashlib.sha256(ARCHIVE.read_bytes()).hexdigest() == '74d40b01c96245e1caf2ac8af786bf3b05394971863bddc81a75a2a2afa45ef5'
EXTRACT = WORK / 'e_reviewed_74d40b01c962'
BUNDLE = EXTRACT / 'answer_repair_v6r1_reviewed_gpu'
if not BUNDLE.exists():
    EXTRACT.mkdir(parents=True, exist_ok=True)
    with tarfile.open(ARCHIVE) as archive:
        archive.extractall(EXTRACT, filter='data')
checksums = json.loads((BUNDLE / 'bundle_checksums.json').read_text())
assert all(hashlib.sha256((BUNDLE / p).read_bytes()).hexdigest() == h for p,h in checksums.items())
OUTPUT = WORK / 'answer_repair_v6r1_reviewed_outputs'
OUTPUT.mkdir(exist_ok=True)
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
def run_v6(argv, filename):
    with (OUTPUT / filename).open('a') as log:
        log.write('COMMAND ' + json.dumps(argv) + '\n'); log.flush()
        process = subprocess.Popen(argv, cwd=BUNDLE, env={**os.environ, 'PYTHONUNBUFFERED':'1'}, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
        try:
            for line in process.stdout:
                print(line, end='', flush=True); log.write(line); log.flush()
            code = process.wait()
        except BaseException:
            process.terminate()
            try: process.wait(timeout=10)
            except subprocess.TimeoutExpired: process.kill(); process.wait()
            raise
    if code: raise subprocess.CalledProcessError(code, argv)
run_v6([sys.executable, '-m', 'unittest', 'discover', '-s', 'tests'], 'tests.log')
run_v6([sys.executable, 'scripts/check_answer_repair_environment.py', '--require-cuda', '--out', str(OUTPUT / 'environment.json')], 'environment.log')
for path in ('manifest.json', 'bundle_checksums.json'):
    shutil.copy2(BUNDLE / path, OUTPUT / path)
V6_RUNS = []
for cohort in ('reviewed', 'dev50'):
    for arm in manifest['protocol']:
        ds, strategy = arm['dataset'], arm['strategy']
        stem = f'{ds}_{strategy}' + ('_dev50' if cohort == 'dev50' else '')
        source, result = BUNDLE / f'inputs/{ds}_{cohort}_c1.jsonl', OUTPUT / (stem + '.jsonl')
        argv = [sys.executable, '-u', 'scripts/run_answer_repair.py', '--dataset', ds, '--c1', str(source), '--all', '--strategy', strategy, '--max-new-tokens', str(arm['max_new_tokens']), '--out', str(result)]
        run_v6(argv + ['--check-only'], stem + '_preflight.log')
        V6_RUNS.append((cohort, stem, source, result, argv))
print('V6_PREFLIGHT_COMPLETE', len(V6_RUNS), flush=True)
for cohort, stem, source, result, argv in V6_RUNS:
    print('V6_START', stem, flush=True)
    run_v6(argv, stem + '_run.log')
    if cohort == 'reviewed':
        run_v6([sys.executable, 'scripts/summarize_reviewed_answer_repair.py', '--manifest', str(BUNDLE / 'manifest.json'), '--c1', str(source), '--results', str(result), '--out-prefix', str(OUTPUT / (stem + '_reviewed'))], stem + '_summary.log')
export = shutil.make_archive(str(WORK / (OUTPUT.name + '_export')), 'zip', OUTPUT)
print('V6_REAL_RUN_COMPLETE', hashlib.sha256(Path(export).read_bytes()).hexdigest(), flush=True)
from google.colab import files
files.download(export)
