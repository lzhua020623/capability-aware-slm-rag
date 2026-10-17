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
    cells = [markdown("""# E：T4 首轮核验与答案修复

在 Colab 选择 **代码执行程序 → 更改运行时类型 → T4 GPU**，依次执行单元格。
Kaggle 可导入同一 notebook，启用 GPU 和 Internet，并把实验包添加到私有输入。

上传 `answer_repair_v1_gpu.tar.gz`；它包含冻结代码和 FEVER/NQ 各 50 条开发样本。
先在 FEVER/NQ 各 10 条上比较 Verify 与带引用的独立候选生成，共 40 次调用。
检查全部输出，再运行可选开发实验。
开发样本排除了 Diagnosis v1/v2 的验证、测试 ID。Smoke 是开发集的子集。
参考答案仅用于离线评分，不会传入修复模型。引用合法不代表语义支持已验证。
"""), code("""import json, os, sys, subprocess, hashlib, tarfile, shutil
from pathlib import Path
import torch

assert sys.version_info >= (3, 11), "需要 Python 3.11+"
assert torch.cuda.is_available(), "请先选择 GPU 运行时；当前 PyTorch 看不到 CUDA"
print(json.dumps({"python": sys.version, "torch": torch.__version__,
                  "cuda": torch.version.cuda,
                  "gpus": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]}, indent=2))
WORK = Path('/kaggle/working') if Path('/kaggle/working').exists() else Path('/content')
OUTPUT = WORK / 'answer_repair_v1_outputs'
OUTPUT.mkdir(parents=True, exist_ok=True)
"""), code(f"""# 已从 Colab 左侧文件面板上传时直接复用；否则弹出上传框。
EXPECTED_BUNDLE_SHA256 = {digest!r}
archive = WORK / 'answer_repair_v1_gpu.tar.gz'
if not archive.exists() and Path('/kaggle/input').exists():
    candidates = list(Path('/kaggle/input').rglob('answer_repair_v1_gpu.tar.gz'))
    assert len(candidates) == 1, '请在 Kaggle 添加唯一的实验包输入'
    archive = candidates[0]
elif not archive.exists():
    from google.colab import files
    uploaded = files.upload()
    assert 'answer_repair_v1_gpu.tar.gz' in uploaded, '请选择 answer_repair_v1_gpu.tar.gz'
    archive.write_bytes(uploaded['answer_repair_v1_gpu.tar.gz'])

assert hashlib.sha256(archive.read_bytes()).hexdigest() == EXPECTED_BUNDLE_SHA256, '实验包哈希不一致'
extract_root = WORK / ('e_bundle_' + EXPECTED_BUNDLE_SHA256[:12])
BUNDLE = extract_root / 'answer_repair_v1_gpu'
if not BUNDLE.exists():
    extract_root.mkdir(parents=True, exist_ok=True)
    with tarfile.open(archive) as handle:
        handle.extractall(extract_root, filter='data')
checksums = json.loads((BUNDLE / 'bundle_checksums.json').read_text())
assert all(hashlib.sha256((BUNDLE / name).read_bytes()).hexdigest() == value
           for name, value in checksums.items()), '代码或数据文件校验失败'
manifest = json.loads((BUNDLE / 'manifest.json').read_text())
print('包校验通过:', len(checksums), '个文件')
print(json.dumps({{ds: {{'development': item['development'], 'smoke': item['smoke']}}
                  for ds, item in manifest['datasets'].items()}}, indent=2))
"""), code("""def run_logged(argv, log_name, cwd=BUNDLE):
    # 显式转发子进程输出，避免 notebook 只显示“正在执行”。
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

# 使用宿主 CUDA PyTorch；环境检查记录实际安装后的版本。
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
"""), code("""# 真实 7B NF4 调用；首次需要下载模型。每条追加保存，可在同一环境断点续跑。
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
    print('离线审核模板，尚未人工填写:', review_file)
"""), markdown("""## 可选开发实验

先审核 smoke 中的 JSON、引用、改对和改错。确认执行正常后，将下一格的
`RUN_DEVELOPMENT` 改为 `True`。它会运行 FEVER/NQ 各 50 条的选定 E 策略和 Reanswer
对照，总计最多 200 次额外生成。这是开发结果，不能当作最终独立测试结果。
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
    print('开发实验未启动；先审核 smoke。')
"""), code("""# 导出已有结果和环境记录；模型权重不打包。
export = shutil.make_archive(str(WORK / 'answer_repair_v1_outputs'), 'zip', OUTPUT)
print('结果包:', export)
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
