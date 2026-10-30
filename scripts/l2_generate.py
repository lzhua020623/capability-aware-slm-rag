"""GPU generation for L2 Top-5 contexts; no gold is read.

Exactly P3-20's regeneration: build_c1_prompt over the selected passages (corpus text),
the shared Qwen generator with the Controller's answer token limit, then answer_rejection.
One JSON line per (sample, Top-5), appended as it finishes, so a run resumes.
"""
import argparse
import json
from pathlib import Path
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path: sys.path.insert(0, str(ROOT))
from scripts.l2_candidates import NQ_RESOURCES, REGISTRATION, file_sha, now  # noqa: E402


def p3_runtime(pipeline):
    """The shared generator, token limit and NQ backend P3-20 uses."""
    from src.controller.schema import ControllerAction
    controller = pipeline.host.executor._registry[ControllerAction.RETRIEVAL_ENHANCEMENT].executor
    return controller.raw_generator, controller.config.answer_max_new_tokens, controller.retriever


def outcome(query, passages, generate):
    from src.compensation.retrieval_enhancement.p3 import answer_rejection, build_c1_prompt
    answer = generate(build_c1_prompt(query, passages))
    if not isinstance(answer, str):
        raise TypeError('generate must return a string')
    return answer, answer_rejection(answer, passages)


def key(row):
    return row['sample_id'], tuple(row['passage_ids'])


def done_keys(path):
    if not path.exists():
        return set()
    data = path.read_bytes()
    if data and not data.endswith(b'\n'):
        path.write_bytes(data[:data.rfind(b'\n') + 1])
    return {key(json.loads(line)) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()}


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--contexts', type=Path, required=True, help='l2_offline.py contexts output')
    p.add_argument('--output-dir', type=Path, required=True)
    args = p.parse_args(argv)
    rows = [json.loads(line) for line in args.contexts.read_text(encoding='utf-8').splitlines() if line.strip()]
    if any(set(r) != {'sample_id', 'query', 'passage_ids'} for r in rows):
        raise ValueError('Context rows must hold sample_id, query and passage_ids only')
    import torch
    if not torch.cuda.is_available(): raise RuntimeError('Generation requires CUDA')
    from src.pipeline.current import load_current_pipeline
    pipeline, environment = load_current_pipeline(root=ROOT, nq_resources=ROOT / NQ_RESOURCES,
                                                  data_role='validation', attach_reference_rag=True)
    pipeline.data_boundary.require(sorted({r['sample_id'] for r in rows}), 'validation')
    generator, max_new_tokens, backend = p3_runtime(pipeline)
    position = {pid: i for i, pid in enumerate(backend.retriever.passage_ids)}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    out = args.output_dir / 'generations.jsonl'
    finished = done_keys(out)
    manifest = {'schema_version': 'l2_generations_v1', 'state': 'running', 'started_UTC': now(),
                'registration_sha256': file_sha(ROOT / REGISTRATION), 'contexts_sha256': file_sha(args.contexts),
                'n': len(rows), 'resumed_with': len(finished), 'answer_max_new_tokens': max_new_tokens,
                'cpu_threads': torch.get_num_threads(), 'environment': environment}
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
    generate = lambda prompt: generator.generate(prompt, max_new_tokens).text
    todo, started = [r for r in rows if key(r) not in finished], time.perf_counter()
    with out.open('a', encoding='utf-8') as handle:
        for i, row in enumerate(todo, 1):
            passages = []
            for pid in row['passage_ids']:
                record_id, text = backend._record(position[pid])
                passages.append({'passage_id': record_id, 'text': text})
            answer, reason = outcome(row['query'], passages, generate)
            handle.write(json.dumps({**row, 'generated_answer': answer, 'reject_reason': reason}, ensure_ascii=False) + '\n')
            handle.flush()
            if i % 200 == 0 or i == len(todo):
                print(json.dumps({'done': len(finished) + i, 'of': len(rows),
                                  'seconds': round(time.perf_counter() - started, 1)}), flush=True)
    manifest.update(state='completed', finished_UTC=now(), rows=len(done_keys(out)), output_sha256=file_sha(out))
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2, default=str) + '\n', encoding='utf-8')
    return 0


if __name__ == '__main__': raise SystemExit(main())
