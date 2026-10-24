"""Synthetic real-provider gates, separate from benchmark quality evaluation."""
from copy import copy
from dataclasses import asdict
from statistics import median
from time import perf_counter

from src.compensation.answer_repair import build_repair_prompt
from src.compensation.short_answer_repair import build_short_answer_prompt
from src.compensation.retrieval_enhancement.p3 import RERANKER_MODEL, RERANKER_REVISION
from src.controller.execution import CallLog, MeteredGenerator
from src.diagnosis.schema import DiagnosisInput, EvidenceItem
from src.generation.prompts import build_rag_prompt, build_fever_rag_prompt
from src.probe.prompts import build_probe_prompt
from src.probe.probe import EvidenceProbe, PROBE_MAX_NEW_TOKENS


def fictional_item(dataset, padding=0):
    text = ('The capital of Luma is Amber. ' +
            'This fictional archive stores unrelated background notes. ' * padding)
    return DiagnosisInput('engineering-fiction-' + dataset, dataset,
                          'What is the capital of Luma?' if dataset == 'nq' else 'The capital of Luma is Amber.',
                          [EvidenceItem(i, f'fictional-{i}', text, .9 - i / 100) for i in range(1, 6)],
                          'Blue' if dataset == 'nq' else 'REFUTES')


def fixture_prompts(item):
    rows = [{'text': e.text} for e in item.evidence]
    regeneration = (build_rag_prompt(item.query, rows) if item.dataset.value == 'nq' else
                    build_fever_rag_prompt(item.query, rows))
    return [build_probe_prompt(item), regeneration,
            build_short_answer_prompt(item) if item.dataset.value == 'nq' else build_repair_prompt(item)]


def long_fixture(tokenizer, dataset, minimum=14000):
    from src.generation.qwen import apply_qwen_chat_template
    padding = 64
    while padding <= 4096:
        item = fictional_item(dataset, padding)
        lengths = [len(tokenizer.encode(apply_qwen_chat_template(tokenizer, p))) for p in fixture_prompts(item)]
        if min(lengths) >= minimum:
            if max(lengths) + 384 > 32768:
                raise ValueError('Synthetic gate exceeds the fixed model window')
            return item, lengths
        padding = int(padding * 1.25) + 1
    raise RuntimeError('Cannot construct registered synthetic context')


class FictionalScorer:
    # Force changed rank order so every intended generation path actually runs.
    # Real CPU/FP32 scoring is timed independently below, never claimed here.
    model_name, revision = RERANKER_MODEL, RERANKER_REVISION
    def __call__(self, query, texts):
        return list(range(len(texts)))


class FictionalRetriever:
    supported_datasets = frozenset({'nq', 'fever'})
    def __init__(self, text):
        self.text = text
    def retrieve(self, query, *, dataset, top_k):
        return [{'passage_id': f'fictional-extra-{i}', 'text': self.text,
                 'score': .95 - i / 100} for i in range(top_k)]


def inspect_engineering_calls(calls, minimum):
    attempted = [e for e in calls if e['kind'] == 'llm' and e['attempted']]
    if not attempted or any(e['status'] != 'completed' or e.get('output', {}).get('input_tokens', 0) < minimum
                            for e in attempted):
        raise RuntimeError('Real generation did not cover the registered complete input length')
    return {'n_llm_calls': len(attempted),
            'minimum_actual_input_tokens': min(e['output']['input_tokens'] for e in attempted)}


def engineering_checks(pipeline, *, minimum=14000, on_check=None):
    isolated = copy(pipeline)
    isolated.scorer = FictionalScorer()
    checks = []
    for dataset in ('nq', 'fever'):
        item, lengths = long_fixture(pipeline.raw_generator.tokenizer, dataset, minimum)
        isolated.retriever = FictionalRetriever(item.evidence[0].text)
        for mode in ('probe', 'always_retrieve', 'always_rerank', 'always_repair'):
            if mode == 'probe':
                calls = CallLog(pipeline.config)
                generator = MeteredGenerator(pipeline.raw_generator, calls, 'engineering_probe')
                result = EvidenceProbe(lambda prompt: generator.generate(prompt, PROBE_MAX_NEW_TOKENS).text).run(item)
                record = {'calls': calls.events, 'probe_result': asdict(result)}
            else:
                record = isolated.run(item, mode=mode)
            if on_check:
                on_check({'dataset': dataset, 'path': mode, 'passed': False, 'record': record})
            inspection = inspect_engineering_calls(record['calls'], minimum)
            checks.append({'dataset': dataset, 'path': mode, 'passed': True,
                           'fixture_prompt_lengths': lengths, **inspection, 'record': record})
            if on_check:
                on_check(checks[-1])
            print(f'ENGINEERING {dataset}/{mode} passed; input_tokens={inspection["minimum_actual_input_tokens"]}', flush=True)
    # Measure true CPU FP32 scoring, without replacing its outputs in benchmarks.
    cpu = []
    for count in (5, 20):
        text = fictional_item('nq', 100).evidence[0].text
        start = perf_counter(); values = list(pipeline.scorer('What is the capital of Luma?', [text] * count))
        seconds = perf_counter() - start
        if len(values) != count:
            raise ValueError('CPU scoring fixture failed')
        cpu.append({'pairs': count, 'seconds': seconds, 'dtype': 'float32', 'device': 'cpu'})
        print(f'ENGINEERING real CPU scorer: {count} pairs, {seconds:.3f}s', flush=True)
    short = []
    for dataset in ('nq', 'fever'):
        item = fictional_item(dataset)
        isolated.retriever = FictionalRetriever(item.evidence[0].text)
        for mode in ('always_retrieve', 'always_rerank', 'always_repair', 'full_enhancement'):
            record = isolated.run(item, mode=mode)
            inspect_engineering_calls(record['calls'], 1)
            if record['cost']['failed_calls'] or record['cost']['budget_blocks']:
                raise RuntimeError('Short execution/Full budget gate failed')
            short.append(record)
    seconds = [e['seconds'] for r in short for e in r['calls'] if e['kind'] == 'llm' and e['attempted']]
    return {'schema_version': 'controller_t3_engineering_v1', 'passed': True,
            'minimum_input_tokens': minimum, 'checks': checks, 'short_checks': short,
            'cpu_scoring': cpu, 'short_llm_median_seconds': median(seconds),
            'scope': 'Real Qwen + independent real CPU scoring; synthetic reversed scorer/retrieval for path coverage; no benchmark/gold read',
            'quality_scored': False}
