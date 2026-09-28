# Larger-model control: Llama-3.3-70B

Run date: 2026-09-29. Branch: `feature/-ce-70B`.

Answers the frozen protocol's larger-model question on the samples the 7B failed
under both C1 and C3: **when a bigger model is handed the same correct evidence,
how many of those failures does it recover?**

## Headline

| Dataset | Cohort | Recovered | 95% CI |
|---|---|---|---|
| NQ | 67 | **15 (22.4%)** | [14.1%, 33.7%] |
| FEVER | 18 | **8 (44.4%)** | [24.6%, 66.3%] |
| Combined | 85 | **23 (27.1%)** | [18.8%, 37.3%] |

Under the frozen taxonomy: **23 samples are a likely small-model capability
limitation**, and **62 are unresolved hard cases** — a 70B with oracle evidence
still gets them wrong, so model scale is not what is missing.

Substituting Qwen2.5-32B for the frozen 32B control was not needed; 72B was used
as the judge and Llama-3.3-70B as the answerer, keeping the two roles in
different model families.

## Setup

- Answerer: `meta-llama/llama-3.3-70b-instruct` via OpenRouter, temperature 0,
  `max_tokens` 64, frozen C3 prompts from `configs/preliminary.yaml`
- Condition: **C3 only** (oracle evidence). This isolates model capability from
  retrieval, matching the frozen routing for the 32B control
- Cohort: C1 wrong **and** C3 wrong, i.e. the failures oracle evidence did not fix
- Judge: `qwen/qwen-2.5-72b-instruct`, applied to the 7B's and the 70B's NQ
  answers with the same prompt

Outputs: `results/preliminary/nq_c3_70b.jsonl`,
`results/preliminary/fever_c3_70b.jsonl`,
`results/preliminary/nq_70b_judge_verdicts.jsonl`.

## Why the NQ cohort is 67 and not 227

The frozen persistent-failure set has 227 samples. Most of them are not failures.

The frozen NQ metric is normalized exact match against a single gold string, but
the stored golds are frequently whole clauses — median 5 words, p75 11, max 41 —
even though the frozen prompt instructs the model to "return only the shortest
answer span". The model obeys the instruction and then fails the metric. The NQ
extractor also joins tokens with spaces, so `P$C` is stored as `P $ C` and
`(1983)` as `( 1983 )`.

A Qwen2.5-72B judge over all 566 exact-match-wrong answers accepted exactly half
of them. It validates: on 60 samples exact match had already scored correct, the
judge agreed 60/60, so it is not systematically strict.

| | Frozen (exact match) | Judge-corrected |
|---|---|---|
| C1 accuracy | 35.2% | 61.6% |
| C3 accuracy | 51.6% | 81.8% |
| **Recoverable Failure Rate** | **29.9%** [25.2, 35.1] | **65.1%** [58.1, 71.5] |
| Persistent failures | 227 | **67** |

The intervals do not overlap. This is a re-analysis, not a protocol change — the
frozen numbers stay the headline unless the team agrees otherwise. But running
the 70B over the frozen 227 and scoring it with the same exact match would have
reproduced the artifact on the same 160 samples and read as a capability ceiling.

FEVER needs none of this: predictions parse cleanly to SUPPORTS/REFUTES with zero
failures, so its 18-sample cohort is used as frozen.

## The same correction applies to the 70B

Scored by exact match the 70B recovered 8/67. The judge rescued 7 more, all the
same shape — the model returned the short span the prompt asked for and the gold
carried extra words:

| Gold | 70B answer |
|---|---|
| `AM 5778` | `5778` |
| `By the mid-1980s` | `mid-1980s` |
| `in the 1980s` | `1980s` |
| `Sarah's brother, Brian` | `Brian` |
| `volcanic and sedimentary rock sequences (magnetostratigraphy)` | `magnetostratigraphy` |

A token-F1 threshold does not rescue these: all seven score between 0.20 and
0.67, because a long gold drags F1 down even when the answer is exactly right.

Scoring the two models by different rules would have made the comparison
meaningless, so both went through the same judge.

## Limits

**No control group.** Every sample here was selected because the 7B got it wrong,
so the results support "of the 7B's failures the 70B recovers 27%" and *not* any
claim that the 70B is more accurate. The cell that is never measured is samples
the 7B got right and the 70B would get wrong. Measuring it needs a random sample
of 7B-correct items run through the same pipeline (`--control-n` on
`scripts/run_llama70b_control.py`).

**Small cohorts.** NQ ±10pp, FEVER ±21pp. NQ's 22.4% and FEVER's 44.4% have
overlapping intervals, so the gap between datasets is not established.

**Judge is unvalidated in the lenient direction.** The 60/60 check catches a
judge that is too strict; it cannot catch one that is too lenient. That needs
human labels on a sample of the flips. The judge shares a model family with the
7B it graded, though not with the 70B.

**Hosted inference is not bit-reproducible.** temperature 0 is greedy in intent,
but batching and kernel nondeterminism can change a token between runs.

## Reproducing

```powershell
$env:LLM_API_KEY = "..."

python scripts/check_backend.py --backend openrouter --model meta-llama/llama-3.3-70b-instruct

python scripts/run_llama70b_control.py `
    --cohort results/preliminary/nq_c1_c3_persistent_failures_judged.jsonl `
    --dataset nq --condition C3 `
    --backend openrouter --model meta-llama/llama-3.3-70b-instruct

python scripts/run_llama70b_control.py `
    --cohort results/preliminary/fever_c1_c3_persistent_failures.jsonl `
    --dataset fever --condition C3 `
    --backend openrouter --model meta-llama/llama-3.3-70b-instruct
```

Both runs resume from their output file, so an interrupted run continues rather
than repeating calls.
