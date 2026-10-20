# NQ Train Diagnosis Annotation Guide

Offline annotation guide for the NQ Train delivery package
(`data/annotations/nq_train_delivery/`).

This document defines how annotators assign the **four frozen Diagnosis v2
primary labels**. It does **not** change runtime Diagnosis, Probe, Controller,
or any Dev/Test artifact.

Code identifiers, field names, and label strings in this guide are English and
must match the frozen taxonomy in `src/diagnosis/labels.py` (Diagnosis branch).

---

## 1. Scope

Annotate each sample with exactly one primary label:

| Label | Meaning (frozen Diagnosis v2) |
|---|---|
| `correct` | The C1 initial answer is acceptable given the question. |
| `evidence_insufficiency` | C1 fails because required evidence is absent or insufficient in the retrieved Top-5. |
| `evidence_accessibility` | Required evidence is present in Top-5, but noise, conflict, or ranking/presentation prevents effective access. |
| `evidence_utilisation` | Required evidence is sufficiently available and accessible, but the generator still fails to use/integrate it correctly. |

Do **not** invent new primary classes. Offline secondary categories from the
preliminary experiment (`oracle_recovered_7b`, capability-limitation, unresolved
hard case) are **out of scope** for this Train delivery unless a separate
secondary-label pass is commissioned later.

---

## 2. What you may look at (offline only)

For each row you may use:

- `query`
- `initial_answer` (frozen C1 / Qwen-A prediction)
- `evidence` (Top-5: `rank`, `passage_id`, `text`, `score`)
- `annotation_reference.gold_answer`
- `annotation_reference.gold_passage_id`
- `annotation_reference.gold_evidence_text` (present only when the gold passage
  appears in Top-5; otherwise may be `null`)
- `annotation_reference.gold_in_top1` / `gold_in_top3` / `gold_in_top5`

These gold fields are **offline supervision references**. They exist so that
annotators can judge failure type. They must **never** be copied into:

- Diagnosis runtime input (`DiagnosisInput`)
- Probe prompts
- Controller inputs
- any online feature that the deployed Diagnosis model would see

Runtime Diagnosis may only observe: `sample_id`, `dataset`, `query`,
`evidence`, `initial_answer`.

---

## 3. Hard rules

1. **One label per sample.** Choose exactly one of the four primary labels.
2. **No class balancing.** Do not change a label to improve class counts.
3. **No automatic EM/F1 override.** Exact Match / token F1 (if you compute them
   locally) are optional references only. Semantic acceptability decides
   `correct`, not string match alone.
4. **Do not rely on `gold_in_top5` alone.** A gold *passage id* missing from
   Top-5 does not automatically mean insufficiency: another Top-5 passage may
   still contain equivalent supporting facts. Conversely, `gold_in_top5=true`
   does not automatically mean utilisation failure.
5. **Do not invent C2/C3 outcomes.** This package does not include Cleaned (C2)
   or Oracle (C3) generations. Primary labels do not require them (see §7).
6. **Leave `label` as `null` until you decide.** Write the final primary label
   into the agreed annotation column/field only after judgment.

---

## 4. Recommended decision procedure

Work top-down:

### Step A — Is `initial_answer` acceptable?

Judge **semantic** correctness against `query` and `gold_answer` (and, if
useful, the evidence). Short paraphrases, normalized spans, and equivalent
entities may still be `correct` even if the string differs from gold.

- If yes → label **`correct`**. Stop.
- If no → continue.

### Step B — Does Top-5 contain enough information to answer?

Read all five passages. Ask: *Could a careful reader produce an acceptable
answer from this Top-5 alone?*

- If **no** (needed facts missing, only tangential mentions, contradictory
  fragments with no usable support) → **`evidence_insufficiency`**. Stop.
- If **yes** → continue.

`gold_in_top5=false` is a **hint** that insufficiency is likely, but confirm
by reading the texts. If equivalent support exists without the gold passage id,
do **not** force insufficiency.

### Step C — Is the needed information accessible, or buried/noisy?

The support is somewhere in Top-5, but check presentation:

- buried under long irrelevant text
- conflicting passages that obscure the right span
- correct fact only in a low-ranked, easy-to-miss fragment
- answerable only by stitching scattered weak hints that a short-context reader
  would reasonably miss

If the failure is best explained by **access / presentation / noise / ranking**,
not by a clear, easy-to-use supporting span → **`evidence_accessibility`**.

If the supporting span is **clear, sufficient, and reasonably easy to find**,
and the model still answered wrong → **`evidence_utilisation`**.

### Boundary: accessibility vs utilisation

| Prefer `evidence_accessibility` when… | Prefer `evidence_utilisation` when… |
|---|---|
| Support exists but is hard to locate or contradicted by louder passages | Support is explicit, contiguous, and easy to quote |
| Ranking/noise would mislead a careful human skimming Top-5 | A careful human would almost certainly find the answer span |
| Multiple passages conflict and the correct one is not salient | Little conflict; the right span is obvious |

When still unsure between accessibility and utilisation, see §5.

---

## 5. Uncertain samples

If you cannot decide after reading query, C1 answer, Top-5, and gold reference:

1. Do **not** guess to force a class balance.
2. Prefer a conservative mark in your worksheet (e.g. `needs_manual_review` /
   uncertain flag) **without** inventing a fifth primary runtime label.
3. Escalate borderline accessibility vs utilisation cases to a second annotator
   or adjudicator.
4. Leave the delivery `label` field as `null` until adjudication finishes.

Historical NQ3216 trusted training excluded rows that still needed manual
review. Keep the same discipline: uncertain ≠ invent a label.

---

## 6. Abstract examples (not project data)

These are **synthetic illustrations**. Do not treat them as real NQ Train rows.

### `correct`

- Q: “Who wrote *Pride and Prejudice*?”  
  Top-5 clearly states Jane Austen; C1 answers “Jane Austen”. → `correct`
- Q: “Capital of France?”  
  C1 answers “Paris” (gold also Paris) even if wording differs slightly. → `correct`

### `evidence_insufficiency`

- Q asks for a 2019 award winner; Top-5 only discusses the person’s early life
  and never mentions the award. → `evidence_insufficiency`
- Q asks for a specific founding year; Top-5 never states any year. →
  `evidence_insufficiency`

### `evidence_accessibility`

- The correct year appears once inside a long, mostly irrelevant rank-5
  paragraph, while ranks 1–3 loudly discuss a different similarly named entity.
  A careful reader *could* find it, but presentation/noise makes access hard.
  C1 answers the wrong entity’s year. → `evidence_accessibility`
- Two passages give conflicting populations; the gold-consistent figure is
  buried mid-paragraph. C1 picks the louder wrong figure. →
  `evidence_accessibility`

### `evidence_utilisation`

- Rank-1 begins with a short, explicit sentence that states the answer span
  clearly; no meaningful conflict. C1 still returns a different wrong span.
  → `evidence_utilisation`
- Top-5 quotes the exact short answer in a clean list-like sentence; C1
  abstains or invents an unsupported alternative. → `evidence_utilisation`

---

## 7. C2 / Oracle dependency (primary labels)

For **primary** Diagnosis v2 labels, historical NQ500 / NQ3216 annotation used:

- C1 prediction
- Top-5 evidence text
- gold answer / gold reference
- human or AI **semantic** judgment

They did **not** require Cleaned (C2) or Oracle (C3) generations to choose among
the four primary labels.

C3 / larger-model outcomes were used only for **offline secondary** analysis
under utilisation (e.g. whether an oracle context recovered the answer). That
secondary taxonomy is **not** part of this Train primary-label delivery.

Therefore:

- Annotators **can** assign the four primary labels with the fields in this
  package.
- Distinguishing accessibility vs utilisation remains a **judgment call** on
  Top-5 readability/salience; C2 is not a substitute for that judgment unless
  a future protocol explicitly redefines labels around C2 outcomes (this guide
  does not).

---

## 8. Delivery package layout

```
data/annotations/nq_train_delivery/
  manifest.json
  nq_train_annotation_000.jsonl
  nq_train_annotation_001.jsonl
  ...
```

- Rows are sharded in stable `sample_id` order (~5000 per shard).
- `label` is `null` in every delivered row until annotation.
- Merge all shards to recover the full 74,097-sample set (see `manifest.json`
  for per-shard SHA256 checksums).

---

## 9. Output expectation

When annotation is finished for a shard, each sample should have:

- `label` ∈ {`correct`, `evidence_insufficiency`, `evidence_accessibility`,
  `evidence_utilisation`}
- optional review flags (worksheet-specific), never a new primary class name

Downstream conversion should produce records compatible with:

```text
{ "input": DiagnosisInput, "target_label": <one of four labels> }
```

without putting `annotation_reference` into `DiagnosisInput`.
