# Phase 6-3 delivery

Read `evaluation/report.md` for per-label and overall EM/F1, 95% CIs,
fixed/broken counts, guard analysis, validation-weighted effects, latency,
provenance, and limitations. `evaluation/report.json` contains full precision.
`evaluation/scoring/per_sample.jsonl` preserves all 2,381 paired scores.
`evaluation/raw/gpu_results/` is the normalized extraction of the GPU result ZIP,
including unchanged source bytes, checksummed result lines, receipt and sessions.
The scoped `.gitattributes` rule preserves their Windows CRLF bytes and permits
CR at end of line for whitespace checks; the raw evidence is not reformatted.

The 2,313 results are complete, unique, ordered, and verified against win2 input,
run metadata, per-record checksums, and frozen answer guard. The 68 unchanged
rows retain baseline answers. The original archive used backslash paths.

Source archive SHA256:

- Desktop and Downloads `gpu_results.zip` (identical): `5a7b92da20e2e3ed227b475cd12d03691ad9e986e6bbcfd35b7f89cdc9ab602e`
- `nq_p320_gpu_generation_win2.zip`: `7f826237b24e045d65a6717b15e339cb38e043dc58f7383d62665e12fdea0069`
- `nq_c1_reproduction_win.zip`: `5bcce8e596dadcb564f761435e53eb4aa5e7f69ce31ddecd9b6468ed4ac98859`

Both Windows source fixes are exact copies of the corresponding win2 files.
They add the msvcrt single-writer lock and model/tokenizer commit fallback with
receipt provenance. Prompt, generator, evaluation normalization, and guard
source were not edited. The evaluator explicitly accepts the pinned win2
manifest with `--package-manifest`; no check is disabled. C1 reproduction has
not been executed. Its package was only inspected and hashed.

Validation weights were recomputed from the three validation-only Git blobs
listed in `evaluation/routing_weights.json`, revision
`338ff34b039ed367220b0b916b48c4f7fc715e20`. They reproduce Phase 5-7's
19/49, 23/49, 0/49, 7/49 composition. No test records were read.

Reproduce from this repository, using fresh output paths:

```bash
unzip ~/Desktop/nq_p320_gpu_generation_win2.zip -d /private/tmp/phase6_3_bundle_recheck
python3 scripts/score_phase6_3.py \
  --results-zip ~/Desktop/gpu_results.zip \
  --bundle-dir /private/tmp/phase6_3_bundle_recheck/gpu_bundle \
  --reference ../../outputs/phase6_1_gpu_package_20261020/mac_offline_only/references.jsonl \
  --routing-repo . \
  --output-dir /private/tmp/phase6_3_score_recheck
python3 -m unittest discover -s tests -v
git diff --check
```

Full suite: 180 tests passed (`tests.txt`). Added tests simulate unavailable
fcntl, nonblocking Windows lock contention, missing tokenizer commit, unresolved
model commit, altered source/config/manifest rejection, Windows ZIP paths and
traversal rejection, and paired guard scoring with unchanged denominators.
These are model-free tests, not a real Windows or C1 reproduction run.

Independent point-estimate recount: 857 baseline EM-correct, 892 final
EM-correct; 108 fixes minus 73 breaks = 35 net. F1 0.510204749285174 to
0.5216536813890935. Latency order matches original input order. Session records
show a 20-row smoke run followed by a resume of 2,293 rows, with zero remaining.

The commit is local only; no push was performed. Awaiting acceptance.
