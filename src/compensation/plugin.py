"""One call to build the evidence-refinement plugin the host pipeline registers.

Assembling this module by hand means wiring five parts -- a candidate pool, a
cross-encoder, a generator, a config and the Enhancement adapter -- and the
measured difference between wiring them well and wiring them plausibly is
large. This module hands back the configuration the experiments actually
support, so a caller does not have to rediscover it.

What the evidence says (held-out accessibility samples, 2,367, exact match):

| setup                                      | passages | EM     |
|--------------------------------------------|----------|--------|
| keep the original answer                   | 5.00     | 0.0672 |
| random 5 of 50                             | 5.00     | 0.0921 |
| rerank inside the frozen Top-5             | 5.00     | 0.1107 |
| shuffle inside the frozen Top-5            | 5.00     | 0.1187 |
| cross-encoder 5 of 50, no threshold        | 5.00     | 0.1618 |
| cross-encoder 5 of 50, ratio 0.4           | 3.54     | 0.1825 |
| **cross-encoder 5 of 50, ratio 0.6 (this)**| **2.77** | **0.2003** |
| cross-encoder 5 of 50, ratio 0.8           | 1.90     | 0.2066 |
| the single best passage of 50              | 1.00     | 0.1990 |

Two decisions come out of that table. Reordering a fixed five is
indistinguishable from shuffling it, while *selecting* five out of fifty beats
every fixed-pool arm (p = 2.8e-12 against reranking inside the Top-5), so the
deeper pool is the default and the narrow version has to be asked for with
`top_n=None`. And handing over fewer than five, chosen by a floor relative to
each question's own best score, beats handing over five (p = 1.6e-08 at ratio
0.6), so a floor is on by default.

The floor is 0.6 rather than the nominally better 0.8 because 0.6-to-1.0 is a
plateau, not a peak: 0.8 is not distinguishable from 0.6 (p = 0.24, 95% CI
[-0.3pp, +1.7pp]) and neither is keeping one passage (p = 0.17 against 0.8).
Picking the sample maximum off a flat region is fitting the test set, so the
default takes the conservative end of the plateau, which keeps 2.77 passages
instead of 1.90, loses its only answer-bearing passage on half as many samples,
and makes the generator refuse less often (31 refusals against 41 and 46).

Registration:

    from src.compensation import EnhancementExecutor
    from src.compensation.plugin import register_refinement_plugin

    executor = EnhancementExecutor()
    register_refinement_plugin(executor)        # frozen local generator
    decision = run_controller(item, classifier, probe)
    result = executor.execute(EnhancementRequest.from_runtime(...))

`generate_fn` accepts any `prompt -> str` callable, so a hosted endpoint works
as well as the frozen local Qwen (`src/generation/backend.py` builds one).

Both the reranker and the pool index are gitignored build artifacts. When one
is missing this raises with the command that rebuilds it rather than quietly
falling back to a setup worth a third as much.
"""

from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from src.compensation.candidates import DEFAULT_CORPUS, DEFAULT_IDS, DEFAULT_INDEX, IndexCandidateProvider
from src.compensation.evidence_refinement import EvidenceRefiner
from src.compensation.refinement_adapter import EvidenceRefinementAdapter
from src.compensation.refinement_schema import RefinementConfig
from src.compensation.rerank import scorer_from_name
from src.controller.schema import ControllerAction

ROOT = Path(__file__).resolve().parents[2]
TRAINED_RERANKER = ROOT / "models" / "reranker" / "accessibility_v1"
POOL_CORPUS = ROOT / "data" / "processed" / "retrieval" / "nq15_pool_corpus.jsonl"
POOL_INDEX = ROOT / "data" / "processed" / "indexes" / "nq15_pool_bge_base.index"
POOL_IDS = ROOT / "data" / "processed" / "indexes" / "nq15_pool_bge_base_ids.jsonl"
VALIDATED_TOP_N = 50
VALIDATED_KEEP_K = 5
VALIDATED_MIN_SCORE_RATIO = 0.6

REBUILD_RERANKER = (
    "python scripts/normalize_nq15_annotations.py && "
    "python scripts/build_rerank_training_data.py && "
    "python scripts/train_rerank_accessibility.py"
)
REBUILD_POOL = "python scripts/build_nq15_pool_index.py"


def validated_config(reranker_model: str | Path = TRAINED_RERANKER) -> RefinementConfig:
    """The config the measurements support: at most five, a relative floor, no pruning."""
    return RefinementConfig(
        keep_k=VALIDATED_KEEP_K,
        min_score_ratio=VALIDATED_MIN_SCORE_RATIO,
        reranker_model=str(reranker_model),
    )


def _require(path: Path, what: str, rebuild: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{what} is missing at {path}. It is a build artifact, not in git. Rebuild it with:\n  {rebuild}")


def build_refiner(
    generate_fn: Callable[[str], str] | None = None,
    config: RefinementConfig | None = None,
    top_n: int | None = VALIDATED_TOP_N,
    corpus_path: Path = POOL_CORPUS,
    index_path: Path = POOL_INDEX,
    ids_path: Path = POOL_IDS,
    pruner=None,
) -> EvidenceRefiner:
    """The refiner with the validated wiring.

    `generate_fn=None` loads the frozen local generator (Qwen2.5-7B, 4-bit NF4,
    greedy). `top_n=None` drops the deeper pool and reranks only the evidence
    already shown, which the experiments show is worth far less.
    """
    config = config or validated_config()
    if not str(config.reranker_model).startswith(("shuffle:", "identity")):
        _require(Path(config.reranker_model), "The trained reranker", REBUILD_RERANKER)

    provider = None
    if top_n:
        for path, what in ((corpus_path, "The pool corpus"), (index_path, "The pool index"), (ids_path, "The pool ids")):
            _require(path, what, REBUILD_POOL)
        provider = IndexCandidateProvider.from_frozen_artifacts(
            corpus_path=corpus_path, index_path=index_path, ids_path=ids_path, top_n=top_n
        )

    if generate_fn is None:
        return EvidenceRefiner.from_frozen_defaults(config=config, candidate_provider=provider)
    return EvidenceRefiner(
        scorer_from_name(config.reranker_model),
        generate_fn,
        config=config,
        candidate_provider=provider,
        pruner=pruner,
    )


def build_refinement_plugin(**kwargs) -> EvidenceRefinementAdapter:
    """The `Enhancement` the executor dispatches to. Takes the same arguments as `build_refiner`."""
    return EvidenceRefinementAdapter(build_refiner(**kwargs))


def register_refinement_plugin(executor, **kwargs) -> EvidenceRefinementAdapter:
    """Build the plugin and register it on the `evidence_refinement` lane."""
    plugin = build_refinement_plugin(**kwargs)
    executor.register(ControllerAction.EVIDENCE_REFINEMENT, plugin)
    return plugin
