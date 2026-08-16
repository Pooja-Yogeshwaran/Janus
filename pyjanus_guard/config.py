"""User-facing configuration: which features run, how they're weighted, and
the pluggable callables (embeddings, LLM-judge fallback, harm-centroid
reference embeddings) that let callers swap in real models without touching
core scoring code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Optional, Sequence

from .embeddings import default_embed, sentence_transformer_embed, sentence_transformers_available
from .types import Thresholds

# (context_so_far, response_text) -> True (refused) / False (complied) / None (abstain, fall back to heuristic)
RefusalJudge = Callable[[str, str], Optional[bool]]

# (current_turn_text, watchlisted_turn_texts) -> True (current turn is a
# softened/escalated restatement of intent already present in one of the
# watchlisted turns) / False (it isn't) / None (abstain, fall back to the
# embedding-similarity heuristic). Same override/abstain contract as
# RefusalJudge. See escalation_watchlist.py -- feeds only the user-turn
# toned-down-reformulation check, never the escalation-trend check (trend is
# about a score history, not a relationship between two turns' text, so a
# judge classifying "is turn X a restatement of turn Y" has nothing to
# adjudicate there).
#
# **Deliberately unvalidated as of this writing** -- see README "LLM-judge
# escalation check (opt-in, unvalidated)" for why: it requires a paid LLM API
# dependency, which conflicts with this project's zero-dependency-by-default
# design, so this hook is opt-in by design (None means never called, exactly
# like RefusalJudge's default) and its *behavior* is untested by design, not
# by oversight -- no API credentials were available to validate a real
# implementation against, and faking that validation (a stub callable, or
# reusing ambient model access as if it were the project's own) was
# deliberately declined rather than reported as if it were evidence. The
# plumbing itself (trigger gating, override/abstain semantics) is unit
# tested with a stub callable -- that tests this code's wiring, not any
# judge's real classification quality.
EscalationJudge = Callable[[str, Sequence[str]], Optional[bool]]

EmbedFn = Callable[[str], Sequence[float]]


@dataclass
class FeatureConfig:
    enabled: bool = True
    weight: float = 1.0
    # per-feature score threshold (0-1 normalized) above which a Flag is emitted
    flag_threshold: float = 0.5
    params: Dict[str, Any] = field(default_factory=dict)


DEFAULT_FEATURE_NAMES = [
    "refusal_detection",
    "compliance_classification",
    "persona_injection",
    "instruction_density",
    "encoding_obfuscation",
    "code_completion_wrapping",
    "topic_drift",
    "step_size",
    "reformulation_after_refusal",
    "anchoring",
    "refusal_retry_count",
    "turn_velocity",
    "convergence_to_target",
    "conversation_length_outlier",
    "escalation_watchlist",
]

# convergence_to_target requires user-supplied reference embeddings and is a
# meaningfully different kind of signal (needs curated harm-category data),
# so it ships OFF by default even though it's implemented and testable.
_OFF_BY_DEFAULT = {"convergence_to_target"}

# escalation_watchlist (feature 15) is new and unvalidated for false-positive
# rate as of its introduction -- ships OFF until it has FPR results from
# eval/data/benign_escalating.jsonl (dev/sanity-check set) and the real
# held-out MHJ+PersonaChat split to justify enabling it, the same bar
# convergence_to_target and topic_drift/step_size were held to. See
# escalation_watchlist.py for the feature itself.
_NEW_UNVALIDATED_FEATURES = {"escalation_watchlist"}

# topic_drift and step_size are OFF by default under BOTH embedding
# backends. Originally suspected to be purely an artifact of the
# zero-dependency hash embedding (100% FPR against 200 real PersonaChat
# conversations under it -- sparse 256-dim hash buckets rarely collide, so
# cosine distance sits near-maximal almost always, on-topic or not) --
# swapping in a real sentence-transformers embedder (see
# `_resolve_default_embed_fn` below) was tried specifically to test that
# hypothesis, and it was WRONG: re-measured at 100% FPR again, unchanged,
# under the real embedder too. Real embeddings correctly recognize that
# ordinary PersonaChat chit-chat genuinely hops between unrelated personal
# topics turn to turn -- that's not an embedding-quality artifact, it's what
# casual multi-topic conversation looks like, and neither embedding backend's
# distance-from-opening-turn / consecutive-turn-distance, at the current
# threshold, separates that from attack-style drift. So this is OFF
# regardless of `embed_fn` until the feature's own design/threshold is
# revisited -- not something a better embedding model fixes on its own. See
# README "Known limitations" / "Embeddings" for the full writeup.
_EMBEDDING_DEPENDENT_FEATURES = {"topic_drift", "step_size"}  # always off; kept as a named set for feature() lookups / clarity of intent

# Default weights, set from this repo's own eval run (see eval/results/) --
# not arbitrary. With every feature weighted equally at 1.0, attack and
# benign risk_score distributions barely separated (benign 0.15-0.35 vs.
# attack 0.17-0.37): the embedding-dependent trajectory features
# (topic_drift, step_size, anchoring) are noisy under the zero-dependency
# default hashed-BOW embedding (see embeddings.py) and contribute similar
# "background" score to nearly every conversation, drowning out the
# heuristic features that are this attack family's actual defining trait
# (refused, then reformulated/escalated). Down-weighted those here and
# up-weighted refusal/compliance/reformulation/retry-count accordingly.
# Swap in a real embedding model via `JanusConfig.embed_fn` and these
# defaults are worth revisiting -- topic_drift/step_size/anchoring should
# become meaningfully more reliable and can be reweighted back up.
_DEFAULT_WEIGHTS: Dict[str, float] = {
    "refusal_detection": 1.5,
    "compliance_classification": 1.3,
    "persona_injection": 1.2,
    "instruction_density": 0.8,
    "encoding_obfuscation": 1.0,
    "code_completion_wrapping": 1.0,
    "topic_drift": 0.4,
    "step_size": 0.4,
    "reformulation_after_refusal": 1.5,
    "anchoring": 0.6,
    "refusal_retry_count": 1.3,
    "turn_velocity": 0.6,
    "convergence_to_target": 1.0,
    "conversation_length_outlier": 0.5,
    # Placeholder, not yet fit -- escalation_watchlist ships disabled (see
    # _NEW_UNVALIDATED_FEATURES), so this weight has no effect until it's
    # enabled and a real value is fit against labeled data.
    "escalation_watchlist": 1.0,
}


def _default_feature_configs() -> Dict[str, FeatureConfig]:
    off = _OFF_BY_DEFAULT | _EMBEDDING_DEPENDENT_FEATURES | _NEW_UNVALIDATED_FEATURES
    return {
        name: FeatureConfig(enabled=name not in off, weight=_DEFAULT_WEIGHTS.get(name, 1.0))
        for name in DEFAULT_FEATURE_NAMES
    }


def _resolve_default_embed_fn() -> EmbedFn:
    """Auto-detection for `JanusConfig.embed_fn`'s default: the real
    sentence-transformers embedder if the optional extra is installed
    (`pip install pyjanus_guard[embeddings]`), else the zero-dependency
    hash-based one -- exactly as before for anyone who hasn't installed it.
    """
    if sentence_transformers_available():
        return sentence_transformer_embed
    return default_embed


# Features that never require an assistant turn to fire -- i.e. still
# meaningful for a caller (or a dataset, like MHJ) that only has the
# user/attacker side of a conversation. refusal_detection,
# compliance_classification, reformulation_after_refusal,
# refusal_retry_count, and anchoring all anchor against an assistant turn and
# are structurally inert without one; see README "Real attack data: MHJ
# results" for why this distinction matters and isn't just theoretical.
#
# Deliberately just the 4 features explicitly named for this mode -- not
# every feature that happens to satisfy the "no assistant turn needed"
# criterion. persona_injection and turn_velocity also qualify structurally
# (both are role-agnostic / user-turn-only) but aren't included here; flagged
# as an open question rather than silently folded in, since expanding this
# set is a judgment call, not something implied by the criterion alone.
PROMPT_ONLY_FEATURES = {
    "instruction_density",
    "encoding_obfuscation",
    "code_completion_wrapping",
    "conversation_length_outlier",
}


def prompt_only_config(threshold: Optional[float] = None) -> "JanusConfig":
    """Config for callers (or eval runs) that only ever see the user side of
    a conversation -- no assistant-turn visibility. Enables only
    `PROMPT_ONLY_FEATURES`; every assistant-turn-dependent feature and
    topic_drift/step_size (off regardless of embedding backend -- see
    `_EMBEDDING_DEPENDENT_FEATURES`) stay disabled. Per-feature weights are
    unchanged from the main default config -- only
    `threshold` (the flagged/likely_attack cut point) differs from
    `JanusConfig()`'s defaults, since aggregate scores under this smaller
    active feature set run structurally lower and the main config's
    thresholds don't transfer.

    `threshold` defaults to the value fit on a 70% held-out split of real
    MHJ data + PersonaChat (see eval/mhj_prompt_only_eval.py and README "Real
    attack data: MHJ results" for the held-out 30% numbers that validate it,
    and for why per-feature weights were deliberately left untuned -- fitting
    six weights on ~530 examples risked more overfitting than fitting this
    one scalar).
    """
    cfg = JanusConfig()
    for name in cfg.features:
        cfg.features[name].enabled = name in PROMPT_ONLY_FEATURES
    fit_threshold = threshold if threshold is not None else _PROMPT_ONLY_FITTED_THRESHOLD
    cfg.thresholds = Thresholds(
        watch=fit_threshold * 0.72,  # same watch:flagged ratio as the main default (0.18/0.25)
        likely_attack=fit_threshold,
        flagged=fit_threshold,
    )
    return cfg


# Fit via eval/mhj_prompt_only_eval.py: threshold maximizing F1 subject to an
# FPR ceiling, on a 70% train split (MHJ conversation-level stratified by
# tactic, 375 conversations + PersonaChat, 140 conversations). The ceiling
# itself was chosen by a budget sweep (1%/3%/5%/10%) via 5-fold stratified CV
# entirely within train -- test was never consulted for this choice. CV F1
# was 0.17/0.38/0.56/0.58 respectively; 10% only beat 5% by 0.02 F1, under
# the 0.03 ambiguity margin, so the selection rule (lowest FPR unless clearly
# better -- FPR is what this project has been most protective of throughout)
# picked 5%, same as the earlier single-cap version. (An unconstrained F1-max
# search, no cap at all, converges on 0.0 -- flag everything -- a base-rate
# artifact of train being 73% "attack"; caught and rejected before shipping,
# see README.) Refit on the full train split at the chosen 5% budget, then
# validated on the untouched 30% held-out split exactly once: precision 1.00,
# recall 0.43, F1 0.60, FPR 0.00 (162 MHJ + 60 PersonaChat held-out
# conversations, N=222) -- identical to the pre-sweep number, now backed by
# CV evidence rather than an assumed cap. See README "Real attack data: MHJ
# results" for the full writeup. Update this constant if that script is
# rerun against a different/larger split.
_PROMPT_ONLY_FITTED_THRESHOLD = 0.0004


@dataclass
class JanusConfig:
    features: Dict[str, FeatureConfig] = field(default_factory=_default_feature_configs)

    # Left `None` (rather than a plain default) so `__post_init__` can
    # auto-detect which embedding backend to use -- see
    # `_resolve_default_embed_fn`. Passing `embed_fn=` explicitly to the
    # constructor skips auto-detection and is honored as-is. Never actually
    # `None` after construction -- treat it as `EmbedFn`.
    embed_fn: Optional[EmbedFn] = None

    thresholds: Thresholds = field(default_factory=Thresholds)
    refusal_judge: Optional[RefusalJudge] = None

    # feature 15 (escalation_watchlist)'s optional LLM-judge escalation
    # check. `None` by default -- never called unless a caller supplies one,
    # same opt-in contract as `refusal_judge`. See `EscalationJudge` above
    # for the signature and why this is deliberately unvalidated as of this
    # writing (README "LLM-judge escalation check (opt-in, unvalidated)").
    escalation_judge: Optional[EscalationJudge] = None

    # feature 13 (convergence_to_target): harm-category name -> reference embedding.
    # None/empty means the feature is a no-op even if explicitly enabled.
    reference_embeddings: Optional[Dict[str, Sequence[float]]] = None

    # feature 14 (conversation_length_outlier): expected length distribution,
    # in number of messages, for "normal" conversations with this integration.
    #
    # These defaults are the actual mean/stdev measured across 200 real
    # PersonaChat conversations (see eval/results/), not guesses -- the
    # previous defaults (mean=6, std=4) were unmeasured placeholders that
    # flagged 83% of ordinary benign conversations as length outliers, because
    # real chit-chat routinely runs into the teens of messages. But "normal
    # conversation length" is inherently product-specific: a customer-support
    # bot and a long-form coding assistant have very different baselines. This
    # is a rough starting point from one dataset's chat, not a
    # universal constant -- set this from your own product's actual
    # conversation-length distribution before relying on this feature.
    baseline_length_mean: float = 14.7
    baseline_length_std: float = 1.8

    def __post_init__(self) -> None:
        if self.embed_fn is None:
            self.embed_fn = _resolve_default_embed_fn()

    def feature(self, name: str) -> FeatureConfig:
        if name not in self.features:
            self.features[name] = FeatureConfig()
        return self.features[name]

    def is_enabled(self, name: str) -> bool:
        return self.feature(name).enabled


__all__ = [
    "FeatureConfig",
    "JanusConfig",
    "RefusalJudge",
    "EscalationJudge",
    "EmbedFn",
    "DEFAULT_FEATURE_NAMES",
    "PROMPT_ONLY_FEATURES",
    "prompt_only_config",
]
