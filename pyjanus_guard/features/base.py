"""Feature protocol shared by every scoring function.

Each feature is independently enable/disable-able and reweight-able (see
:class:`pyjanus_guard.config.FeatureConfig`) and feeds one normalized score in
[0, 1] into the aggregator (see :mod:`pyjanus_guard.aggregator`). A feature
may apply to only some turns (e.g. refusal detection only looks at assistant
turns) -- returning ``None`` from :meth:`Feature.score_turn` means "no new
information this turn," and the aggregator carries the feature's previous
score forward so ``turn_scores`` stays continuous rather than yo-yoing to 0.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..config import JanusConfig
from ..types import Message


@dataclass
class ConversationState:
    """Mutable state threaded through every feature as turns arrive.

    The same state object backs both batch and incremental scoring: batch
    mode is implemented as "construct one state, replay all messages through
    it" (see core.py), which is what guarantees the two modes can never drift
    out of sync with each other.
    """

    config: JanusConfig
    messages: List[Message] = field(default_factory=list)
    prior_branches: List[Message] = field(default_factory=list)

    # turn_index -> embedding vector, cached lazily
    embeddings: Dict[int, List[float]] = field(default_factory=dict)
    # turn_index -> True (refused) / False (complied) -- assistant turns only
    refusals: Dict[int, bool] = field(default_factory=dict)
    # turn_index -> "full_comply" | "partial_comply" | "refuse" -- assistant turns only
    compliance: Dict[int, str] = field(default_factory=dict)

    # "Shadow" history = prior_branches (discarded/edited-out turns) prepended
    # ahead of the live thread, index-aligned as one continuous sequence:
    # shadow_messages[i] for i < len(prior_branches) is a prior_branches
    # entry; for i >= len(prior_branches) it mirrors messages[i - offset].
    # Only reformulation_after_refusal and refusal_retry_count read this --
    # every other feature still only sees `messages`, the live thread.
    shadow_messages: List[Message] = field(default_factory=list)
    shadow_refusals: Dict[int, bool] = field(default_factory=dict)
    shadow_embeddings: Dict[int, List[float]] = field(default_factory=dict)

    # feature_name -> latest normalized score in [0, 1], carried forward
    # across turns the feature doesn't fire on.
    feature_scores: Dict[str, float] = field(default_factory=dict)
    # arbitrary per-feature scratch space (e.g. running counters), keyed by feature name
    feature_memory: Dict[str, Any] = field(default_factory=dict)

    # turn_index -> {feature_name: normalized_score}, populated only for
    # features that produced a *fresh* (non-None) result on that exact turn.
    # Unlike `feature_scores` (which holds each feature's latest known score,
    # carried forward across turns it doesn't apply to -- e.g.
    # refusal_detection's value is stale/irrelevant on a user turn),
    # this dict lets a later feature ask "what did the turn-level features
    # actually say about turn N specifically." Purely additive bookkeeping:
    # populated in incremental.py, read only by escalation_watchlist.py --
    # every pre-existing feature is unaffected by this field's existence.
    turn_feature_scores: Dict[int, Dict[str, float]] = field(default_factory=dict)
    # Turns that cleared escalation_watchlist's "mild concern" floor on at
    # least one turn-level feature, in conversation order. Entries are
    # `escalation_watchlist.WatchlistEntry` instances; typed as `List[Any]`
    # here (rather than importing that class) to avoid a circular import,
    # since escalation_watchlist.py imports ConversationState from this
    # module. Populated and consumed only by escalation_watchlist.py.
    watchlist: List[Any] = field(default_factory=list)

    def embed(self, turn_index: int) -> List[float]:
        if turn_index not in self.embeddings:
            text = self.messages[turn_index].get("content", "")
            self.embeddings[turn_index] = list(self.config.embed_fn(text))
        return self.embeddings[turn_index]

    def shadow_embed(self, shadow_index: int) -> List[float]:
        if shadow_index not in self.shadow_embeddings:
            text = self.shadow_messages[shadow_index].get("content", "")
            self.shadow_embeddings[shadow_index] = list(self.config.embed_fn(text))
        return self.shadow_embeddings[shadow_index]

    @property
    def shadow_offset(self) -> int:
        """Number of prior_branches entries prepended ahead of the live
        thread in shadow-space. live turn_index N == shadow_index N + offset.
        """
        return len(self.prior_branches)

    def user_indices(self, up_to: Optional[int] = None) -> List[int]:
        limit = len(self.messages) if up_to is None else up_to + 1
        return [i for i in range(limit) if self.messages[i].get("role") == "user"]

    def assistant_indices(self, up_to: Optional[int] = None) -> List[int]:
        limit = len(self.messages) if up_to is None else up_to + 1
        return [i for i in range(limit) if self.messages[i].get("role") == "assistant"]


@dataclass
class FeatureResult:
    raw_value: float
    normalized_score: float  # in [0, 1]
    reason: str
    turn_indices: List[int]


class Feature(ABC):
    name: str = "base_feature"

    @abstractmethod
    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        """Compute this feature's contribution for the newly-appended turn at
        ``turn_index``. Return ``None`` if the feature has nothing new to say
        about this turn (e.g. a user-turn-only feature seeing an assistant
        turn).
        """
        raise NotImplementedError


__all__ = ["ConversationState", "FeatureResult", "Feature"]
