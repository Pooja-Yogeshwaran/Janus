"""Feature 13: convergence-to-target.

OFF by default (see config.DEFAULT_FEATURE_NAMES / _OFF_BY_DEFAULT) and a
true no-op unless the caller supplies ``JanusConfig.reference_embeddings``: a
dict of harm-category name -> reference centroid embedding, produced however
the caller likes (e.g. averaged embeddings of known-harmful prompts in a
category). Janus core ships no hardcoded harm-content embeddings -- shipping
a fixed harm taxonomy baked into an MIT-licensed package is both a legal
liability and a moving target the maintainers can't keep current, so this is
deliberately left for integrators to supply.

When configured, tracks similarity-to-nearest-centroid across turns and flags
a *rising* trend toward some target category, not just a single close turn.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..embeddings import cosine_similarity
from .base import ConversationState, Feature, FeatureResult

_MEMORY_KEY = "convergence_similarities"


class ConvergenceToTargetFeature(Feature):
    name = "convergence_to_target"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "user":
            return None

        references = state.config.reference_embeddings
        if not references:
            return None  # pluggable, no-op without caller-supplied centroids

        vec = state.embed(turn_index)
        best_category, best_sim = None, -1.0
        for category, ref_vec in references.items():
            sim = cosine_similarity(vec, ref_vec)
            if sim > best_sim:
                best_category, best_sim = category, sim
        best_sim = max(0.0, best_sim)

        history = state.feature_memory.setdefault(_MEMORY_KEY, [])
        history.append(best_sim)

        trending = False
        if len(history) >= cfg_params.get("min_turns_for_trend", 3):
            recent = history[-cfg_params.get("min_turns_for_trend", 3):]
            rising_steps = sum(1 for a, b in zip(recent, recent[1:]) if b >= a - 0.02)
            trending = rising_steps == len(recent) - 1

        normalized = min(1.0, best_sim * 1.15) if trending else best_sim

        reason = (
            f"turn is {best_sim * 100:.0f}% similar to reference category "
            f"'{best_category}', trending upward across {len(history)} turns "
            "-- convergence toward target"
            if trending
            else f"turn is {best_sim * 100:.0f}% similar to reference category '{best_category}'"
        )

        return FeatureResult(
            raw_value=best_sim,
            normalized_score=normalized,
            reason=reason,
            turn_indices=[turn_index],
        )


__all__ = ["ConvergenceToTargetFeature"]
