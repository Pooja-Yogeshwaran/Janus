"""Feature 9: reformulation-after-refusal similarity.

Signature move of FITD/crescendo-style attacks: get refused, then come back
softened or reworded but aimed at the same underlying ask. This compares the
current user turn against the most recent user turn whose follow-up was
refused, via the pluggable embedding function.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..embeddings import cosine_similarity
from .base import ConversationState, Feature, FeatureResult


def _most_recently_refused_user_turn(state: ConversationState, before_shadow_index: int) -> Optional[int]:
    """Walk backward in shadow-space (prior_branches + live thread) from just
    before `before_shadow_index` for the last assistant refusal, then return
    the shadow index of the user turn that prompted it. Searching shadow-space
    rather than the live thread means a refusal hidden in edited-out
    prior_branches is still visible here.
    """
    for i in range(before_shadow_index - 1, -1, -1):
        if state.shadow_messages[i].get("role") == "assistant" and state.shadow_refusals.get(i):
            for j in range(i - 1, -1, -1):
                if state.shadow_messages[j].get("role") == "user":
                    return j
    return None


class ReformulationAfterRefusalFeature(Feature):
    name = "reformulation_after_refusal"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "user":
            return None

        offset = state.shadow_offset
        shadow_index = offset + turn_index

        refused_shadow_idx = _most_recently_refused_user_turn(state, shadow_index)
        if refused_shadow_idx is None:
            return None

        sim = cosine_similarity(state.shadow_embed(refused_shadow_idx), state.shadow_embed(shadow_index))
        sim = max(0.0, sim)

        if refused_shadow_idx >= offset:
            refused_live_idx = refused_shadow_idx - offset
            turn_indices = [refused_live_idx, turn_index]
            location = f"turn {refused_live_idx}"
        else:
            # the refused ask lives only in prior_branches (edited out of the
            # live thread) -- there's no live index to point at.
            turn_indices = [turn_index]
            location = "a prior refused turn (from edit history)"

        return FeatureResult(
            raw_value=sim,
            normalized_score=sim,
            reason=(
                f"{sim * 100:.0f}% similar to {location} "
                "-- flagged as reformulation-after-refusal"
            ),
            turn_indices=turn_indices,
        )


__all__ = ["ReformulationAfterRefusalFeature"]
