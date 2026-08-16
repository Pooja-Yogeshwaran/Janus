"""Feature 8: step size between consecutive user turns.

Complements topic_drift (distance from conversation start): a conversation
can drift a long way from its opener yet still take small, plausible steps
turn-to-turn. A large single jump between two *consecutive* user turns is its
own signal -- an abrupt pivot, not a gradual one.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..embeddings import cosine_distance
from .base import ConversationState, Feature, FeatureResult


class StepSizeFeature(Feature):
    name = "step_size"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "user":
            return None

        user_turns = state.user_indices(up_to=turn_index)
        if len(user_turns) < 2:
            return None

        previous_idx = user_turns[-2]
        distance = cosine_distance(state.embed(previous_idx), state.embed(turn_index))
        distance = max(0.0, min(1.0, distance))

        return FeatureResult(
            raw_value=distance,
            normalized_score=distance,
            reason=f"large step ({distance:.2f} cosine distance) from previous user turn {previous_idx}",
            turn_indices=[previous_idx, turn_index],
        )


__all__ = ["StepSizeFeature"]
