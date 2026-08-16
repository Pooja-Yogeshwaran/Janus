"""Feature 7: topic drift.

Embedding distance from the conversation's opening turn to the current turn,
plus monotonic-trend detection across the running distance sequence -- a
single far-off turn can be a topic change; a *steadily increasing* distance
across several turns is the gradual-escalation signature crescendo/CoA
attacks rely on.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..embeddings import cosine_distance
from .base import ConversationState, Feature, FeatureResult

_MEMORY_KEY = "topic_drift_distances"


def _is_monotonic_rising(distances: list[float], slack: float = 0.02) -> float:
    """Fraction of consecutive steps that rise (or hold). Returns 0..1."""
    if len(distances) < 2:
        return 0.0
    rising = sum(
        1 for a, b in zip(distances, distances[1:]) if b >= a - slack
    )
    return rising / (len(distances) - 1)


class TopicDriftFeature(Feature):
    name = "topic_drift"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "user":
            return None

        user_turns = state.user_indices(up_to=turn_index)
        if len(user_turns) < 2:
            return None  # need a start-of-conversation anchor

        anchor_idx = user_turns[0]
        distance = cosine_distance(state.embed(anchor_idx), state.embed(turn_index))
        distance = max(0.0, min(1.0, distance))

        history = state.feature_memory.setdefault(_MEMORY_KEY, [])
        history.append(distance)

        trend = _is_monotonic_rising(history)
        min_turns = cfg_params.get("min_turns_for_trend", 3)
        trending = len(history) >= min_turns and trend >= cfg_params.get("trend_ratio", 0.7)

        normalized = distance
        if trending:
            normalized = min(1.0, distance * 1.25)

        reason = (
            f"topic has drifted {distance:.2f} (cosine distance) from the opening "
            f"turn with a rising trend across {len(history)} turns"
            if trending
            else f"topic distance {distance:.2f} from opening turn {anchor_idx}"
        )

        return FeatureResult(
            raw_value=distance,
            normalized_score=normalized,
            reason=reason,
            turn_indices=[anchor_idx, turn_index],
        )


__all__ = ["TopicDriftFeature"]
