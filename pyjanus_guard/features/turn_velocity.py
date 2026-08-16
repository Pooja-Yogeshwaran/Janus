"""Feature 12: turn velocity (messages/min), where timestamps are available.

Purely optional: a message dict needs an explicit numeric ``timestamp`` (unix
seconds) for this to activate. Without timestamps on both the current and
previous message, the feature is a documented no-op (returns None) rather
than guessing -- unlike the other trajectory features it has no meaningful
text-only fallback.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import ConversationState, Feature, FeatureResult


class TurnVelocityFeature(Feature):
    name = "turn_velocity"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        if turn_index == 0:
            return None

        current = state.messages[turn_index]
        previous = state.messages[turn_index - 1]
        ts_current = current.get("timestamp")
        ts_previous = previous.get("timestamp")
        if ts_current is None or ts_previous is None:
            return None

        delta_seconds = ts_current - ts_previous
        if delta_seconds <= 0:
            return None

        rate_per_min = 60.0 / delta_seconds

        # threshold: sustained sub-3-second turnaround (>20 msgs/min) reads as
        # scripted/automated rather than a human typing and reading replies.
        threshold = cfg_params.get("fast_rate_per_min", 20.0)
        normalized = min(1.0, rate_per_min / threshold) if rate_per_min > threshold * 0.5 else 0.0

        return FeatureResult(
            raw_value=rate_per_min,
            normalized_score=normalized,
            reason=f"turn velocity {rate_per_min:.1f} messages/min -- unusually fast, possibly scripted",
            turn_indices=[turn_index - 1, turn_index],
        )


__all__ = ["TurnVelocityFeature"]
