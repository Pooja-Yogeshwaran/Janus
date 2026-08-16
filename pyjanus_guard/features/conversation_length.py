"""Feature 14: conversation length outlier (structural).

Gradual multi-turn attacks need turns to work with -- FITD/crescendo/CoA
transcripts run measurably longer than typical single- or few-shot chat
sessions. Only unusually *long* conversations are flagged; short ones are not
penalized (most real conversations are short, and brevity isn't a risk signal).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import ConversationState, Feature, FeatureResult


class ConversationLengthOutlierFeature(Feature):
    name = "conversation_length_outlier"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        length = turn_index + 1
        mean = cfg_params.get("baseline_mean", state.config.baseline_length_mean)
        std = cfg_params.get("baseline_std", state.config.baseline_length_std)
        if std <= 0:
            return None

        z = (length - mean) / std
        if z <= 1.0:
            return FeatureResult(
                raw_value=z, normalized_score=0.0,
                reason=f"conversation length {length} is within the expected baseline",
                turn_indices=[turn_index],
            )

        normalized = min(1.0, (z - 1.0) / 2.0)
        return FeatureResult(
            raw_value=z,
            normalized_score=normalized,
            reason=(
                f"conversation length ({length} messages) is {z:.1f} standard "
                f"deviations above the expected baseline ({mean:.0f}±{std:.0f})"
            ),
            turn_indices=[turn_index],
        )


__all__ = ["ConversationLengthOutlierFeature"]
