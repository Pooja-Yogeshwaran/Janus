"""Feature 11: refusal-retry count.

A running count of how many times the assistant has refused in this
conversation, read from shadow-space (``ConversationState.shadow_refusals``:
``prior_branches`` prepended ahead of the live thread) -- so that editing away
a refused turn in a UI that supports message editing doesn't silently reset
this counter to 0 for callers who pass their edit history in via
``prior_branches``. See :mod:`pyjanus_guard.incremental`.

Evaluated on every turn regardless of role -- not just assistant turns -- so
it isn't blind to conversation shapes where the live thread has no assistant
turn at all (e.g. exactly the edited-history case above: prior_branches holds
the refusals, the live thread is a single fresh-looking user turn). It still
only *re-fires* on assistant turns after the first evaluation, since that's
the only time the count can change; this avoids emitting a redundant flag on
every subsequent turn once the count has already crossed the flag threshold.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from .base import ConversationState, Feature, FeatureResult


class RefusalRetryCountFeature(Feature):
    name = "refusal_retry_count"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        first_evaluation = self.name not in state.feature_scores
        if message.get("role") != "assistant" and not first_evaluation:
            return None

        offset = state.shadow_offset
        shadow_index = offset + turn_index
        refused_shadow_indices = sorted(
            i for i, v in state.shadow_refusals.items() if v and i <= shadow_index
        )
        count = len(refused_shadow_indices)

        if count == 0:
            return FeatureResult(
                raw_value=0.0, normalized_score=0.0, reason="no refusals yet",
                turn_indices=[turn_index],
            )

        cap = cfg_params.get("saturation_count", 3)
        normalized = min(1.0, count / cap)

        live_refused_indices = [i - offset for i in refused_shadow_indices if i >= offset]
        from_prior_branches = count - len(live_refused_indices)

        reason = f"{count} refusal(s) so far in this conversation"
        if from_prior_branches:
            reason += f" (including {from_prior_branches} from prior_branches/edit history)"

        return FeatureResult(
            raw_value=float(count),
            normalized_score=normalized,
            reason=reason,
            turn_indices=live_refused_indices or [turn_index],
        )


__all__ = ["RefusalRetryCountFeature"]
