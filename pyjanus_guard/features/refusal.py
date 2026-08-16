"""Feature 1: refusal detection.

Regex/heuristic pass by default, with an optional pluggable LLM-judge
callable (``config.refusal_judge: (context, response) -> Optional[bool]``)
that can override the heuristic per-turn. Returning ``None`` from the judge
means "abstain," and Janus falls back to the heuristic for that turn --
callers don't need their judge to handle every case.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

from ..types import Message
from ._refusal_patterns import heuristic_is_refusal
from .base import ConversationState, Feature, FeatureResult


def _context_text(state: ConversationState, turn_index: int) -> str:
    prior: list[Message] = state.messages[:turn_index]
    return "\n".join(f"{m.get('role', '')}: {m.get('content', '')}" for m in prior)


class RefusalDetectionFeature(Feature):
    name = "refusal_detection"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "assistant":
            return None

        text = message.get("content", "")
        refused = heuristic_is_refusal(text)
        judge = state.config.refusal_judge
        judge_used = False
        if judge is not None:
            verdict = judge(_context_text(state, turn_index), text)
            if verdict is not None:
                refused = bool(verdict)
                judge_used = True

        state.refusals[turn_index] = refused

        if not refused:
            return FeatureResult(
                raw_value=0.0,
                normalized_score=0.0,
                reason="complied",
                turn_indices=[turn_index],
            )

        source = "llm-judge" if judge_used else "heuristic"
        return FeatureResult(
            raw_value=1.0,
            normalized_score=1.0,
            reason=f"refused ({source})",
            turn_indices=[turn_index],
        )


__all__ = ["RefusalDetectionFeature"]
