"""Feature 10: anchoring score.

Does the current user turn quote or build directly on the assistant's own
prior output? A common crescendo move is to get the assistant to produce
something borderline-but-innocuous, then ask it to "continue that", "expand
on what you just wrote", or "now do the same thing but for X" -- anchoring the
next ask to the model's own words rather than a fresh request.
"""

from __future__ import annotations

import re
from typing import Any, Dict, Optional, Set

from .base import ConversationState, Feature, FeatureResult

_QUOTE_MARKERS = re.compile(
    r"\byou (?:just )?(?:said|wrote|mentioned|described|listed)\b|"
    r"\bas you (?:said|mentioned|wrote)\b|"
    r"\b(?:continue|expand on|build on|elaborate on) (?:that|what you|the above|your (?:last|previous))\b|"
    r"\bnow do the same (?:thing )?for\b",
    re.I,
)

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.I)


def _shingles(text: str, n: int = 4) -> Set[str]:
    tokens = [t.lower() for t in _TOKEN_RE.findall(text)]
    if len(tokens) < n:
        return {" ".join(tokens)} if tokens else set()
    return {" ".join(tokens[i : i + n]) for i in range(len(tokens) - n + 1)}


class AnchoringFeature(Feature):
    name = "anchoring"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "user":
            return None

        prev_assistant_idx = None
        for i in range(turn_index - 1, -1, -1):
            if state.messages[i].get("role") == "assistant":
                prev_assistant_idx = i
                break
        if prev_assistant_idx is None:
            return None

        current_text = message.get("content", "")
        prior_text = state.messages[prev_assistant_idx].get("content", "")

        current_shingles = _shingles(current_text)
        prior_shingles = _shingles(prior_text)
        overlap = 0.0
        if current_shingles and prior_shingles:
            shared = current_shingles & prior_shingles
            overlap = len(shared) / max(1, min(len(current_shingles), len(prior_shingles)))

        explicit_quote = bool(_QUOTE_MARKERS.search(current_text))
        score = min(1.0, max(overlap, 0.6 if explicit_quote else 0.0) + (0.2 if explicit_quote and overlap > 0 else 0.0))

        if score <= 0:
            return FeatureResult(
                raw_value=0.0, normalized_score=0.0, reason="no anchoring to prior assistant turn",
                turn_indices=[turn_index],
            )

        reason_bits = []
        if overlap > 0:
            reason_bits.append(f"{overlap * 100:.0f}% phrase overlap")
        if explicit_quote:
            reason_bits.append("explicit reference to assistant's prior output")

        return FeatureResult(
            raw_value=score,
            normalized_score=score,
            reason=(
                f"anchored to assistant turn {prev_assistant_idx}'s own output "
                f"({', '.join(reason_bits)})"
            ),
            turn_indices=[prev_assistant_idx, turn_index],
        )


__all__ = ["AnchoringFeature"]
