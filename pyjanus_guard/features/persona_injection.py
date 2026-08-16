"""Feature 3: persona/role/hypothetical-injection detection.

DAN-style jailbreak framing, fiction-wrapping, "pretend you're an AI without
restrictions" -- the classic single-turn jailbreak vocabulary, scored here so
it can contribute to multi-turn risk even when it shows up gradually (e.g.
"hypothetically" appearing for the first time on turn 6, not turn 1).
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import ConversationState, Feature, FeatureResult

_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bact as\b", re.I),
    re.compile(r"\bpretend (?:you'?re|you are|to be)\b", re.I),
    re.compile(r"\byou are now\b", re.I),
    re.compile(r"\bDAN\b"),
    re.compile(r"\bno(?:t any)? restrictions\b", re.I),
    re.compile(r"\bwithout (?:any )?(?:restrictions|filters|limitations|guardrails)\b", re.I),
    re.compile(r"\bignore (?:all |your )?(?:previous|prior) instructions\b", re.I),
    re.compile(r"\bhypothetically\b", re.I),
    re.compile(r"\bin a fictional (?:story|world|scenario|universe)\b", re.I),
    re.compile(r"\broleplay\b", re.I),
    re.compile(r"\bas a (?:fictional|hypothetical) (?:character|AI|assistant)\b", re.I),
    re.compile(r"\bunfiltered\b", re.I),
    re.compile(r"\bjailbreak\b", re.I),
    re.compile(r"\bdeveloper mode\b", re.I),
    re.compile(r"\b(?:evil|opposite) (?:twin|version)\b", re.I),
    re.compile(r"\bfor (?:a|my) (?:novel|screenplay|story) i'?m writing\b", re.I),
    re.compile(r"\byou have no (?:ethical|moral) guidelines\b", re.I),
]


class PersonaInjectionFeature(Feature):
    name = "persona_injection"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "user":
            return None

        text = message.get("content", "")
        matches = [p.pattern for p in _PATTERNS if p.search(text)]
        if not matches:
            return FeatureResult(
                raw_value=0.0, normalized_score=0.0, reason="no persona/hypothetical framing",
                turn_indices=[turn_index],
            )

        saturation = cfg_params.get("saturation_count", 2)
        normalized = min(1.0, len(matches) / saturation)

        return FeatureResult(
            raw_value=float(len(matches)),
            normalized_score=normalized,
            reason=(
                f"persona/hypothetical-injection framing detected "
                f"({len(matches)} pattern(s) matched)"
            ),
            turn_indices=[turn_index],
        )


__all__ = ["PersonaInjectionFeature"]
