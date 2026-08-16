"""Feature 6: code-completion wrapping detection.

"Complete this function" / "finish this script" framing is a well-known way
to get a model to produce content it would refuse if asked for directly --
the harmful payload is smuggled in as a code-completion task instead of a
request. Scored higher when a completion-request phrase and an actual code
fence/definition both appear together.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import ConversationState, Feature, FeatureResult

_REQUEST_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bcomplete (?:the|this) (?:following )?(?:function|code|script)\b", re.I),
    re.compile(r"\bfinish (?:this|the) (?:code|function|script)\b", re.I),
    re.compile(r"\bfill in the (?:blanks?|missing (?:code|part))\b", re.I),
    re.compile(r"\bTODO:?\s*implement\b", re.I),
    re.compile(r"\bwrite a (?:python|bash|shell|javascript|js|c\+\+|java|powershell) script that\b", re.I),
    re.compile(r"\bcontinue (?:this|the) (?:code|function|program)\b", re.I),
]

_CODE_MARKERS = re.compile(r"```|\bdef \w+\(|\bfunction \w+\(|\bclass \w+[:\(]")


class CodeCompletionWrappingFeature(Feature):
    name = "code_completion_wrapping"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "user":
            return None

        text = message.get("content", "")
        has_request = any(p.search(text) for p in _REQUEST_PATTERNS)
        has_code = bool(_CODE_MARKERS.search(text))

        if has_request and has_code:
            score, reason = 0.8, "code-completion request wrapping actual code -- classic smuggling pattern"
        elif has_request:
            score, reason = 0.4, "code-completion-style request phrasing"
        elif has_code:
            score, reason = 0.15, "contains code, no completion-request phrasing"
        else:
            score, reason = 0.0, "no code-completion wrapping"

        return FeatureResult(
            raw_value=score, normalized_score=score, reason=reason, turn_indices=[turn_index],
        )


__all__ = ["CodeCompletionWrappingFeature"]
