"""Shared refusal-phrase heuristics.

Factored out of refusal.py so that other code paths -- notably the
``prior_branches``/edit-history handling in incremental.py, which needs to
detect refusals in *discarded* branches the live thread no longer shows --
can reuse the exact same heuristic without importing the Feature class.
"""

from __future__ import annotations

import re
from typing import List

_REFUSAL_VERBS = (
    r"help(?:\s+with)?|assist|provide|do that|do this|comply|generate|write|"
    r"walk(?:\s+you)?\s+through|lay out|go into|share|give you|outline|"
    r"detail|discuss|describe|explain that|continue|proceed|create|produce"
)

_REFUSAL_PATTERNS: List[re.Pattern] = [
    # "I can't/won't/will not [filler words] <refusal verb>" -- filler allows
    # for "I can't in good conscience provide", "I can't really walk you
    # through", etc. without requiring the verb immediately after the modal.
    re.compile(
        r"\bi\s*(?:can(?:not|'t)|won'?t|will not)\s+(?:\w+\s+){0,3}?(?:" + _REFUSAL_VERBS + r")\b",
        re.I,
    ),
    re.compile(r"\bi'?m\s+(?:not able|unable)\s+to\b", re.I),
    re.compile(r"\bas an ai\b.{0,40}\b(?:cannot|can'?t|won'?t)\b", re.I),
    re.compile(r"\bagainst\s+(?:my|the|our)\s+(?:guidelines|policy|policies|values)\b", re.I),
    re.compile(r"\bi\s+(?:must|have to)\s+decline\b", re.I),
    re.compile(r"\bsorry,?\s+(?:but\s+)?i\s+(?:can'?t|cannot)\b", re.I),
    re.compile(r"\bi\s+don'?t\s+(?:feel comfortable|think i (?:can|should))\b", re.I),
    re.compile(r"\bi'?m\s+not\s+(?:going to|comfortable)\b", re.I),
    re.compile(r"\bthis\s+(?:request|content)\s+(?:violates|goes against)\b", re.I),
    re.compile(r"\bnot\s+something\s+i\s+can\s+(?:help with|do)\b", re.I),
    re.compile(r"\bi\s+won'?t\s+be\s+able\s+to\s+assist\b", re.I),
    re.compile(r"\bi\s+(?:can'?t|cannot|won'?t)\s+(?:lay\s+out|walk\s+you\s+through)\b", re.I),
    re.compile(r"\bi\s+still\s+can'?t\b", re.I),
    re.compile(r"\bi'?d\s+rather\s+not\b", re.I),
]


def refusal_match_count(text: str) -> int:
    return sum(1 for p in _REFUSAL_PATTERNS if p.search(text))


def heuristic_is_refusal(text: str) -> bool:
    return refusal_match_count(text) > 0


__all__ = ["refusal_match_count", "heuristic_is_refusal"]
