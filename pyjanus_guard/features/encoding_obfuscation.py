"""Feature 5: encoding/obfuscation flags.

Composite of four cheap, stdlib-only sub-checks: base64-looking blocks,
unusual/zero-width unicode, leetspeak-style letter/digit substitution, and
Shannon entropy. Any one of these firing strongly is meaningful on its own
(e.g. a valid base64 blob), so the composite takes the max of the sub-scores
rather than averaging them down.
"""

from __future__ import annotations

import base64
import math
import re
import unicodedata
from typing import Any, Dict, List, Optional

from .base import ConversationState, Feature, FeatureResult

_BASE64_RE = re.compile(r"(?:[A-Za-z0-9+/]{4}){5,}(?:[A-Za-z0-9+/]{2}==|[A-Za-z0-9+/]{3}=)?")
# zero-width space, ZWNJ, ZWJ, BOM/ZWNBSP, word joiner
_ZERO_WIDTH = {chr(0x200B), chr(0x200C), chr(0x200D), chr(0xFEFF), chr(0x2060)}
_WORD_RE = re.compile(r"[A-Za-z0-9]+")


def _base64_score(text: str) -> float:
    for match in _BASE64_RE.finditer(text):
        candidate = match.group(0)
        if len(candidate) < 20:
            continue
        try:
            decoded = base64.b64decode(candidate, validate=True)
        except Exception:
            continue
        if not decoded:
            continue
        printable = sum(1 for b in decoded if 32 <= b < 127 or b in (9, 10, 13))
        if printable / len(decoded) > 0.6:
            return 0.9
    return 0.0


def _unicode_score(text: str) -> float:
    if not text:
        return 0.0
    zero_width_count = sum(1 for ch in text if ch in _ZERO_WIDTH)
    if zero_width_count > 0:
        return min(1.0, 0.6 + 0.1 * zero_width_count)
    non_ascii = sum(1 for ch in text if ord(ch) > 127)
    ratio = non_ascii / len(text)
    # allow ordinary use of accented/non-latin text; only flag when mixed with
    # ascii in a way that looks like homoglyph substitution (moderate ratio,
    # not "the whole message is in another script")
    if 0.05 <= ratio <= 0.5:
        return min(1.0, ratio * 1.5)
    return 0.0


def _leetspeak_score(text: str) -> float:
    words = _WORD_RE.findall(text)
    if not words:
        return 0.0
    mixed = [
        w for w in words
        if len(w) >= 4 and any(c.isdigit() for c in w) and any(c.isalpha() for c in w)
    ]
    if not mixed:
        return 0.0
    return min(1.0, len(mixed) / max(3, len(words)) * 3)


def _entropy_score(text: str) -> float:
    stripped = text.strip()
    if len(stripped) < 20:
        return 0.0
    freq: Dict[str, int] = {}
    for ch in stripped:
        freq[ch] = freq.get(ch, 0) + 1
    n = len(stripped)
    entropy = -sum((c / n) * math.log2(c / n) for c in freq.values())
    # typical English prose sits ~3.8-4.3 bits/char; cipher-like/random text runs higher
    if entropy <= 4.4:
        return 0.0
    return min(1.0, (entropy - 4.4) / 1.2)


class EncodingObfuscationFeature(Feature):
    name = "encoding_obfuscation"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "user":
            return None

        text = message.get("content", "")
        if not text.strip():
            return None

        scores = {
            "base64": _base64_score(text),
            "unusual_unicode": _unicode_score(text),
            "leetspeak": _leetspeak_score(text),
            "high_entropy": _entropy_score(text),
        }
        fired: List[str] = [k for k, v in scores.items() if v > 0.2]
        best = max(scores.values())

        return FeatureResult(
            raw_value=best,
            normalized_score=best,
            reason=(
                f"possible obfuscation: {', '.join(fired)}"
                if fired
                else "no obfuscation signals"
            ),
            turn_indices=[turn_index],
        )


__all__ = ["EncodingObfuscationFeature"]
