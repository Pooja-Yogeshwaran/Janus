"""Feature 4: instruction density.

Ratio of imperative/directive sentences to total sentences in a turn. Attack
prompts (especially CoA-style, which often chain several directives in one
message) tend to read as a list of commands; ordinary conversational turns
mostly don't.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from .base import ConversationState, Feature, FeatureResult

_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")

_IMPERATIVE_VERBS = {
    "write", "generate", "give", "tell", "explain", "describe", "list",
    "create", "provide", "show", "output", "continue", "pretend", "act",
    "ignore", "respond", "answer", "translate", "summarize", "complete",
    "repeat", "reveal", "include", "start", "begin", "make", "do", "detail",
    "outline", "draft", "produce", "compose", "convert", "format", "print",
    "return", "stop", "never", "always", "imagine",
}

# Anchored to the START of the sentence deliberately: true imperative mood
# in English drops the subject and puts the directive at/near the front
# ("Now give me...", "Don't skip any", "Always include..."). A bare
# `.search()` for these words anywhere in the sentence previously matched
# ordinary declarative statements too -- "I never eat vegetables" or "I
# always do that" are not commands, and that unanchored match was
# responsible for a 55% false-positive rate against real PersonaChat
# chit-chat (see eval/results/). Anchoring to sentence-start is still a
# heuristic, not a real parse, but it tracks actual imperative mood far
# better than an unanchored substring search.
_DIRECTIVE_START = re.compile(
    r"^(?:must|should|now|immediately|need to|have to|make sure|don'?t|never|always)\b",
    re.I,
)

# "do"/"does"/"did" immediately followed by a subject pronoun is do-support
# question inversion ("Do you like hiking?", "Did she call?") -- interrogative,
# not imperative. Sentence-initial "do" was 81 of 84 residual false positives
# against real PersonaChat data even after anchoring to sentence-start (see
# eval/results/); almost all of it was this construction, not "Do this now."
_QUESTION_AUX = re.compile(r"^(?:do|does|did)\s+(?:you|i|we|they|he|she|it|u)\b", re.I)


def _is_imperative(sentence: str) -> bool:
    stripped = sentence.strip()
    if not stripped:
        return False
    if _QUESTION_AUX.match(stripped):
        return False
    words = stripped.split()
    first = re.sub(r"[^a-zA-Z']", "", words[0]).lower()
    if first in _IMPERATIVE_VERBS:
        return True
    return bool(_DIRECTIVE_START.match(stripped))


class InstructionDensityFeature(Feature):
    name = "instruction_density"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        message = state.messages[turn_index]
        if message.get("role") != "user":
            return None

        text = message.get("content", "").strip()
        if not text:
            return None

        sentences: List[str] = [s for s in _SENTENCE_SPLIT.split(text) if s.strip()]
        if not sentences:
            return None

        imperative_count = sum(1 for s in sentences if _is_imperative(s))
        ratio = imperative_count / len(sentences)

        min_sentences = cfg_params.get("min_sentences", 2)
        if len(sentences) < min_sentences:
            # a single short imperative ("Write a poem") isn't itself suspicious
            ratio *= 0.5

        return FeatureResult(
            raw_value=ratio,
            normalized_score=min(1.0, ratio),
            reason=(
                f"{imperative_count}/{len(sentences)} sentences are imperative/"
                "directive -- high instruction density"
            ),
            turn_indices=[turn_index],
        )


__all__ = ["InstructionDensityFeature"]
