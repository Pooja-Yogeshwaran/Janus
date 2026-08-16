"""Core data types shared by every scoring path (batch and incremental).

The schema here is the contract: batch mode (:func:`pyjanus_guard.core.score_conversation`)
and incremental mode (:class:`pyjanus_guard.incremental.IncrementalScorer`) both
produce :class:`RiskResult` instances built the same way, so there is exactly one
output shape for callers to learn.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, List, Optional, TypedDict


class Role(str, Enum):
    USER = "user"
    ASSISTANT = "assistant"
    SYSTEM = "system"


class Message(TypedDict, total=False):
    role: str  # "user" | "assistant" | "system"
    content: str
    timestamp: float  # optional, unix seconds — enables turn-velocity feature


class Verdict(str, Enum):
    CLEAR = "clear"
    WATCH = "watch"
    LIKELY_ATTACK = "likely_attack"


@dataclass
class Flag:
    """A single feature firing on one or more turns.

    ``confidence`` is deliberately left as an optional field with no populated
    value in v1: there is no per-feature precision data until the eval harness
    (eval/run_eval.py) has been run against labeled transcripts. Adding a
    populated value later is additive, not a schema break.
    """

    feature_name: str
    turn_indices: List[int]
    raw_value: float
    human_readable_reason: str
    confidence: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        return {
            "feature_name": self.feature_name,
            "turn_indices": list(self.turn_indices),
            "raw_value": self.raw_value,
            "human_readable_reason": self.human_readable_reason,
            "confidence": self.confidence,
        }


@dataclass
class Thresholds:
    """Verdict/flag cut points. All configurable — v1 aggregation is a plain
    weighted sum (see aggregator.py) so these thresholds are the only "model"
    being fit, and they are fit by hand, not learned. Documented as a known
    v1 simplification; see README for the v2 (logistic regression) upgrade path.
    """

    # Set from this repo's own eval run (see eval/results/ and
    # config._DEFAULT_WEIGHTS): under the default weights, benign
    # (PersonaChat) conversations scored 0.08-0.17 with one 0.27 outlier,
    # while synthetic FITD/crescendo/CoA attacks scored 0.27-0.43 (one
    # single-turn-refusal control case scored 0.08 -- correctly low, since
    # a lone refusal with no follow-up isn't the escalation pattern Janus
    # targets). These cut points are hand-set from that separation, not
    # learned -- see aggregator.py for the v2 (logistic regression) upgrade
    # path once there's enough labeled data to fit thresholds properly.
    watch: float = 0.18
    likely_attack: float = 0.25
    # threshold for the `flagged` convenience bool; independent of verdict
    # bands so callers can wire `if result.flagged: block()` at a different
    # sensitivity than the watch/likely_attack labels.
    flagged: float = 0.25


@dataclass
class RiskResult:
    risk_score: float
    verdict: Verdict
    turn_scores: List[float]
    flags: List[Flag] = field(default_factory=list)
    thresholds: Thresholds = field(default_factory=Thresholds)

    @property
    def flagged(self) -> bool:
        return self.risk_score >= self.thresholds.flagged

    @property
    def categories(self) -> Dict[str, List[Flag]]:
        """Flags grouped by feature_name, OpenAI-moderation-response-shaped
        (a dict keyed by category name) for developers with that muscle memory.
        """
        grouped: Dict[str, List[Flag]] = {}
        for flag in self.flags:
            grouped.setdefault(flag.feature_name, []).append(flag)
        return grouped

    def to_dict(self) -> Dict[str, Any]:
        return {
            "risk_score": self.risk_score,
            "flagged": self.flagged,
            "verdict": self.verdict.value,
            "turn_scores": list(self.turn_scores),
            "flags": [f.to_dict() for f in self.flags],
            "categories": {
                name: [f.to_dict() for f in flags]
                for name, flags in self.categories.items()
            },
        }

    def to_json(self, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(), **kwargs)

    def to_human_readable_trace(self) -> str:
        """A step-by-step explanation, e.g.:

        Turn 2: refused. Turn 7: 89% similar to turn 2 -- flagged as
        reformulation-after-refusal.
        """
        if not self.flags:
            header = (
                f"verdict={self.verdict.value} risk_score={self.risk_score:.2f} "
                "-- no features fired."
            )
            return header

        by_turn: Dict[int, List[Flag]] = {}
        for flag in self.flags:
            anchor_turn = flag.turn_indices[-1] if flag.turn_indices else -1
            by_turn.setdefault(anchor_turn, []).append(flag)

        lines = [
            f"verdict={self.verdict.value} risk_score={self.risk_score:.2f} "
            f"flagged={self.flagged}"
        ]
        for turn_idx in sorted(by_turn):
            reasons = "; ".join(f.human_readable_reason for f in by_turn[turn_idx])
            lines.append(f"Turn {turn_idx}: {reasons}")
        return "\n".join(lines)


__all__ = ["Role", "Message", "Verdict", "Flag", "Thresholds", "RiskResult"]
