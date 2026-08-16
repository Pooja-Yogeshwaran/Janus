"""The stateful, turn-by-turn scoring engine.

This is the one true scoring path. :func:`pyjanus_guard.core.score_conversation`
(batch mode) is a thin wrapper that constructs an :class:`IncrementalScorer`
and replays every message through :meth:`IncrementalScorer.add_turn` -- so
batch and incremental mode are, by construction, never able to diverge in
output shape or values.
"""

from __future__ import annotations

from typing import List, Optional

from .aggregator import score_and_verdict
from .config import JanusConfig
from .features import FEATURE_REGISTRY, ConversationState
from .features._refusal_patterns import heuristic_is_refusal
from .types import Flag, Message, RiskResult


class IncrementalScorer:
    """Turn-by-turn scorer for live chat apps.

    Example:
        scorer = IncrementalScorer()
        for msg in incoming_messages:
            result = scorer.add_turn(msg)
            if result.flagged:
                block()

    ``prior_branches``/edit_history is a known blind spot when *not* passed:
    if a UI lets a user edit an earlier turn, the live ``messages`` list Janus
    sees no longer contains the original ask or the refusal it triggered, and
    reformulation-after-refusal / refusal-retry-count cannot see across that
    edit. Passing the discarded branches in prepends them into a "shadow"
    history (see ``ConversationState.shadow_messages`` in features/base.py)
    that only those two features read, so they can count/compare against
    refusals hidden in edit history; every other feature still only looks at
    the live thread. See README "Known limitations."
    """

    def __init__(
        self,
        config: Optional[JanusConfig] = None,
        prior_branches: Optional[List[Message]] = None,
    ) -> None:
        self.config = config or JanusConfig()
        self.state = ConversationState(config=self.config)
        if prior_branches:
            self.state.prior_branches = list(prior_branches)
            self.state.shadow_messages = list(prior_branches)
            for i, m in enumerate(prior_branches):
                if m.get("role") == "assistant" and heuristic_is_refusal(m.get("content", "")):
                    self.state.shadow_refusals[i] = True
        self._turn_scores: List[float] = []
        self._flags: List[Flag] = []

    def add_turn(self, message: Message) -> RiskResult:
        turn_index = len(self.state.messages)
        self.state.messages.append(message)
        self.state.shadow_messages.append(message)
        shadow_index = len(self.state.shadow_messages) - 1

        for name, feature in FEATURE_REGISTRY.items():
            fcfg = self.config.feature(name)
            result = feature.score_turn(self.state, turn_index, fcfg.params)
            if name == "refusal_detection" and turn_index in self.state.refusals:
                # mirror into shadow-space immediately so reformulation /
                # retry-count see it later in this same pass over turn_index.
                self.state.shadow_refusals[shadow_index] = self.state.refusals[turn_index]
            if result is None:
                continue
            self.state.feature_scores[name] = result.normalized_score
            self.state.turn_feature_scores.setdefault(turn_index, {})[name] = result.normalized_score
            if fcfg.enabled and result.normalized_score >= fcfg.flag_threshold:
                self._flags.append(
                    Flag(
                        feature_name=name,
                        turn_indices=list(result.turn_indices),
                        raw_value=result.raw_value,
                        human_readable_reason=result.reason,
                    )
                )

        risk_score, verdict = score_and_verdict(self.state.feature_scores, self.config)
        self._turn_scores.append(risk_score)

        return RiskResult(
            risk_score=risk_score,
            verdict=verdict,
            turn_scores=list(self._turn_scores),
            flags=list(self._flags),
            thresholds=self.config.thresholds,
        )

    @property
    def messages(self) -> List[Message]:
        return list(self.state.messages)


__all__ = ["IncrementalScorer"]
