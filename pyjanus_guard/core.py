"""Stateless batch entry point.

``score_conversation`` is intentionally a thin wrapper around
:class:`pyjanus_guard.incremental.IncrementalScorer` -- see that module's
docstring for why that's what guarantees batch and incremental mode share one
RiskResult shape.
"""

from __future__ import annotations

from typing import List, Optional

from .config import JanusConfig
from .incremental import IncrementalScorer
from .types import Message, RiskResult


def score_conversation(
    messages: List[Message],
    config: Optional[JanusConfig] = None,
    prior_branches: Optional[List[Message]] = None,
) -> RiskResult:
    """Score a full conversation transcript at once.

    Args:
        messages: ``[{"role": "user"|"assistant", "content": str}, ...]``
            (OpenAI/Anthropic message shape). Optional ``"timestamp"`` (unix
            seconds) per message enables the turn-velocity feature.
        config: feature enable/disable/reweight + pluggable callables. Uses
            documented defaults if omitted.
        prior_branches: discarded/edited earlier message branches, for callers
            whose chat UI supports editing an earlier turn. Without this,
            editing away a refused turn removes it from ``messages`` entirely
            and Janus has no way to see it -- a documented blind spot, not a
            bug. See README "Known limitations."

    Returns:
        A :class:`~pyjanus_guard.types.RiskResult` reflecting the full
        conversation, with ``turn_scores`` giving the running risk score after
        each message (computed incrementally, so it's safe to plot).
    """
    scorer = IncrementalScorer(config=config, prior_branches=prior_branches)
    cfg = scorer.config

    result: Optional[RiskResult] = None
    for message in messages:
        result = scorer.add_turn(message)

    if result is None:
        from .aggregator import verdict_for

        result = RiskResult(
            risk_score=0.0,
            verdict=verdict_for(0.0, cfg.thresholds),
            turn_scores=[],
            thresholds=cfg.thresholds,
        )
    return result


__all__ = ["score_conversation"]
