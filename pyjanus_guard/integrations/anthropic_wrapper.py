"""Anthropic client wrapper example.

Requires the ``anthropic`` package (not a pyjanus-guard dependency -- bring
your own). Wraps ``client.messages.create`` so every call is scored against
the running conversation automatically.

    from anthropic import Anthropic
    from pyjanus_guard.integrations.anthropic_wrapper import JanusGuardedAnthropic

    client = JanusGuardedAnthropic(Anthropic())
    response = client.messages.create(model="claude-sonnet-4-5", max_tokens=1024, messages=messages)
    if client.last_result.flagged:
        print(client.last_result.to_human_readable_trace())
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..config import JanusConfig
from ..incremental import IncrementalScorer
from ..types import RiskResult


class JanusGuardedMessages:
    def __init__(self, messages_api: Any, guard: "JanusGuardedAnthropic") -> None:
        self._inner = messages_api
        self._guard = guard

    def create(self, *args: Any, messages: Optional[List[dict]] = None, **kwargs: Any) -> Any:
        response = self._inner.create(*args, messages=messages, **kwargs)

        already_seen = len(self._guard.scorer.messages)
        for message in (messages or [])[already_seen:]:
            content = message.get("content", "")
            if isinstance(content, list):  # Anthropic content-block format
                content = "".join(block.get("text", "") for block in content if isinstance(block, dict))
            self._guard.scorer.add_turn({"role": message.get("role", "user"), "content": content})

        try:
            reply_text = "".join(block.text for block in response.content if getattr(block, "type", "") == "text")
        except (AttributeError, TypeError):
            reply_text = ""
        self._guard.last_result = self._guard.scorer.add_turn({"role": "assistant", "content": reply_text})

        return response


class JanusGuardedAnthropic:
    """Wraps an existing `anthropic.Anthropic()` client instance."""

    def __init__(self, client: Any, config: Optional[JanusConfig] = None) -> None:
        self._client = client
        self.scorer = IncrementalScorer(config=config)
        self.last_result: Optional[RiskResult] = None
        self.messages = JanusGuardedMessages(client.messages, self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


__all__ = ["JanusGuardedAnthropic"]
