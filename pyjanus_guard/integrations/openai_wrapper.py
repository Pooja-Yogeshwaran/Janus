"""OpenAI client wrapper example.

Requires the ``openai`` package (not a pyjanus-guard dependency -- bring your
own). Wraps ``client.chat.completions.create`` so every call is scored
against the running conversation automatically.

    from openai import OpenAI
    from pyjanus_guard.integrations.openai_wrapper import JanusGuardedOpenAI

    client = JanusGuardedOpenAI(OpenAI())
    response = client.chat.completions.create(model="gpt-4o-mini", messages=messages)
    if client.last_result.flagged:
        print(client.last_result.to_human_readable_trace())
"""

from __future__ import annotations

from typing import Any, List, Optional

from ..config import JanusConfig
from ..incremental import IncrementalScorer
from ..types import RiskResult


class JanusGuardedChatCompletions:
    def __init__(self, chat_completions: Any, guard: "JanusGuardedOpenAI") -> None:
        self._inner = chat_completions
        self._guard = guard

    def create(self, *args: Any, messages: Optional[List[dict]] = None, **kwargs: Any) -> Any:
        response = self._inner.create(*args, messages=messages, **kwargs)

        # `messages` is typically the *full*, growing conversation history on
        # every call (standard OpenAI chat pattern) -- only feed the scorer
        # turns it hasn't already seen, or every call would re-score earlier
        # turns and duplicate them in turn_scores/flags.
        already_seen = len(self._guard.scorer.messages)
        for message in (messages or [])[already_seen:]:
            self._guard.scorer.add_turn({"role": message.get("role", "user"), "content": message.get("content", "")})
        try:
            reply_text = response.choices[0].message.content or ""
        except (AttributeError, IndexError):
            reply_text = ""
        self._guard.last_result = self._guard.scorer.add_turn({"role": "assistant", "content": reply_text})

        return response


class JanusGuardedOpenAI:
    """Wraps an existing `openai.OpenAI()` client instance."""

    def __init__(self, client: Any, config: Optional[JanusConfig] = None) -> None:
        self._client = client
        self.scorer = IncrementalScorer(config=config)
        self.last_result: Optional[RiskResult] = None

        class _Chat:
            def __init__(self, guard: "JanusGuardedOpenAI") -> None:
                self.completions = JanusGuardedChatCompletions(client.chat.completions, guard)

        self.chat = _Chat(self)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._client, name)


__all__ = ["JanusGuardedOpenAI"]
