"""LangChain callback handler example.

Requires ``langchain-core`` (``pip install -e ".[langchain]"``) -- imported
lazily inside the class so the rest of pyjanus_guard never needs it installed.

    from pyjanus_guard.integrations.langchain_callback import JanusCallbackHandler

    handler = JanusCallbackHandler(on_flag=lambda result: print("FLAGGED:", result.to_human_readable_trace()))
    llm.invoke(messages, config={"callbacks": [handler]})
    print(handler.last_result.risk_score)
"""

from __future__ import annotations

from typing import Any, Callable, List, Optional

from ..config import JanusConfig
from ..incremental import IncrementalScorer
from ..types import RiskResult


class JanusCallbackHandler:
    """Scores each human/AI message turn-by-turn as a LangChain chain runs.

    Note: this is intentionally a plain class, not a subclass of
    ``langchain_core.callbacks.BaseCallbackHandler`` at import time -- the
    subclassing happens lazily in ``__init__`` so importing this module
    doesn't require langchain-core to be installed unless you actually
    instantiate the handler.
    """

    def __init__(
        self,
        config: Optional[JanusConfig] = None,
        on_flag: Optional[Callable[[RiskResult], None]] = None,
    ) -> None:
        try:
            from langchain_core.callbacks import BaseCallbackHandler  # noqa: F401
        except ImportError as e:
            raise ImportError(
                "JanusCallbackHandler requires langchain-core: pip install -e '.[langchain]'"
            ) from e

        self._scorer = IncrementalScorer(config=config)
        self._on_flag = on_flag
        self.last_result: Optional[RiskResult] = None

    def on_chat_model_start(self, serialized: Any, messages: List[List[Any]], **kwargs: Any) -> None:
        for message_list in messages:
            for message in message_list:
                role = "assistant" if getattr(message, "type", "") == "ai" else "user"
                self._score(role, getattr(message, "content", str(message)))

    def on_llm_end(self, response: Any, **kwargs: Any) -> None:
        try:
            text = response.generations[0][0].text
        except (AttributeError, IndexError):
            return
        self._score("assistant", text)

    def _score(self, role: str, content: str) -> None:
        self.last_result = self._scorer.add_turn({"role": role, "content": content})
        if self.last_result.flagged and self._on_flag:
            self._on_flag(self.last_result)


__all__ = ["JanusCallbackHandler"]
