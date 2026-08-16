"""Loader for lmsys/chatbot_arena_conversations -- optional secondary benign
set, closer to assistant-directed conversation than PersonaChat.

Licensing is per-field, not per-dataset: the user-prompt turns are CC-BY-4.0,
but the model-output turns are CC-BY-NC-4.0. This loader only ever keeps
``role == "user"`` messages and drops every assistant/model-output field
before anything is written to disk -- so what lands in eval/data/ is entirely
within the CC-BY-4.0-licensed slice. Because assistant turns are dropped,
refusal/compliance-family features have nothing to score in this set; it
exists to stress-test the *user-turn* features (persona injection,
instruction density, encoding, topic drift, step size) against real
assistant-directed prompts rather than PersonaChat's human-to-human chat.
"""

from __future__ import annotations

import argparse
from typing import List, Optional

from .datasets_common import Conversation, fetch_hf_rows, write_jsonl

DATASET_ID = "lmsys/chatbot_arena_conversations"


def _extract_user_turns(conversation_field: List[dict]) -> List[dict]:
    return [
        {"role": "user", "content": turn.get("content", "")}
        for turn in conversation_field
        if turn.get("role") == "user" and turn.get("content")
    ]


def fetch(limit: int, hf_token: Optional[str] = None) -> List[Conversation]:
    conversations: List[Conversation] = []
    offset = 0
    page = 100
    while len(conversations) < limit and offset < 5000:
        data = fetch_hf_rows(DATASET_ID, offset=offset, length=page, hf_token=hf_token)
        rows = data.get("rows", [])
        if not rows:
            break
        for r in rows:
            row = r["row"]
            for side in ("conversation_a", "conversation_b"):
                turns = _extract_user_turns(row.get(side, []) or [])
                if len(turns) >= 1:
                    conversations.append(
                        Conversation(
                            messages=turns,
                            label="benign",
                            source=DATASET_ID,
                            conversation_id=f"arena-{offset}-{side}-{row.get('question_id', '')}",
                        )
                    )
        offset += page
    return conversations[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch lmsys/chatbot_arena_conversations user-prompt-only (CC-BY-4.0) benign set."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    conversations = fetch(args.limit)
    write_jsonl(conversations, args.out)
    print(f"Wrote {len(conversations)} user-turn-only conversations to {args.out}")


if __name__ == "__main__":
    main()
