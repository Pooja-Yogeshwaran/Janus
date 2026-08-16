"""Loader for awsaf49/persona-chat (MIT) -- the primary benign baseline.

The dataset ships in raw ConvAI2 text format: one row per line of the
original file. Lines starting a persona block ("N your persona: ...") mark
the start of a new two-person conversation; subsequent numbered lines each
contain one exchange as ``<A's turn>\\t<B's turn>\\t<distractor
candidates...>``, tab-separated, with the numbering continuing across the
whole conversation. This loader reconstructs each conversation's alternating
turns and discards the distractor-candidates field entirely (not real
dialogue, just negative-sampling noise for the original ranking task this
dataset was built for).

See DATASETS.md for the human-to-human-chitchat-not-assistant-directed
methodology caveat.
"""

from __future__ import annotations

import argparse
import re
from typing import List

from .datasets_common import Conversation, fetch_hf_rows, write_jsonl

_LINE_NUM_RE = re.compile(r"^(\d+)\s(.*)$")

DATASET_ID = "awsaf49/persona-chat"


def _parse_rows_into_conversations(lines: List[str]) -> List[Conversation]:
    conversations: List[Conversation] = []
    current_messages: List[dict] = []
    conv_idx = 0

    for raw in lines:
        m = _LINE_NUM_RE.match(raw)
        if not m:
            continue
        turn_num, rest = int(m.group(1)), m.group(2)

        if turn_num == 1:
            # new conversation block starting -- flush the previous one
            if len(current_messages) >= 2:
                conversations.append(
                    Conversation(
                        messages=current_messages,
                        label="benign",
                        source=DATASET_ID,
                        conversation_id=f"personachat-{conv_idx}",
                    )
                )
                conv_idx += 1
            current_messages = []

        if "your persona:" in rest or "partner's persona:" in rest:
            continue  # persona setup, not dialogue

        if "\t" not in rest:
            continue  # not a dialogue exchange line

        fields = rest.split("\t")
        turn_a, turn_b = fields[0].strip(), fields[1].strip()
        if turn_a:
            current_messages.append({"role": "user", "content": turn_a})
        if turn_b:
            current_messages.append({"role": "assistant", "content": turn_b})

    if len(current_messages) >= 2:
        conversations.append(
            Conversation(
                messages=current_messages,
                label="benign",
                source=DATASET_ID,
                conversation_id=f"personachat-{conv_idx}",
            )
        )
    return conversations


def fetch(limit: int, hf_token: str | None = None) -> List[Conversation]:
    """Fetch enough raw rows to build ~`limit` conversations.

    Each conversation spans ~10-20 raw text lines, so we over-fetch rows and
    stop once we have enough reconstructed conversations. Conversation blocks
    can straddle a page boundary, so all fetched lines are accumulated and
    re-parsed together rather than parsed page-by-page in isolation.
    """
    all_lines: List[str] = []
    conversations: List[Conversation] = []
    offset = 0
    page = 100
    while len(conversations) < limit and offset < 20000:
        data = fetch_hf_rows(DATASET_ID, offset=offset, length=page, hf_token=hf_token)
        rows = data.get("rows", [])
        if not rows:
            break
        all_lines.extend(r["row"]["text"] for r in rows)
        conversations = _parse_rows_into_conversations(all_lines)
        offset += page
    return conversations[:limit]


def main() -> None:
    parser = argparse.ArgumentParser(description="Fetch PersonaChat benign baseline conversations.")
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=60)
    args = parser.parse_args()

    conversations = fetch(args.limit)
    write_jsonl(conversations, args.out)
    print(f"Wrote {len(conversations)} conversations to {args.out}")


if __name__ == "__main__":
    main()
