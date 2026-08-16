"""``janus`` CLI: score a conversation transcript from a JSON file and print
a human-readable trace (or the full JSON result with --json).

    janus --transcript convo.json
    janus --transcript convo.json --json

``convo.json`` is either a bare list of ``{"role", "content"}`` messages, or
an object ``{"messages": [...]}``.
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any, Dict, List

from .core import score_conversation


def _load_messages(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict) and "messages" in data:
        return data["messages"]
    if isinstance(data, list):
        return data
    raise ValueError(
        f"{path} must contain a JSON list of messages or an object with a 'messages' key"
    )


def main(argv: List[str] = None) -> int:
    parser = argparse.ArgumentParser(prog="janus", description="Score a conversation transcript for multi-turn jailbreak patterns.")
    parser.add_argument("--transcript", required=True, help="path to a JSON transcript (list of {role, content} messages)")
    parser.add_argument("--json", action="store_true", help="print the full RiskResult as JSON instead of the human-readable trace")
    args = parser.parse_args(argv)

    try:
        messages = _load_messages(args.transcript)
    except (OSError, ValueError, json.JSONDecodeError) as e:
        print(f"error: {e}", file=sys.stderr)
        return 1

    result = score_conversation(messages)

    if args.json:
        print(result.to_json(indent=2))
    else:
        print(result.to_human_readable_trace())

    return 1 if result.flagged else 0


if __name__ == "__main__":
    raise SystemExit(main())
