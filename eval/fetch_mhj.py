"""Loader for ScaleAI/mhj (CC-BY-NC-4.0, gated).

Requires accepting the dataset's access conditions on its Hugging Face page
and an HF_TOKEN environment variable (or `.env` file, see datasets_common.py)
set to a token with access. CC-BY-NC-4.0 permits this non-commercial research
use (computing eval metrics); it does not permit redistributing the
transcripts themselves, so this loader writes conversations to your local
eval/data/ only -- never commit that directory's contents.

The real conversation data does NOT show up through HF's lightweight
datasets-server "rows" preview API (used by fetch_hf_rows for other loaders
in this package) -- that API auto-detects this repo as an `imagefolder`
dataset (it only recognizes `main_results.png`) and returns zero real rows.
The actual data lives in `harmbench_behaviors.csv`, a misleadingly-named file
containing one row per conversation with up to 100 `message_N` columns, each
a JSON-encoded `{"role": ..., "body": ...}` object. Verified directly against
the live file: 537 rows / 2,912 total messages, matching the dataset card's
"2,912 prompts across 537 multi-turn conversations" exactly.

IMPORTANT, verified directly against the live data: every message in this
file is role "system" or "user" -- there are zero "assistant"/model-response
messages anywhere in the dataset. MHJ ships only the human red-teamers'
prompt sequences; target-model completions were redacted from this public
release (the dataset card notes "we redacted some of the completions").
This means Janus features that require an assistant turn to fire at all --
refusal_detection, compliance_classification, reformulation_after_refusal,
refusal_retry_count, anchoring -- are structurally inert on this dataset,
not failing. See README/DATASETS.md for how eval results against MHJ are
caveated because of this.
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import os
import urllib.error
import urllib.request
from typing import List, Optional

from .datasets_common import Conversation, write_jsonl

DATASET_ID = "ScaleAI/mhj"
CSV_URL = f"https://huggingface.co/datasets/{DATASET_ID}/raw/main/harmbench_behaviors.csv"
_MAX_MESSAGE_COLUMNS = 101  # message_0 .. message_100, per the live schema

# Local cache of the raw CSV so repeated `fetch()` calls (e.g. rerunning eval
# iteration, or fetch_datasets.py --dataset mhj) don't re-download the same
# ~1MB gated file every time. This caches the raw bytes only -- the parsed
# Conversation objects are unaffected either way (same CSV content in, same
# parse logic, same output), so this cannot change fetch()'s results, only
# whether the network gets hit. CC-BY-NC-4.0 content, never committed --
# lives in the same gitignored eval/.cache/ tree as the embedding cache.
_CSV_CACHE_PATH = os.path.join("eval", ".cache", "mhj_harmbench_behaviors.csv")


def _download_csv(hf_token: Optional[str] = None) -> str:
    token = hf_token or os.environ.get("HF_TOKEN")
    headers = {"Authorization": f"Bearer {token}"} if token else {}
    req = urllib.request.Request(CSV_URL, headers=headers)
    try:
        with urllib.request.urlopen(req, timeout=60) as resp:
            return resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        if e.code in (401, 403):
            raise PermissionError(
                f"{DATASET_ID} returned HTTP {e.code}. This dataset is gated -- "
                "accept its conditions on the HF dataset page and set HF_TOKEN "
                "(env var or .env) to a token with access."
            ) from e
        raise


def _get_csv(hf_token: Optional[str] = None, use_cache: bool = True) -> str:
    if use_cache and os.path.isfile(_CSV_CACHE_PATH):
        with open(_CSV_CACHE_PATH, "r", encoding="utf-8") as f:
            return f.read()

    raw_csv = _download_csv(hf_token)

    if use_cache:
        os.makedirs(os.path.dirname(_CSV_CACHE_PATH), exist_ok=True)
        tmp_path = _CSV_CACHE_PATH + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            f.write(raw_csv)
        os.replace(tmp_path, _CSV_CACHE_PATH)

    return raw_csv


def fetch(limit: int, hf_token: Optional[str] = None, use_cache: bool = True) -> List[Conversation]:
    raw_csv = _get_csv(hf_token, use_cache=use_cache)
    reader = csv.DictReader(io.StringIO(raw_csv))

    conversations: List[Conversation] = []
    for idx, row in enumerate(reader):
        if len(conversations) >= limit:
            break

        messages = []
        for i in range(_MAX_MESSAGE_COLUMNS):
            cell = row.get(f"message_{i}", "")
            if not cell:
                continue
            try:
                obj = json.loads(cell)
            except json.JSONDecodeError:
                continue
            role = obj.get("role", "user")
            content = obj.get("body", "")
            if content:
                messages.append({"role": role, "content": content})

        if not messages:
            continue

        conversations.append(
            Conversation(
                messages=messages,
                label="attack",
                source=DATASET_ID,
                attack_type=row.get("tactic") or None,
                conversation_id=f"mhj-{row.get('question_id', '?')}-{idx}",
            )
        )

    return conversations


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fetch ScaleAI/mhj (gated, CC-BY-NC-4.0). Requires HF_TOKEN with dataset access."
    )
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=1000)
    parser.add_argument(
        "--no-cache", action="store_true",
        help="force a fresh download instead of reusing eval/.cache/mhj_harmbench_behaviors.csv",
    )
    args = parser.parse_args()

    conversations = fetch(args.limit, use_cache=not args.no_cache)
    write_jsonl(conversations, args.out)
    print(f"Wrote {len(conversations)} conversations to {args.out}")
    print("Reminder: CC-BY-NC-4.0 -- do not commit eval/data/ or redistribute transcripts.")
    print(
        "Reminder: this dataset has zero assistant-turn messages -- refusal-dependent "
        "Janus features are structurally inert on it. See module docstring / DATASETS.md."
    )


if __name__ == "__main__":
    main()
