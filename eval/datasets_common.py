"""Shared plumbing for dataset loaders: a small JSON-over-HTTP helper (stdlib
only -- no `requests` dependency needed for the lightweight HF
datasets-server "rows" preview API used by the loaders in this package) and
the on-disk JSONL shape all loaders write to.
"""

from __future__ import annotations

import json
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import asdict, dataclass, field
from typing import Any, Dict, List, Optional

HF_ROWS_ENDPOINT = "https://datasets-server.huggingface.co/rows"


def _load_dotenv() -> None:
    """Minimal, stdlib-only ``.env`` loader (repo root, `KEY=VALUE` per line,
    `#` comments, optional surrounding quotes). Not a new dependency -- this
    project stays dependency-light by design (see pyproject.toml) -- and only
    fills in variables not already set in the real environment, so an actual
    `export HF_TOKEN=...` always takes precedence over the file.
    """
    env_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    if not os.path.isfile(env_path):
        return
    with open(env_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key = key.strip()
            value = value.strip().strip('"').strip("'")
            os.environ.setdefault(key, value)


_load_dotenv()


@dataclass
class Conversation:
    """One labeled transcript for the eval harness."""

    messages: List[Dict[str, Any]]
    label: str  # "attack" | "benign"
    source: str  # dataset name, for per-source breakdowns
    attack_type: Optional[str] = None  # "fitd" | "crescendo" | "coa" | None
    conversation_id: str = ""

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


def fetch_hf_rows(
    dataset: str,
    config: str = "default",
    split: str = "train",
    offset: int = 0,
    length: int = 100,
    hf_token: Optional[str] = None,
) -> Dict[str, Any]:
    """Hit HF's lightweight dataset-preview "rows" API. Good enough for
    sampling a few dozen/hundred rows without pulling in the full `datasets`
    package or downloading an entire split. For bulk/production eval runs,
    prefer `datasets.load_dataset(...)` directly (see DATASETS.md).
    """
    url = (
        f"{HF_ROWS_ENDPOINT}?dataset={urllib.parse.quote(dataset, safe='')}"
        f"&config={config}&split={split}&offset={offset}&length={length}"
    )
    headers = {}
    token = hf_token or os.environ.get("HF_TOKEN")
    if token:
        headers["Authorization"] = f"Bearer {token}"

    req = urllib.request.Request(url, headers=headers)
    max_retries = 5
    backoff = 5.0
    for attempt in range(max_retries):
        try:
            with urllib.request.urlopen(req, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except urllib.error.HTTPError as e:
            if e.code in (401, 403):
                raise PermissionError(
                    f"{dataset} returned HTTP {e.code}. This dataset is likely gated -- "
                    "accept its conditions on the HF dataset page and set the HF_TOKEN "
                    "environment variable to a token with access."
                ) from e
            if e.code == 429 and attempt < max_retries - 1:
                time.sleep(backoff)
                backoff *= 2
                continue
            raise
    raise RuntimeError(f"Exhausted retries fetching {dataset}")


def write_jsonl(conversations: List[Conversation], out_path: str) -> None:
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for c in conversations:
            f.write(json.dumps(c.to_dict()) + "\n")


def read_jsonl(path: str) -> List[Conversation]:
    conversations = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            conversations.append(Conversation(**d))
    return conversations


__all__ = ["Conversation", "fetch_hf_rows", "write_jsonl", "read_jsonl", "HF_ROWS_ENDPOINT"]
