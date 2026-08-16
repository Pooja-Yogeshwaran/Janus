"""Single CLI entry point for every dataset loader.

    python -m eval.fetch_datasets --dataset personachat --out eval/data/personachat.jsonl --limit 60
    python -m eval.fetch_datasets --dataset synthetic_attacks --out eval/data/synthetic_attacks.jsonl
    python -m eval.fetch_datasets --dataset mhj --out eval/data/mhj.jsonl --limit 100   # requires HF_TOKEN + gate acceptance
    python -m eval.fetch_datasets --dataset chatbot_arena --out eval/data/arena.jsonl --limit 60

See DATASETS.md for what each dataset is, its real license, and gating status.
"""

from __future__ import annotations

import argparse

from . import fetch_benign_escalating, fetch_chatbot_arena, fetch_mhj, fetch_personachat, fetch_synthetic_attacks
from .datasets_common import write_jsonl
from .fetch_unavailable import fetch_mt_jailbench, fetch_multibreak

_LOADERS = {
    "personachat": lambda limit, token: fetch_personachat.fetch(limit, token),
    "chatbot_arena": lambda limit, token: fetch_chatbot_arena.fetch(limit, token),
    "mhj": lambda limit, token: fetch_mhj.fetch(limit, token),
    "synthetic_attacks": lambda limit, token: fetch_synthetic_attacks.build(),
    "benign_escalating": lambda limit, token: fetch_benign_escalating.build(),
    "multibreak": lambda limit, token: fetch_multibreak(),
    "mt_jailbench": lambda limit, token: fetch_mt_jailbench(),
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", required=True, choices=sorted(_LOADERS))
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=60)
    parser.add_argument("--hf-token", default=None, help="overrides HF_TOKEN env var")
    args = parser.parse_args()

    conversations = _LOADERS[args.dataset](args.limit, args.hf_token)
    write_jsonl(conversations, args.out)
    print(f"Wrote {len(conversations)} conversations from '{args.dataset}' to {args.out}")


if __name__ == "__main__":
    main()
