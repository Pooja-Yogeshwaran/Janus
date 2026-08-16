"""Orchestrates a full eval run: load labeled conversations, score them,
compute metrics, write a README-ready markdown table, and render the
risk-over-time chart PNG.

    python -m eval.run_eval --benign eval/data/personachat.jsonl \\
        --attack eval/data/synthetic_attacks.jsonl \\
        --results-dir eval/results
"""

from __future__ import annotations

import argparse
import os

from .datasets_common import read_jsonl
from .embedding_cache import build_cached_config
from .make_chart import plot_risk_over_time
from .metrics import evaluate


def _pick_chart_transcripts(report):
    attack_scored = [(c, r) for c, r in report.scored if c.label == "attack"]
    benign_scored = [(c, r) for c, r in report.scored if c.label == "benign"]

    if not attack_scored or not benign_scored:
        return None, None

    # a transcript with a few turns reads best on the chart -- prefer one
    # with turn_scores length in a legible range, else just take the highest-risk one.
    attack_scored.sort(key=lambda cr: (3 <= len(cr[1].turn_scores) <= 8, cr[1].risk_score), reverse=True)
    benign_scored.sort(key=lambda cr: (3 <= len(cr[1].turn_scores) <= 8,), reverse=True)
    return attack_scored[0], benign_scored[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--benign", required=True, nargs="+", help="one or more JSONL files of benign conversations")
    parser.add_argument("--attack", required=True, nargs="+", help="one or more JSONL files of attack conversations")
    parser.add_argument("--results-dir", default="eval/results")
    parser.add_argument(
        "--no-cache", action="store_true",
        help="skip the on-disk embedding cache (eval/embedding_cache.py) and recompute every embedding",
    )
    parser.add_argument(
        "--batch-warm", action="store_true",
        help="warm the embedding cache with one batched model.encode() call instead of per-text calls -- "
             "faster on a cold cache, NOT bit-identical to per-text embedding (~1e-7/component, verified to "
             "not change eval-report numbers in practice; see eval/embedding_cache.py docstring)",
    )
    args = parser.parse_args()

    conversations = []
    for path in args.benign:
        conversations.extend(read_jsonl(path))
    for path in args.attack:
        conversations.extend(read_jsonl(path))

    config = None
    if not args.no_cache:
        config = build_cached_config(conversations, use_batch_warmup=args.batch_warm)

    report = evaluate(conversations, config=config)
    table = report.to_markdown()

    os.makedirs(args.results_dir, exist_ok=True)
    table_path = os.path.join(args.results_dir, "benchmark_results.md")
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(table + "\n")
    print(table)
    print(f"\nWrote results table to {table_path}")

    attack_pick, benign_pick = _pick_chart_transcripts(report)
    if attack_pick and benign_pick:
        chart_path = os.path.join(args.results_dir, "risk_over_time.png")
        (_, attack_result) = attack_pick
        (_, benign_result) = benign_pick
        plot_risk_over_time(attack_result, benign_result, chart_path)
        print(f"Wrote chart to {chart_path}")
    else:
        print("Not enough scored attack/benign transcripts to render a chart.")


if __name__ == "__main__":
    main()
