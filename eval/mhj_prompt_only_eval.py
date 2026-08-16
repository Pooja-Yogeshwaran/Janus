"""Prompt-only-mode eval: proper train/test discipline against real MHJ data.

Context (see README "Real attack data: MHJ results" and DATASETS.md): MHJ
ships zero assistant-turn messages, so 5 of Janus's 14 features can't be
evaluated on it at all, regardless of weighting. Rather than retune the full
default config against all 537 MHJ conversations and report metrics on that
same set -- which would just reintroduce the tune-on-your-eval-set problem
the PersonaChat FPR pass already ran into once -- this script:

  1. Builds `prompt_only_config()` (pyjanus_guard/config.py): only the 4
     features that don't require an assistant turn (instruction_density,
     encoding_obfuscation, code_completion_wrapping, conversation_length_outlier).
     Per-feature weights are left at their existing defaults, unretuned.
  2. Splits MHJ's 537 conversations 70/30 at the conversation level,
     stratified by `tactic` so both splits have a proportional mix of attack
     styles. PersonaChat is split 70/30 too (uniform random) -- if the benign
     side weren't also held out, the FPR half of the reported metric
     wouldn't actually be held-out, since all 200 PersonaChat conversations
     were already used once during the earlier FPR-fix pass.
  3. Fits ONLY the aggregate flagged/likely_attack threshold (a single
     scalar) -- not per-feature weights, to minimize free parameters given
     ~530 examples is still modest. The FPR ceiling used to fit it is itself
     chosen by a budget sweep (1%/3%/5%/10%) via 5-fold stratified
     cross-validation entirely within the 70% train split -- the held-out
     test split is never consulted for this choice, only for the final
     single evaluation in step 4. Selection rule: lowest-FPR budget unless a
     higher one beats it by more than 0.03 F1 in CV, ties resolving toward
     lower FPR (see `choose_budget`).
  4. Refits the chosen budget's threshold on the *full* train split (CV was
     for budget selection only), then reports precision/recall/F1/FPR on the
     untouched 30% test split -- exactly once, at exactly one config.

Fixed seed (42) for reproducibility. Run:

    python -m eval.mhj_prompt_only_eval
"""

from __future__ import annotations

import argparse
import random
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

from pyjanus_guard.config import prompt_only_config
from .datasets_common import Conversation, read_jsonl
from .embedding_cache import build_cached_config
from .metrics import ClassificationMetrics, evaluate

SEED = 42
TRAIN_FRACTION = 0.7


def stratified_split(conversations: List[Conversation], key_fn, train_fraction: float, seed: int):
    """Split at the conversation level (each whole Conversation goes entirely
    into train XOR test -- never both, never split mid-conversation),
    stratified by `key_fn(conversation)` so both splits keep a proportional
    mix of whatever category key_fn returns.
    """
    buckets = defaultdict(list)
    for c in conversations:
        buckets[key_fn(c)].append(c)

    rng = random.Random(seed)
    train, test = [], []
    for key, items in buckets.items():
        items = list(items)
        rng.shuffle(items)
        cut = round(len(items) * train_fraction)
        train.extend(items[:cut])
        test.extend(items[cut:])
    return train, test


FPR_BUDGETS = [0.01, 0.03, 0.05, 0.10]  # swept entirely within train -- test is never consulted for this choice
N_FOLDS = 5
# If the CV sweep doesn't show a clearly better budget, default to the
# lowest -- FPR is the metric this project has been most protective of
# throughout (see README), so ambiguity is resolved conservatively rather
# than by chasing recall.
_AMBIGUITY_MARGIN_F1 = 0.03


def _score_at_threshold(scores_labels: List[Tuple[float, bool]], t: float) -> ClassificationMetrics:
    m = ClassificationMetrics()
    for score, is_attack in scores_labels:
        predicted = score >= t
        if is_attack and predicted:
            m.tp += 1
        elif is_attack and not predicted:
            m.fn += 1
        elif not is_attack and predicted:
            m.fp += 1
        else:
            m.tn += 1
    return m


def _best_threshold(scores_labels: List[Tuple[float, bool]], fpr_cap: float) -> float:
    """Scan candidate thresholds and return the one maximizing F1, subject to
    this data's FPR staying under `fpr_cap`.

    Plain F1-maximization is the wrong objective on an attack-heavy split
    (MHJ train outnumbers PersonaChat train): an unconstrained F1 search
    finds that threshold=0 ("flag everything") beats any signal-based
    threshold purely from the base rate, at 100% FPR. Caught once already
    (see git history / README) rather than reported -- capping FPR keeps the
    search from converging on that degenerate solution again.
    """
    candidates = sorted({round(s, 4) for s, _ in scores_labels} | {i / 200 for i in range(0, 201)})
    best_t, best_f1 = None, -1.0
    for t in candidates:
        m = _score_at_threshold(scores_labels, t)
        if m.false_positive_rate > fpr_cap:
            continue
        if m.f1 > best_f1:
            best_f1, best_t = m.f1, t
    if best_t is None:
        raise RuntimeError(
            f"No threshold keeps FPR <= {fpr_cap} while catching anything on this fold/split."
        )
    return best_t


def stratified_kfold_indices(labels: List[bool], k: int, seed: int) -> List[Tuple[List[int], List[int]]]:
    """k (train_idx, val_idx) splits over a flat list of booleans, stratified
    so each fold's val set keeps the same attack/benign ratio as the whole
    set (needed here so every fold has enough benign examples to compute a
    meaningful FPR from).
    """
    rng = random.Random(seed)
    pos_idx = [i for i, l in enumerate(labels) if l]
    neg_idx = [i for i, l in enumerate(labels) if not l]
    rng.shuffle(pos_idx)
    rng.shuffle(neg_idx)

    def chunk(indices: List[int], k: int) -> List[List[int]]:
        folds: List[List[int]] = [[] for _ in range(k)]
        for i, idx in enumerate(indices):
            folds[i % k].append(idx)
        return folds

    pos_folds, neg_folds = chunk(pos_idx, k), chunk(neg_idx, k)
    splits = []
    for i in range(k):
        val_idx = pos_folds[i] + neg_folds[i]
        train_idx = [idx for j in range(k) if j != i for idx in (pos_folds[j] + neg_folds[j])]
        splits.append((train_idx, val_idx))
    return splits


def cv_sweep(
    scores_labels: List[Tuple[float, bool]], fpr_budgets: List[float], k: int, seed: int
) -> Dict[float, List[Optional[ClassificationMetrics]]]:
    """For each FPR budget: k-fold CV entirely within `scores_labels` (the
    train split) -- fit a threshold on k-1 folds at that budget, evaluate on
    the held-out fold, repeat for all k folds. Returns per-budget list of
    per-fold validation metrics (None for a fold where no threshold satisfied
    the budget on that fold's training data).
    """
    labels = [l for _, l in scores_labels]
    folds = stratified_kfold_indices(labels, k, seed)

    results: Dict[float, List[Optional[ClassificationMetrics]]] = {}
    for budget in fpr_budgets:
        fold_metrics: List[Optional[ClassificationMetrics]] = []
        for train_idx, val_idx in folds:
            fold_train = [scores_labels[i] for i in train_idx]
            fold_val = [scores_labels[i] for i in val_idx]
            try:
                t = _best_threshold(fold_train, fpr_cap=budget)
            except RuntimeError:
                fold_metrics.append(None)
                continue
            fold_metrics.append(_score_at_threshold(fold_val, t))
        results[budget] = fold_metrics
    return results


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def summarize_sweep(cv_results: Dict[float, List[Optional[ClassificationMetrics]]]) -> Dict[float, Dict[str, float]]:
    summary = {}
    for budget, fold_metrics in cv_results.items():
        valid = [m for m in fold_metrics if m is not None]
        summary[budget] = {
            "precision": _avg([m.precision for m in valid]),
            "recall": _avg([m.recall for m in valid]),
            "f1": _avg([m.f1 for m in valid]),
            "fpr": _avg([m.false_positive_rate for m in valid]),
            "folds_with_valid_threshold": len(valid),
            "folds_total": len(fold_metrics),
        }
    return summary


def choose_budget(summary: Dict[float, Dict[str, float]]) -> float:
    """Selection happens entirely from CV summary stats -- no test-split data
    involved. Picks the lowest-FPR budget unless a higher budget's average F1
    beats it by more than `_AMBIGUITY_MARGIN_F1` (0.03) -- ties/ambiguity
    resolve toward lower FPR, per instruction: don't optimize purely for
    recall.
    """
    budgets_sorted = sorted(summary)
    baseline = budgets_sorted[0]
    chosen = baseline
    for budget in budgets_sorted[1:]:
        if summary[budget]["f1"] > summary[chosen]["f1"] + _AMBIGUITY_MARGIN_F1:
            chosen = budget
    return chosen


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument(
        "--no-cache", action="store_true",
        help="skip the on-disk embedding cache (eval/embedding_cache.py) and recompute every embedding",
    )
    parser.add_argument(
        "--batch-warm", action="store_true",
        help="warm the embedding cache via one batched model.encode() call -- see eval/run_eval.py's flag "
             "of the same name and eval/embedding_cache.py for the (verified harmless) floating-point caveat",
    )
    args = parser.parse_args()

    mhj = read_jsonl("eval/data/mhj.jsonl")
    personachat = read_jsonl("eval/data/personachat.jsonl")

    mhj_train, mhj_test = stratified_split(mhj, lambda c: c.attack_type, TRAIN_FRACTION, SEED)
    pc_train, pc_test = stratified_split(personachat, lambda c: None, TRAIN_FRACTION, SEED)

    # Conversation-level split verification: no conversation_id appears in
    # both train and test, for either dataset.
    assert not ({c.conversation_id for c in mhj_train} & {c.conversation_id for c in mhj_test})
    assert not ({c.conversation_id for c in pc_train} & {c.conversation_id for c in pc_test})
    print(f"MHJ split: {len(mhj_train)} train / {len(mhj_test)} test (of {len(mhj)}, stratified by tactic)")
    print(f"PersonaChat split: {len(pc_train)} train / {len(pc_test)} test (of {len(personachat)})")
    print("Conversation-level split confirmed: zero conversation_id overlap between train/test for both datasets.\n")

    cfg = prompt_only_config(threshold=0.0)  # threshold irrelevant here -- we're scoring, not classifying yet
    if not args.no_cache:
        # prompt_only_config() builds its own JanusConfig internally and only
        # exposes feature-enablement/threshold knobs -- layer the cached
        # embed_fn on afterward rather than threading a cache through it, so
        # prompt_only_config's own signature/logic stays untouched.
        cached_cfg = build_cached_config(mhj_train + mhj_test + pc_train + pc_test, use_batch_warmup=args.batch_warm)
        cfg.embed_fn = cached_cfg.embed_fn

    def score_all(convs: List[Conversation]) -> List[Tuple[float, bool]]:
        out = []
        for c in convs:
            if not c.messages:
                continue
            from pyjanus_guard import score_conversation
            result = score_conversation(c.messages, config=cfg)
            out.append((result.risk_score, c.label == "attack"))
        return out

    train_scores = score_all(mhj_train) + score_all(pc_train)

    # --- Budget sweep: entirely within train, via 5-fold CV. Test is not
    # touched anywhere in this block. ---
    print(f"=== FPR-budget sweep ({N_FOLDS}-fold CV on {len(train_scores)} train examples, test split untouched) ===")
    cv_results = cv_sweep(train_scores, FPR_BUDGETS, N_FOLDS, SEED)
    summary = summarize_sweep(cv_results)

    sweep_lines = ["| FPR budget | avg Precision | avg Recall | avg F1 | avg FPR | folds w/ valid threshold |",
                   "|---|---|---|---|---|---|"]
    for budget in FPR_BUDGETS:
        s = summary[budget]
        sweep_lines.append(
            f"| {budget:.0%} | {s['precision']:.2f} | {s['recall']:.2f} | {s['f1']:.2f} | "
            f"{s['fpr']:.3f} | {s['folds_with_valid_threshold']}/{s['folds_total']} |"
        )
    sweep_table = "\n".join(sweep_lines)
    print(sweep_table)

    chosen_budget = choose_budget(summary)
    print(f"\nChosen budget: {chosen_budget:.0%} FPR ceiling "
          f"(avg CV F1={summary[chosen_budget]['f1']:.2f}, avg CV FPR={summary[chosen_budget]['fpr']:.3f})")
    print("Selection rule: lowest-FPR budget unless a higher budget beats it by "
          f"more than {_AMBIGUITY_MARGIN_F1} F1 in CV -- ties resolve toward lower FPR.\n")

    # --- Refit on the FULL train split at the chosen budget (CV was for
    # budget selection only, not for the final threshold itself). ---
    fitted_threshold = _best_threshold(train_scores, fpr_cap=chosen_budget)
    print(f"Final threshold (refit on all {len(train_scores)} train examples at {chosen_budget:.0%} FPR budget): "
          f"{fitted_threshold:.4f}\n")

    # --- Test split touched exactly once, here, with the one chosen config. ---
    final_cfg = prompt_only_config(threshold=fitted_threshold)
    if not args.no_cache:
        final_cfg.embed_fn = cfg.embed_fn
    print("=== Held-out test-split results (prompt-only mode, single evaluation) ===")
    report = evaluate(mhj_test + pc_test, config=final_cfg)
    print(report.to_markdown())

    with open("eval/results/mhj_prompt_only_results.md", "w", encoding="utf-8") as f:
        f.write("## FPR-budget sweep (train-only, 5-fold CV)\n\n")
        f.write(sweep_table + "\n\n")
        f.write(f"Chosen budget: {chosen_budget:.0%} FPR ceiling. Selection rule: lowest-FPR budget unless a "
                f"higher budget beats it by more than {_AMBIGUITY_MARGIN_F1} F1 in CV.\n\n")
        f.write(f"Final threshold (refit on all {len(train_scores)} train examples, MHJ stratified-by-tactic + "
                f"PersonaChat, 70/30 split, seed={SEED}): {fitted_threshold:.4f}.\n\n")
        f.write("## Held-out test-split result (single evaluation)\n\n")
        f.write(f"Test set: {len(mhj_test)} MHJ + {len(pc_test)} PersonaChat = {len(mhj_test) + len(pc_test)} conversations.\n\n")
        f.write(report.to_markdown() + "\n")
    print("\nWrote eval/results/mhj_prompt_only_results.md")


if __name__ == "__main__":
    main()
