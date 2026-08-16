"""Step 4: leak-safe CV-fit + held-out evaluation of escalation_watchlist
(feature 15, see pyjanus_guard/features/escalation_watchlist.py), on its own,
with escalation_judge left at its default `None` throughout -- that hook
(Step 3) has no tested implementation to evaluate (no LLM API access was
available; see README "LLM-judge escalation check (opt-in, unvalidated)").

Fits ONLY the three escalation_watchlist-internal hyperparameters named for
this pass -- `mild_concern_floor`, `min_turns_for_trend` ("watchlist
window"), `trend_ratio` ("escalation-trend threshold") -- via k-fold CV on
the train split of the same MHJ+PersonaChat 70/30 split
`eval/mhj_prompt_only_eval.py` already uses (same seed, same stratification,
reused directly from that module so the split can't drift out of sync).
`reformulation_similarity_threshold` (the fourth Step-2 hyperparameter) is
deliberately NOT re-fit here -- left at its Step 2 default (0.55) -- because
it wasn't named in this pass's scope; noted explicitly in the report rather
than silently also tuned.

Since escalation_watchlist is a fixed heuristic (nothing is gradient-fit),
"k-fold CV" here means: score every train conversation once per candidate
hyperparameter combination (cheap -- scoring doesn't depend on which fold a
conversation lands in), then compute the feature's own standalone
precision/recall/F1/FPR (same per-feature methodology as
eval/metrics.py's per-feature table: did escalation_watchlist itself fire a
Flag, treated as a binary predictor) on each of 5 stratified folds and
average -- this measures whether a candidate combination's performance is
stable across different slices of train, not whether it was fit only to
happen to work on the exact whole train set. The test split is never
consulted for hyperparameter selection, only for the one final evaluation.

The evaluation config throughout is `prompt_only_config()` (the 4 features
that never require an assistant turn) with escalation_watchlist ALSO
enabled -- not the full 15-feature default config. This matches how the
Step 0 MHJ baseline was actually produced: MHJ ships zero assistant-turn
messages, so the 5 assistant-turn-dependent features (refusal_detection,
compliance_classification, reformulation_after_refusal, refusal_retry_count,
anchoring) are structurally inert on it regardless of weighting, and
including them would just reproduce the already-documented "0/537 flagged"
structural gap rather than isolating escalation_watchlist's own marginal
contribution. escalation_watchlist itself is NOT one of PROMPT_ONLY_FEATURES
(that set is deliberately conservative -- see config.py), but on MHJ-shaped
data its watchlist can only ever be seeded by the same 4 user-turn features
already in that set (refusal_detection/compliance_classification never fire
on MHJ either), so evaluating it stacked on top of prompt_only_config() is
the correct apples-to-apples comparison against the Step 0 baseline number,
not an inconsistency.

The aggregate `Thresholds` (flagged/likely_attack/watch) are left exactly as
prompt_only_config()'s existing fitted value (`_PROMPT_ONLY_FITTED_THRESHOLD`)
-- not re-fit here. This pass measures what enabling escalation_watchlist
*does* to that already-fixed decision boundary, not a new boundary chosen to
flatter it.

Run:

    python -m eval.watchlist_step4_eval
"""

from __future__ import annotations

import itertools
from typing import Dict, List, Tuple

from pyjanus_guard import score_conversation
from pyjanus_guard.config import _PROMPT_ONLY_FITTED_THRESHOLD, prompt_only_config

from .datasets_common import Conversation, read_jsonl
from .embedding_cache import build_cached_config
from .metrics import ClassificationMetrics, evaluate
from .mhj_prompt_only_eval import N_FOLDS, SEED, TRAIN_FRACTION, stratified_kfold_indices, stratified_split

FLOOR_CANDIDATES = [0.2, 0.3, 0.4]
MIN_TURNS_CANDIDATES = [2, 3, 4]
TREND_RATIO_CANDIDATES = [0.6, 0.7, 0.8]
FPR_CAP = 0.05  # same cap already justified in mhj_prompt_only_eval.py for this project's aggregate threshold


def _build_config(embed_fn, floor: float, min_turns: int, trend_ratio: float):
    cfg = prompt_only_config(threshold=_PROMPT_ONLY_FITTED_THRESHOLD)
    cfg.features["escalation_watchlist"].enabled = True
    cfg.features["escalation_watchlist"].params = {
        "mild_concern_floor": floor,
        "min_turns_for_trend": min_turns,
        "trend_ratio": trend_ratio,
    }
    cfg.embed_fn = embed_fn
    return cfg


def _standalone_fired(conversations: List[Conversation], config) -> List[bool]:
    fired = []
    for c in conversations:
        if not c.messages:
            fired.append(False)
            continue
        result = score_conversation(c.messages, config=config)
        fired.append(any(f.feature_name == "escalation_watchlist" for f in result.flags))
    return fired


def _fold_metrics(labels: List[bool], fired: List[bool], val_idx: List[int]) -> ClassificationMetrics:
    m = ClassificationMetrics()
    for i in val_idx:
        is_attack, predicted = labels[i], fired[i]
        if is_attack and predicted:
            m.tp += 1
        elif is_attack and not predicted:
            m.fn += 1
        elif not is_attack and predicted:
            m.fp += 1
        else:
            m.tn += 1
    return m


def _avg(values: List[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def main() -> None:
    mhj = read_jsonl("eval/data/mhj.jsonl")
    personachat = read_jsonl("eval/data/personachat.jsonl")

    mhj_train, mhj_test = stratified_split(mhj, lambda c: c.attack_type, TRAIN_FRACTION, SEED)
    pc_train, pc_test = stratified_split(personachat, lambda c: None, TRAIN_FRACTION, SEED)
    assert not ({c.conversation_id for c in mhj_train} & {c.conversation_id for c in mhj_test})
    assert not ({c.conversation_id for c in pc_train} & {c.conversation_id for c in pc_test})
    train = mhj_train + pc_train
    test = mhj_test + pc_test
    print(f"Train: {len(mhj_train)} MHJ + {len(pc_train)} PersonaChat = {len(train)}")
    print(f"Test (held out, touched once, below): {len(mhj_test)} MHJ + {len(pc_test)} PersonaChat = {len(test)}\n")

    cached_cfg = build_cached_config(train + test, use_batch_warmup=False)
    embed_fn = cached_cfg.embed_fn

    labels = [c.label == "attack" for c in train]
    folds = stratified_kfold_indices(labels, N_FOLDS, SEED)

    print(f"=== CV sweep: {len(FLOOR_CANDIDATES) * len(MIN_TURNS_CANDIDATES) * len(TREND_RATIO_CANDIDATES)} "
          f"hyperparameter combinations x {N_FOLDS}-fold CV on {len(train)} train examples ===")
    print("(reformulation_similarity_threshold held at its Step 2 default, 0.55 -- not in this pass's fit scope)\n")

    results = []
    for floor, min_turns, trend_ratio in itertools.product(FLOOR_CANDIDATES, MIN_TURNS_CANDIDATES, TREND_RATIO_CANDIDATES):
        cfg = _build_config(embed_fn, floor, min_turns, trend_ratio)
        fired = _standalone_fired(train, cfg)
        fold_metrics = [_fold_metrics(labels, fired, val_idx) for _, val_idx in folds]
        avg_precision = _avg([m.precision for m in fold_metrics])
        avg_recall = _avg([m.recall for m in fold_metrics])
        avg_f1 = _avg([m.f1 for m in fold_metrics])
        avg_fpr = _avg([m.false_positive_rate for m in fold_metrics])
        results.append((floor, min_turns, trend_ratio, avg_precision, avg_recall, avg_f1, avg_fpr))

    print("| floor | min_turns | trend_ratio | avg P | avg R | avg F1 | avg FPR |")
    print("|---|---|---|---|---|---|---|")
    for floor, min_turns, trend_ratio, p, r, f1, fpr in results:
        print(f"| {floor} | {min_turns} | {trend_ratio} | {p:.3f} | {r:.3f} | {f1:.3f} | {fpr:.3f} |")

    # Selection: highest avg CV F1 subject to avg CV FPR <= FPR_CAP -- same
    # priority this project has applied throughout (FPR capped first,
    # optimize F1 within that). Ties broken toward lower FPR, then lower
    # floor/min_turns/trend_ratio (arbitrary but deterministic) for
    # reproducibility.
    within_cap = [row for row in results if row[6] <= FPR_CAP]
    candidates = within_cap if within_cap else results
    if not within_cap:
        print(f"\nNo combination achieved avg CV FPR <= {FPR_CAP} -- selecting from all combinations by F1 instead.")
    chosen = max(candidates, key=lambda row: (row[5], -row[6]))
    floor, min_turns, trend_ratio = chosen[0], chosen[1], chosen[2]
    print(f"\nChosen: mild_concern_floor={floor}, min_turns_for_trend={min_turns}, trend_ratio={trend_ratio} "
          f"(avg CV F1={chosen[5]:.3f}, avg CV FPR={chosen[6]:.3f})\n")

    final_cfg = _build_config(embed_fn, floor, min_turns, trend_ratio)

    # --- (a) crescendo-3/4/9 recovery on the synthetic set -- dev signal only ---
    print("=== (a) crescendo-3/4/9 on the synthetic set (dev signal only, NOT proof) ===")
    synthetic = {c.conversation_id: c for c in read_jsonl("eval/data/synthetic_attacks.jsonl")}
    # Synthetic set exercises the full feature set (has assistant turns) --
    # use the plain default-equivalent config (all features on except the
    # documented off-by-default ones) with escalation_watchlist enabled at
    # the chosen hyperparameters, matching how Step 2's spot checks were run.
    from pyjanus_guard import JanusConfig

    synthetic_cfg = JanusConfig(embed_fn=embed_fn)
    synthetic_cfg.features["escalation_watchlist"].enabled = True
    synthetic_cfg.features["escalation_watchlist"].params = {
        "mild_concern_floor": floor,
        "min_turns_for_trend": min_turns,
        "trend_ratio": trend_ratio,
    }
    for cid in ["synthetic-crescendo-3", "synthetic-crescendo-4", "synthetic-crescendo-9"]:
        c = synthetic[cid]
        r = score_conversation(c.messages, config=synthetic_cfg)
        ew_fired = any(f.feature_name == "escalation_watchlist" for f in r.flags)
        print(f"  {cid}: risk_score={r.risk_score:.3f} flagged={r.flagged} escalation_watchlist_fired={ew_fired}")

    # --- (b) real held-out MHJ+PersonaChat test split, touched exactly once ---
    print("\n=== (b) Held-out MHJ+PersonaChat test split (N={}, touched exactly once) ===".format(len(test)))
    print("--- without escalation_watchlist (prompt_only_config() baseline, matches Step 0) ---")
    baseline_cfg = prompt_only_config(threshold=_PROMPT_ONLY_FITTED_THRESHOLD)
    baseline_cfg.embed_fn = embed_fn
    baseline_report = evaluate(test, config=baseline_cfg)
    print(baseline_report.to_markdown())

    print("\n--- with escalation_watchlist enabled (chosen hyperparameters) ---")
    final_report = evaluate(test, config=final_cfg)
    print(final_report.to_markdown())

    # --- (c) benign_escalating set, per category ---
    print("\n=== (c) benign_escalating set, per category (escalation_watchlist standalone firing) ===")
    benign_escalating = read_jsonl("eval/data/benign_escalating.jsonl")
    from collections import defaultdict

    by_cat: Dict[str, List[bool]] = defaultdict(list)
    for c in benign_escalating:
        cat = c.conversation_id[len("benign-escalating-"):].rsplit("-", 1)[0]
        # Use the plain enabled-escalation_watchlist config (same shape as
        # Step 2's per-category check), chosen hyperparameters, full feature
        # set active (benign_escalating conversations have assistant turns).
        cfg = JanusConfig(embed_fn=embed_fn)
        cfg.features["escalation_watchlist"].enabled = True
        cfg.features["escalation_watchlist"].params = {
            "mild_concern_floor": floor,
            "min_turns_for_trend": min_turns,
            "trend_ratio": trend_ratio,
        }
        r = score_conversation(c.messages, config=cfg)
        fired = any(f.feature_name == "escalation_watchlist" for f in r.flags)
        by_cat[cat].append(fired)

    total, total_fired = 0, 0
    print("| category | fired/n |")
    print("|---|---|")
    for cat in sorted(by_cat):
        entries = by_cat[cat]
        n, fired = len(entries), sum(entries)
        total += n
        total_fired += fired
        print(f"| {cat} | {fired}/{n} |")
    print(f"| **OVERALL** | **{total_fired}/{total}** |")

    with open("eval/results/watchlist_step4_results.md", "w", encoding="utf-8") as f:
        f.write("# Step 4: escalation_watchlist leak-safe evaluation\n\n")
        f.write(f"Chosen hyperparameters: mild_concern_floor={floor}, min_turns_for_trend={min_turns}, "
                f"trend_ratio={trend_ratio} (avg CV F1={chosen[5]:.3f}, avg CV FPR={chosen[6]:.3f})\n\n")
        f.write("## (b) Held-out test split, without escalation_watchlist\n\n")
        f.write(baseline_report.to_markdown() + "\n\n")
        f.write("## (b) Held-out test split, with escalation_watchlist\n\n")
        f.write(final_report.to_markdown() + "\n\n")
        f.write("## (c) benign_escalating per category\n\n")
        f.write("| category | fired/n |\n|---|---|\n")
        for cat in sorted(by_cat):
            entries = by_cat[cat]
            f.write(f"| {cat} | {sum(entries)}/{len(entries)} |\n")
        f.write(f"| **OVERALL** | **{total_fired}/{total}** |\n")
    print("\nWrote eval/results/watchlist_step4_results.md")


if __name__ == "__main__":
    main()
