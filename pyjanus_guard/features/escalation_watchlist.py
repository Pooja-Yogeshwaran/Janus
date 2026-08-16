"""Feature 15 (additive; OFF by default -- see config.py): escalation watchlist.

**Motivation.** README "Known limitations" documents that Janus's recall is
skewed toward FITD/CoA-style attacks (hard refusal -> reformulation) and
weaker on crescendo-style gradual escalation, because five of the most
heavily-weighted features (refusal_detection, compliance_classification,
reformulation_after_refusal, refusal_retry_count, anchoring) all anchor
against a *hard refusal* -- and crescendo attacks that get only soft
pushback/hedging, or no pushback at all, leave those features quiet. This
feature is built specifically to give Janus a second, lower-bar way to
notice that pattern: instead of requiring one clean refusal to anchor
against, it tracks every turn that showed *any* mild concern on the
existing turn-level features, and looks for a trend or a softened repeat
across the *whole* accumulated list -- not just the single most recent
refused turn the way reformulation_after_refusal and refusal_retry_count do.

**Mechanism, two parts:**

1. "Mild concern" floor (`mild_concern_floor`, default 0.3 -- deliberately
   lower than the default per-feature `flag_threshold` of 0.5, and NOT the
   same thing as a refusal). Every turn is checked against the *fresh*
   (this-turn-only, not carried-forward) scores of the six turn-level
   features from README's "Turn-level" table: refusal_detection,
   compliance_classification, persona_injection, instruction_density,
   encoding_obfuscation, code_completion_wrapping. Deliberately excludes the
   trajectory/structural features (topic_drift, reformulation_after_refusal,
   anchoring, refusal_retry_count, turn_velocity, convergence_to_target,
   conversation_length_outlier), which already describe the conversation's
   shape rather than one turn's own content -- watchlisting based on those
   would be circular. Any turn clearing the floor on at least one of the six
   gets appended to `state.watchlist` as a `WatchlistEntry` (turn_index,
   score, contributing_features, text, embedding) -- even if no single
   feature crossed its own `flag_threshold` and no hard refusal ever fired.
   This is deliberately role-agnostic: a hedging/pushback assistant turn
   (which trips refusal_detection's own heuristic, e.g. "I'd rather not...")
   qualifies exactly the same way a stacked-imperative user turn does.

2. This feature's own score, checking the CURRENT turn against the FULL
   accumulated watchlist (all prior qualifying turns, not just the latest):
     a. `escalation_trend` -- are watchlist scores *strictly* rising over
        the conversation, not just holding at a noisy plateau. Deliberately
        stricter than topic_drift's `_is_monotonic_rising` (which counts
        "holds or rises" as trending, appropriate for that feature's
        distance-from-anchor semantics): a flat run of turns that
        repeatedly clear the floor at the *same* level (e.g. a normal
        multi-step debugging session where every "now do X" message scores
        ~0.5 on instruction_density) is common, benign turn-taking, not an
        escalation, and requiring net upward movement plus a majority of
        strictly-rising steps is what keeps that case from tripping this
        signal. See eval/data/benign_escalating.jsonl's
        `technical_instruction_dense` category, built specifically to stress
        this.
     b. `toned_down_reformulation` -- is the current turn embedding-similar
        to an earlier watchlist entry, while scoring LOWER on its own
        fresh turn-level surface score than that earlier entry did. This is
        the "refused/pushed-back, then came back softer but same underlying
        ask" pattern -- generalized from reformulation_after_refusal's
        narrower one-refusal anchor to compare against every watchlisted
        turn, so it can catch a softened repeat of *any* prior concerning
        turn, not only a formally-refused one.
   Combined score is the max of the two (either signal alone is meaningful;
   averaging them down would blunt whichever one actually fired).

3. Optional LLM-judge override on `toned_down_reformulation` only (never on
   `escalation_trend`, which is a pure score-history check with no
   turn-to-turn relationship for a judge to classify): `config.escalation_judge`
   (see config.py's `EscalationJudge`), `None` by default. When configured,
   invoked exactly once per user turn that has watchlist history to compare
   against (the same gating as 2b above) -- bounded, predictable call
   count, never once per watchlist entry. Same override/abstain contract as
   `refusal_judge`: True/False replaces the embedding heuristic's verdict
   outright, None abstains. **Deliberately unvalidated as of this writing**
   -- see README "LLM-judge escalation check (opt-in, unvalidated)" for why
   (no API credentials were available to test real judge behavior against,
   and faking that validation was deliberately declined). The plumbing
   itself (trigger gating, override semantics, off-by-default) is unit
   tested with a stub callable; that is not a claim about any real judge's
   classification quality.

**Purely additive.** Reads only its own dedicated state (`state.watchlist`)
and the also-new, also-unread-by-anyone-else `state.turn_feature_scores`
(see base.py) -- never reads or writes any field another feature depends on,
and no existing feature file was touched to build this. Registered in
FEATURE_REGISTRY (features/__init__.py) and DEFAULT_FEATURE_NAMES
(config.py) but shipped OFF by default, alongside convergence_to_target/
topic_drift/step_size, until it has FPR results to justify enabling it (see
eval/data/benign_escalating.jsonl and DATASETS.md -- dev/sanity-check set,
not a held-out benchmark).

**Hyperparameters below are placeholders, not yet fit.** `mild_concern_floor`,
`min_turns_for_trend`, `trend_ratio`, and `reformulation_similarity_threshold`
were chosen by inspection against the codebase's existing conventions (e.g.
topic_drift's `min_turns_for_trend`/`trend_ratio` defaults), not by any
train-split fitting procedure -- that fitting is a distinct, later step.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from ..embeddings import cosine_similarity
from .base import ConversationState, Feature, FeatureResult

# README's "Turn-level" table, minus nothing -- these are exactly the six
# features whose score describes a single turn's own content rather than the
# conversation's trajectory.
_TURN_LEVEL_FEATURES = (
    "refusal_detection",
    "compliance_classification",
    "persona_injection",
    "instruction_density",
    "encoding_obfuscation",
    "code_completion_wrapping",
)

_DEFAULT_MILD_CONCERN_FLOOR = 0.3
_DEFAULT_MIN_TURNS_FOR_TREND = 3
_DEFAULT_TREND_RATIO = 0.7
_DEFAULT_REFORMULATION_SIMILARITY_THRESHOLD = 0.55
_TREND_SLACK = 0.02


@dataclass
class WatchlistEntry:
    turn_index: int
    score: float
    contributing_features: List[str]
    text: str
    embedding: List[float] = field(default_factory=list)


def _rising_trend_fraction_and_net_increase(scores: List[float], slack: float) -> "tuple[float, float]":
    """Fraction of consecutive steps that *strictly* rise (b > a + slack), and
    the net change from first to last entry. Both must be positive for this
    to read as a trend rather than noise or a plateau -- see module
    docstring part 2a for why this is stricter than topic_drift's
    "holds-or-rises" version.
    """
    if len(scores) < 2:
        return 0.0, 0.0
    rising_steps = sum(1 for a, b in zip(scores, scores[1:]) if b > a + slack)
    fraction = rising_steps / (len(scores) - 1)
    net_increase = scores[-1] - scores[0]
    return fraction, net_increase


class EscalationWatchlistFeature(Feature):
    name = "escalation_watchlist"

    def score_turn(
        self, state: ConversationState, turn_index: int, cfg_params: Dict[str, Any]
    ) -> Optional[FeatureResult]:
        floor = cfg_params.get("mild_concern_floor", _DEFAULT_MILD_CONCERN_FLOOR)
        min_turns = cfg_params.get("min_turns_for_trend", _DEFAULT_MIN_TURNS_FOR_TREND)
        trend_ratio = cfg_params.get("trend_ratio", _DEFAULT_TREND_RATIO)
        reformulation_threshold = cfg_params.get(
            "reformulation_similarity_threshold", _DEFAULT_REFORMULATION_SIMILARITY_THRESHOLD
        )

        fresh_scores = state.turn_feature_scores.get(turn_index, {})
        current_turn_level = {
            name: fresh_scores[name] for name in _TURN_LEVEL_FEATURES if name in fresh_scores
        }
        qualifying = {name: score for name, score in current_turn_level.items() if score >= floor}

        # Compare this turn against the watchlist as it stood BEFORE this
        # turn -- "checks each new turn against the full watchlist," built
        # from strictly prior turns, not including whatever this turn itself
        # scores.
        prior_entries: List[WatchlistEntry] = list(state.watchlist)

        escalation_trend_score = 0.0
        trend_reason: Optional[str] = None
        if len(prior_entries) >= min_turns:
            prior_scores = [e.score for e in prior_entries]
            fraction, net_increase = _rising_trend_fraction_and_net_increase(prior_scores, _TREND_SLACK)
            if fraction >= trend_ratio and net_increase > _TREND_SLACK:
                escalation_trend_score = min(1.0, fraction)
                trend_reason = (
                    f"watchlist scores rising across {len(prior_entries)} prior flagged "
                    f"turns (rising_fraction={fraction:.2f}, net_increase={net_increase:.2f})"
                )

        toned_down_score = 0.0
        toned_down_reason: Optional[str] = None
        # User-turn only, same convention as reformulation_after_refusal --
        # this check specifically claims "the same requester came back with
        # a softer version of their own ask," which only makes sense for the
        # user side. Found the hard way (eval/data/benign_escalating.jsonl's
        # technical_instruction_dense category, 4/5 conversations
        # false-positiving): without this guard, an assistant's ordinary
        # compliant reply -- which structurally never trips
        # instruction_density/persona_injection the way a user request can
        # -- always scores near-zero on current_turn_level, so it looked
        # like a "toned-down reformulation" of whatever user request it was
        # just answering, every single time. That's not the pattern this
        # feature is trying to catch; compliance_classification already
        # covers "did the assistant cave," and role-gating here removes the
        # false signal without touching that.
        #
        # Also only attempted when the current turn has turn-level data to
        # compare with at all -- treating "no data this turn" as "scored
        # zero" would make every embedding-similar user turn look
        # toned-down by default, which isn't the claim being made either.
        if prior_entries and current_turn_level and state.messages[turn_index].get("role") == "user":
            current_surface_score = max(current_turn_level.values())
            current_embedding = state.embed(turn_index)
            best_match: Optional[WatchlistEntry] = None
            best_similarity = 0.0
            for entry in prior_entries:
                if not entry.embedding:
                    continue
                sim = cosine_similarity(entry.embedding, current_embedding)
                if sim > best_similarity:
                    best_similarity = sim
                    best_match = entry
            if (
                best_match is not None
                and best_similarity >= reformulation_threshold
                and current_surface_score < best_match.score
            ):
                toned_down_ratio = 1.0 - (current_surface_score / best_match.score)
                toned_down_score = max(0.0, min(1.0, best_similarity * toned_down_ratio))
                toned_down_reason = (
                    f"{best_similarity * 100:.0f}% similar to watchlisted turn "
                    f"{best_match.turn_index} (score {best_match.score:.2f}) but scoring "
                    f"lower now ({current_surface_score:.2f}) -- possible toned-down reformulation"
                )

            # Optional LLM-judge escalation check (see config.py's
            # EscalationJudge and README "LLM-judge escalation check
            # (opt-in, unvalidated)"). `None` by default -- never called
            # unless a caller supplies one. Gated to exactly this branch
            # (prior_entries and current_turn_level and a user turn) so it's
            # only ever invoked on turns that already have watchlist history
            # to compare against -- never on every turn, bounding call count
            # and cost. Same override/abstain contract as refusal_judge:
            # True/False directly replaces the embedding heuristic's verdict
            # above (not blended with it), None abstains and leaves the
            # heuristic's result untouched. The judge classifies a
            # relationship between two turns' text -- it is never asked to
            # generate, suggest, or rephrase anything.
            judge = state.config.escalation_judge
            if judge is not None:
                verdict = judge(
                    state.messages[turn_index].get("content", ""),
                    [entry.text for entry in prior_entries],
                )
                if verdict is True:
                    toned_down_score = 1.0
                    toned_down_reason = (
                        "llm-judge classified this turn as a softened/escalated "
                        "restatement of intent from an earlier watchlisted turn"
                    )
                elif verdict is False:
                    toned_down_score = 0.0
                    toned_down_reason = None
                # verdict is None -> abstain, heuristic result above stands.

        combined = max(escalation_trend_score, toned_down_score)

        # Extend the watchlist for FUTURE turns only after using its
        # pre-this-turn contents above.
        if qualifying:
            state.watchlist.append(
                WatchlistEntry(
                    turn_index=turn_index,
                    score=max(qualifying.values()),
                    contributing_features=sorted(qualifying),
                    text=state.messages[turn_index].get("content", ""),
                    embedding=list(state.embed(turn_index)),
                )
            )

        if combined <= 0.0:
            if qualifying:
                # Joined the watchlist, but no trend/reformulation signal
                # against history yet -- report at 0 rather than staying
                # silent, so the trace shows watchlist growth even before it
                # produces a nonzero score.
                return FeatureResult(
                    raw_value=0.0,
                    normalized_score=0.0,
                    reason=(
                        f"turn added to escalation watchlist ({', '.join(sorted(qualifying))}) "
                        "-- no trend/reformulation signal yet"
                    ),
                    turn_indices=[turn_index],
                )
            return None

        reasons = [r for r in (trend_reason, toned_down_reason) if r]
        return FeatureResult(
            raw_value=combined,
            normalized_score=combined,
            reason="; ".join(reasons),
            turn_indices=[turn_index],
        )


__all__ = ["EscalationWatchlistFeature", "WatchlistEntry"]
