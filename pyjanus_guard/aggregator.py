"""v1 aggregation: a plain, transparent weighted sum of normalized per-feature
scores. No training data dependency, fully inspectable -- every number in
`risk_score` can be traced back to which feature contributed what.

v2 upgrade path (not built here, by design -- see README "Aggregation"
section): once the eval harness in eval/ has produced labeled precision/recall
data per feature, those numbers are exactly what's needed to fit a logistic
regression over the same normalized feature vector instead of hand-set
weights. Swapping the aggregator function below for a fitted model is the
whole migration; RiskResult's shape does not change.
"""

from __future__ import annotations

from typing import Dict, Tuple

from .config import JanusConfig
from .types import Thresholds, Verdict


def aggregate_score(feature_scores: Dict[str, float], config: JanusConfig) -> float:
    total_weight = 0.0
    weighted_sum = 0.0
    for name, score in feature_scores.items():
        fcfg = config.feature(name)
        if not fcfg.enabled:
            continue
        weighted_sum += fcfg.weight * score
        total_weight += fcfg.weight
    if total_weight == 0:
        return 0.0
    return max(0.0, min(1.0, weighted_sum / total_weight))


def verdict_for(risk_score: float, thresholds: Thresholds) -> Verdict:
    if risk_score >= thresholds.likely_attack:
        return Verdict.LIKELY_ATTACK
    if risk_score >= thresholds.watch:
        return Verdict.WATCH
    return Verdict.CLEAR


def score_and_verdict(feature_scores: Dict[str, float], config: JanusConfig) -> Tuple[float, Verdict]:
    score = aggregate_score(feature_scores, config)
    return score, verdict_for(score, config.thresholds)


__all__ = ["aggregate_score", "verdict_for", "score_and_verdict"]
