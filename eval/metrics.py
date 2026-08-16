"""Precision/recall/F1/false-positive-rate computation, overall and broken
down per feature.

The per-feature breakdown answers "if this were the only signal, how good
would it be alone?" -- treating each feature's own firing (independent of the
aggregator/weights) as a standalone binary classifier against the
attack/benign label. This is exactly the data needed to eventually populate
`Flag.confidence` (see types.py) once there's enough labeled data to trust
per-feature numbers; v1 doesn't populate it yet because this eval run's
sample size (see README) is a smoke test, not enough to publish trustworthy
per-feature confidence values from.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from pyjanus_guard import JanusConfig, RiskResult, score_conversation
from pyjanus_guard.features import FEATURE_REGISTRY

from .datasets_common import Conversation


@dataclass
class ClassificationMetrics:
    tp: int = 0
    fp: int = 0
    tn: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        denom = self.tp + self.fp
        return self.tp / denom if denom else 0.0

    @property
    def recall(self) -> float:
        denom = self.tp + self.fn
        return self.tp / denom if denom else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r) if (p + r) else 0.0

    @property
    def false_positive_rate(self) -> float:
        denom = self.fp + self.tn
        return self.fp / denom if denom else 0.0

    @property
    def support(self) -> int:
        return self.tp + self.fp + self.tn + self.fn

    def as_row(self, name: str) -> Dict[str, str]:
        return {
            "name": name,
            "precision": f"{self.precision:.2f}",
            "recall": f"{self.recall:.2f}",
            "f1": f"{self.f1:.2f}",
            "fpr": f"{self.false_positive_rate:.2f}",
            "n": str(self.support),
        }


@dataclass
class EvalReport:
    overall: ClassificationMetrics
    per_feature: Dict[str, ClassificationMetrics]
    scored: List[Tuple[Conversation, RiskResult]] = field(default_factory=list)

    def to_markdown(self) -> str:
        lines = [
            "| Metric | Precision | Recall | F1 | FPR | N |",
            "|---|---|---|---|---|---|",
        ]
        row = self.overall.as_row("**Overall (risk_score >= flagged threshold)**")
        lines.append(
            f"| {row['name']} | {row['precision']} | {row['recall']} | "
            f"{row['f1']} | {row['fpr']} | {row['n']} |"
        )
        lines.append("")
        lines.append("### Per-feature (feature firing alone as a standalone classifier)")
        lines.append("")
        lines.append("| Feature | Precision | Recall | F1 | FPR | N |")
        lines.append("|---|---|---|---|---|---|")
        for name in sorted(self.per_feature):
            row = self.per_feature[name].as_row(name)
            lines.append(
                f"| {row['name']} | {row['precision']} | {row['recall']} | "
                f"{row['f1']} | {row['fpr']} | {row['n']} |"
            )
        return "\n".join(lines)


def evaluate(conversations: List[Conversation], config: Optional[JanusConfig] = None) -> EvalReport:
    config = config or JanusConfig()
    overall = ClassificationMetrics()
    per_feature: Dict[str, ClassificationMetrics] = {name: ClassificationMetrics() for name in FEATURE_REGISTRY}
    scored: List[Tuple[Conversation, RiskResult]] = []

    for conv in conversations:
        if not conv.messages:
            continue
        result = score_conversation(conv.messages, config=config)
        scored.append((conv, result))
        is_attack = conv.label == "attack"

        _accumulate(overall, is_attack, result.flagged)

        fired_features = {f.feature_name for f in result.flags}
        for name, metrics in per_feature.items():
            _accumulate(metrics, is_attack, name in fired_features)

    return EvalReport(overall=overall, per_feature=per_feature, scored=scored)


def _accumulate(metrics: ClassificationMetrics, actual_positive: bool, predicted_positive: bool) -> None:
    if actual_positive and predicted_positive:
        metrics.tp += 1
    elif actual_positive and not predicted_positive:
        metrics.fn += 1
    elif not actual_positive and predicted_positive:
        metrics.fp += 1
    else:
        metrics.tn += 1


__all__ = ["ClassificationMetrics", "EvalReport", "evaluate"]
