"""Risk-over-time line chart: one attack transcript vs. one benign
transcript, plotting turn_scores turn-by-turn. This is the core visual for
the README/blog -- built directly from RiskResult.turn_scores, no separate
plotting-specific score computation.

Requires matplotlib (``pip install -e ".[eval]"``).
"""

from __future__ import annotations

from typing import Optional

from pyjanus_guard import RiskResult
from pyjanus_guard.types import Thresholds


def plot_risk_over_time(
    attack_result: RiskResult,
    benign_result: RiskResult,
    out_path: str,
    attack_label: str = "Attack transcript (synthetic FITD)",
    benign_label: str = "Benign transcript (PersonaChat)",
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    thresholds: Thresholds = attack_result.thresholds

    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)

    attack_x = list(range(1, len(attack_result.turn_scores) + 1))
    benign_x = list(range(1, len(benign_result.turn_scores) + 1))

    ax.plot(attack_x, attack_result.turn_scores, marker="o", color="#c0392b", linewidth=2, label=attack_label)
    ax.plot(benign_x, benign_result.turn_scores, marker="o", color="#2980b9", linewidth=2, label=benign_label)

    ax.axhline(thresholds.watch, color="#f39c12", linestyle="--", linewidth=1, label=f"watch threshold ({thresholds.watch:.2f})")
    ax.axhline(thresholds.likely_attack, color="#c0392b", linestyle=":", linewidth=1, label=f"likely_attack threshold ({thresholds.likely_attack:.2f})")

    ax.set_xlabel("Turn number (each message)")
    ax.set_ylabel("Risk score (0-1)")
    ax.set_title("Janus risk_score over the course of a conversation")
    ax.set_ylim(-0.02, 1.02)
    ax.set_xlim(0.5, max(max(attack_x, default=1), max(benign_x, default=1)) + 0.5)
    ax.legend(loc="upper left", fontsize=8, framealpha=0.9)
    ax.grid(True, alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


__all__ = ["plot_risk_over_time"]
