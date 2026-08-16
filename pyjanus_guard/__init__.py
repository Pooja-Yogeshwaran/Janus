"""pyjanus-guard: multi-turn LLM jailbreak pattern detection.

Named for Janus, the two-faced Roman god of thresholds and transitions --
looking back at prior refused turns in a conversation and forward at
escalation/drift, standing at the threshold of whether a conversation should
be flagged.

This is a defensive tool: it scores conversation transcripts for the
foot-in-the-door / crescendo / chain-of-attack signature (refused, then
reformulated or gradually escalated). It does not generate attacks.
"""

from .config import FeatureConfig, JanusConfig, PROMPT_ONLY_FEATURES, prompt_only_config
from .core import score_conversation
from .incremental import IncrementalScorer
from .types import Flag, Message, RiskResult, Thresholds, Verdict

__version__ = "0.1.0"

__all__ = [
    "score_conversation",
    "IncrementalScorer",
    "JanusConfig",
    "FeatureConfig",
    "prompt_only_config",
    "PROMPT_ONLY_FEATURES",
    "RiskResult",
    "Flag",
    "Thresholds",
    "Verdict",
    "Message",
    "__version__",
]
