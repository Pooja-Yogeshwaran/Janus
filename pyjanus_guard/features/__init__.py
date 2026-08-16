"""Feature registry.

Every implemented feature is instantiated once here and looked up by name
from :mod:`pyjanus_guard.incremental`. Names must match
:data:`pyjanus_guard.config.DEFAULT_FEATURE_NAMES` exactly -- that list is
also what config.py uses to build the default enabled/weight table, so the
two are the single source of truth for "what features exist."
"""

from __future__ import annotations

from typing import Dict

from .base import ConversationState, Feature, FeatureResult
from .refusal import RefusalDetectionFeature
from .reformulation import ReformulationAfterRefusalFeature
from .topic_drift import TopicDriftFeature
from .refusal_retry import RefusalRetryCountFeature
from .compliance import ComplianceClassificationFeature
from .persona_injection import PersonaInjectionFeature
from .instruction_density import InstructionDensityFeature
from .encoding_obfuscation import EncodingObfuscationFeature
from .code_completion import CodeCompletionWrappingFeature
from .step_size import StepSizeFeature
from .anchoring import AnchoringFeature
from .turn_velocity import TurnVelocityFeature
from .convergence import ConvergenceToTargetFeature
from .conversation_length import ConversationLengthOutlierFeature
from .escalation_watchlist import EscalationWatchlistFeature

FEATURE_REGISTRY: Dict[str, Feature] = {
    f.name: f
    for f in [
        RefusalDetectionFeature(),
        ComplianceClassificationFeature(),
        PersonaInjectionFeature(),
        InstructionDensityFeature(),
        EncodingObfuscationFeature(),
        CodeCompletionWrappingFeature(),
        TopicDriftFeature(),
        StepSizeFeature(),
        ReformulationAfterRefusalFeature(),
        AnchoringFeature(),
        RefusalRetryCountFeature(),
        TurnVelocityFeature(),
        ConvergenceToTargetFeature(),
        ConversationLengthOutlierFeature(),
        # Registered LAST deliberately: escalation_watchlist reads
        # state.turn_feature_scores for the six turn-level features above
        # (see escalation_watchlist.py), which must have already run for
        # this same turn_index in this same pass over FEATURE_REGISTRY.
        EscalationWatchlistFeature(),
    ]
}

__all__ = [
    "ConversationState",
    "Feature",
    "FeatureResult",
    "FEATURE_REGISTRY",
]
