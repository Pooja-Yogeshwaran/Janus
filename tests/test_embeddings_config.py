"""Tests for embed_fn auto-detection in JanusConfig, without requiring
sentence-transformers to actually be installed -- the detection hook
(`sentence_transformers_available`) is monkeypatched so both branches are
exercised regardless of what's in the current environment.

Note what this auto-detection does and doesn't affect: only `embed_fn`
itself. topic_drift/step_size stay OFF regardless of which embedding backend
is active -- a real embedder was tried specifically to fix their 100% FPR
against real PersonaChat data and it didn't (see README "Known limitations"
/ config.py's `_EMBEDDING_DEPENDENT_FEATURES` comment), so there's no
backend-conditional feature-enabling left to test here anymore.
"""

from __future__ import annotations

import pyjanus_guard.config as config_module
from pyjanus_guard import JanusConfig
from pyjanus_guard.embeddings import default_embed


def test_default_config_uses_hash_embedding_when_extra_not_installed(monkeypatch):
    monkeypatch.setattr(config_module, "sentence_transformers_available", lambda: False)
    cfg = JanusConfig()
    assert cfg.embed_fn is default_embed


def test_default_config_uses_real_embedding_when_extra_installed(monkeypatch):
    from pyjanus_guard.embeddings import sentence_transformer_embed

    monkeypatch.setattr(config_module, "sentence_transformers_available", lambda: True)
    cfg = JanusConfig()
    assert cfg.embed_fn is sentence_transformer_embed
    # feature enablement is independent of embed_fn -- unaffected either way
    assert not cfg.is_enabled("topic_drift")
    assert not cfg.is_enabled("step_size")
    assert cfg.is_enabled("refusal_detection")
    assert not cfg.is_enabled("convergence_to_target")


def test_explicit_embed_fn_override_is_honored(monkeypatch):
    monkeypatch.setattr(config_module, "sentence_transformers_available", lambda: True)

    def custom_embed(text: str):
        return [0.0]

    cfg = JanusConfig(embed_fn=custom_embed)
    assert cfg.embed_fn is custom_embed


def test_explicit_features_override_is_honored():
    custom_features = {"refusal_detection": config_module.FeatureConfig(enabled=False)}
    cfg = JanusConfig(features=custom_features)
    assert cfg.features is custom_features
    assert not cfg.is_enabled("refusal_detection")
