"""Text embedding + vector helpers, with two backends behind one pluggable hook.

The zero-dependency default is a deterministic hashed bag-of-words vector
(the "hashing trick"): no model download, no API key, no network call, works
offline in pure standard library. It is intentionally crude -- good enough to
catch near-duplicate reformulations sharing literal vocabulary, measurably
not good enough for real semantic similarity (see README "Known
limitations" -- topic_drift/step_size measured 100% FPR under it, and
reformulation_after_refusal confirmed to miss genuine synonym-based
paraphrase).

If the optional `sentence-transformers` extra is installed
(`pip install pyjanus_guard[embeddings]`), :func:`sentence_transformer_embed`
is available and :class:`pyjanus_guard.config.JanusConfig` picks it up
automatically as the default `embed_fn` -- see config.py's
`_resolve_default_embed_fn`. Without that extra installed, everything here
behaves exactly as before: `default_embed` only, zero new dependencies.

Anywhere Janus needs a text vector, it calls a pluggable ``embed_fn`` from
:class:`pyjanus_guard.config.JanusConfig`. Swap in a different embeddings
endpoint (OpenAI/Anthropic/Cohere/etc.) with any callable matching
``Callable[[str], Sequence[float]]``.
"""

from __future__ import annotations

import hashlib
import importlib.util
import math
import re
import threading
from typing import List, Optional, Sequence

_TOKEN_RE = re.compile(r"[a-z0-9]+", re.IGNORECASE)

DEFAULT_DIM = 256

DEFAULT_SENTENCE_TRANSFORMER_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def _tokenize(text: str) -> List[str]:
    return [t.lower() for t in _TOKEN_RE.findall(text)]


def _bucket(token: str, dim: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "big") % dim


def default_embed(text: str, dim: int = DEFAULT_DIM) -> List[float]:
    """Deterministic hashed term-frequency vector, L2-normalized."""
    vec = [0.0] * dim
    tokens = _tokenize(text)
    if not tokens:
        return vec
    for tok in tokens:
        vec[_bucket(tok, dim)] += 1.0
    norm = math.sqrt(sum(x * x for x in vec))
    if norm > 0:
        vec = [x / norm for x in vec]
    return vec


def cosine_similarity(a: Sequence[float], b: Sequence[float]) -> float:
    if not a or not b or len(a) != len(b):
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    if norm_a == 0 or norm_b == 0:
        return 0.0
    # clamp for float drift
    return max(-1.0, min(1.0, dot / (norm_a * norm_b)))


def cosine_distance(a: Sequence[float], b: Sequence[float]) -> float:
    """0 == identical direction, 2 == opposite. Most callers want this in
    [0, 1] against non-negative TF vectors, where it behaves like 1 - similarity.
    """
    return 1.0 - cosine_similarity(a, b)


def sentence_transformers_available() -> bool:
    """Cheap check -- does not import torch/sentence_transformers, just asks
    the import system whether the package is installed.
    """
    return importlib.util.find_spec("sentence_transformers") is not None


_st_model_cache: dict = {}
_st_model_lock = threading.Lock()


def _get_sentence_transformer(model_name: str):
    if model_name not in _st_model_cache:
        with _st_model_lock:
            if model_name not in _st_model_cache:  # re-check inside the lock
                from sentence_transformers import SentenceTransformer  # heavy import, deferred to first use

                _st_model_cache[model_name] = SentenceTransformer(model_name)
    return _st_model_cache[model_name]


def make_sentence_transformer_embed(model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL):
    """Build an ``embed_fn`` backed by a real sentence-transformers model.
    The model is downloaded/loaded lazily on first call (not at import time)
    and cached per model name for the life of the process -- repeated calls,
    even across different `JanusConfig` instances using the same model name,
    reuse one loaded model rather than reloading it.

    Raises ``ImportError`` with a clear install hint if `sentence-transformers`
    isn't installed -- callers should check :func:`sentence_transformers_available`
    first if they want to avoid that.
    """
    if not sentence_transformers_available():
        raise ImportError(
            "sentence-transformers is not installed. Install the optional extra: "
            "pip install pyjanus_guard[embeddings]"
        )

    def _embed(text: str) -> List[float]:
        model = _get_sentence_transformer(model_name)
        vec = model.encode(text, normalize_embeddings=True)
        return vec.tolist()

    return _embed


# Pre-bound convenience callable using the default model -- what
# `JanusConfig`'s auto-detection (config.py `_resolve_default_embed_fn`)
# resolves to when `sentence-transformers` is installed. Referencing this
# function does NOT load the model or require sentence-transformers to be
# installed -- only calling it does (deferred to `_get_sentence_transformer`,
# which raises a clear ImportError at call time if it's missing).
def sentence_transformer_embed(text: str) -> List[float]:
    model = _get_sentence_transformer(DEFAULT_SENTENCE_TRANSFORMER_MODEL)
    vec = model.encode(text, normalize_embeddings=True)
    return vec.tolist()


def sentence_transformer_embed_batch(
    texts: Sequence[str], model_name: str = DEFAULT_SENTENCE_TRANSFORMER_MODEL
) -> List[List[float]]:
    """Batched counterpart to :func:`sentence_transformer_embed` -- one
    `model.encode()` call over the whole list instead of one call per text.

    Not wired into `JanusConfig()`'s default `embed_fn` (that interface is
    one-text-at-a-time by design -- see `ConversationState.embed`, which
    embeds lazily per turn as different features request it). This exists
    for eval-harness cache warm-up (`eval/embedding_cache.py`), which can
    collect every text in a dataset up front, where batching cuts wall-clock
    model-inference time substantially.

    **Not guaranteed bit-identical to calling `sentence_transformer_embed`
    once per text** -- verified directly: batched transformer inference pads
    sequences to the batch's max length, which measurably perturbs
    attention-masked floating-point output by up to ~1e-7 per vector
    component. That's far below this project's decision thresholds (set to
    2-3 decimal places) and was verified directly to not change any eval
    report's precision/recall/F1/FPR or any single conversation's `flagged`
    outcome -- but it is not the same bits, so callers that need strict
    byte-for-byte reproducibility against prior one-at-a-time runs should
    use `sentence_transformer_embed` per text instead. See
    `eval/embedding_cache.py`'s module docstring for the full writeup.
    """
    if not sentence_transformers_available():
        raise ImportError(
            "sentence-transformers is not installed. Install the optional extra: "
            "pip install pyjanus_guard[embeddings]"
        )
    model = _get_sentence_transformer(model_name)
    vecs = model.encode(list(texts), normalize_embeddings=True)
    return [v.tolist() for v in vecs]


__all__ = [
    "default_embed",
    "cosine_similarity",
    "cosine_distance",
    "DEFAULT_DIM",
    "sentence_transformers_available",
    "make_sentence_transformer_embed",
    "sentence_transformer_embed",
    "sentence_transformer_embed_batch",
    "DEFAULT_SENTENCE_TRANSFORMER_MODEL",
]
