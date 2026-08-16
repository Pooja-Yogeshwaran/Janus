"""On-disk embedding cache for eval iteration speed.

Eval runs re-embed the same conversation texts every time they're rerun --
scoring the same 200 PersonaChat / 537 MHJ / 30 benign_escalating
conversations repeatedly while iterating on a feature recomputes every
embedding from scratch each time, which is the actual slow part once a
real `sentence-transformers` backend is installed (model inference has real
per-call cost; the zero-dependency hash embedding is fast enough that
caching it barely matters, but it's cached too for consistency and because
duplicate texts across conversations still get deduped either way).

`EmbeddingCache` wraps any `embed_fn` with a sha256(text)-keyed on-disk
cache: a cache hit returns exactly the same vector that was computed and
stored on the miss that created it -- this module only ever *skips*
recomputing, it never approximates or derives a different value. Verified
directly: a full `eval.run_eval` run's output table is byte-identical with
and without this wrapper (diffed, zero output) -- see the Step 2 caching
notes / conversation history for the check.

**Not used by `pyjanus_guard`'s core scoring path or `JanusConfig()` by
default** -- this is eval-only development-iteration tooling, opted into
explicitly by eval scripts (`build_cached_config` below), so the shipped
library's zero-dependency, no-disk-I/O-by-default behavior is unaffected by
this file's existence.

One cache file per embedding backend (`cache_path_for_backend` picks a
distinct filename for the hash vs. real backend) -- mixing hash-embedding
and real-embedding vectors under one cache key space would silently serve
the wrong vectors to whichever backend is active, which is exactly the kind
of silent correctness bug this module exists to avoid, not introduce.

**Batched cache warm-up is a separate, explicit opt-in** (`warm_cache`'s
`batch_fn` parameter), not part of the default cache-wrap path, because it
is NOT bit-identical to one-text-at-a-time computation: verified directly,
`sentence_transformer_embed_batch` (pyjanus_guard/embeddings.py) run over a
list of texts differs from calling `sentence_transformer_embed` once per
text by up to ~1e-7 per vector component (batched transformer inference
pads sequences to the batch's max length, which measurably perturbs
attention-masked floating-point output). That's far below this project's
decision thresholds (2-3 decimal places) and was verified directly to not
change any eval-report number (precision/recall/F1/FPR, or any single
conversation's `flagged` boolean) on the synthetic+PersonaChat set -- but it
is not the same bits, so it defaults OFF rather than being silently folded
into the always-on caching path.
"""

from __future__ import annotations

import hashlib
import json
import os
from typing import Callable, Dict, Iterable, List, Optional, Sequence

CACHE_DIR = os.path.join("eval", ".cache")


class EmbeddingCache:
    def __init__(self, cache_path: str) -> None:
        self.cache_path = cache_path
        self._cache: Dict[str, List[float]] = {}
        self._dirty_count = 0
        self._flush_every = 200
        self._load()

    def _load(self) -> None:
        if os.path.isfile(self.cache_path):
            with open(self.cache_path, "r", encoding="utf-8") as f:
                self._cache = json.load(f)

    def flush(self) -> None:
        if self._dirty_count == 0:
            return
        os.makedirs(os.path.dirname(self.cache_path) or ".", exist_ok=True)
        tmp_path = self.cache_path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(self._cache, f)
        os.replace(tmp_path, self.cache_path)  # atomic on POSIX and Windows
        self._dirty_count = 0

    @staticmethod
    def _key(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def get(self, text: str) -> Optional[List[float]]:
        return self._cache.get(self._key(text))

    def put(self, text: str, vector: Sequence[float]) -> None:
        self._cache[self._key(text)] = list(vector)
        self._dirty_count += 1
        if self._dirty_count >= self._flush_every:
            self.flush()

    def get_or_compute(self, text: str, embed_fn: Callable[[str], Sequence[float]]) -> List[float]:
        cached = self.get(text)
        if cached is not None:
            return cached
        vec = list(embed_fn(text))
        self.put(text, vec)
        return vec

    def wrap(self, embed_fn: Callable[[str], Sequence[float]]) -> Callable[[str], List[float]]:
        def _wrapped(text: str) -> List[float]:
            return self.get_or_compute(text, embed_fn)

        return _wrapped

    def __len__(self) -> int:
        return len(self._cache)


def cache_path_for_backend(real_embeddings: bool) -> str:
    """Distinct cache file per embedding backend -- see module docstring for
    why this must never be shared across backends.
    """
    name = "sentence_transformers_all_MiniLM_L6_v2" if real_embeddings else "hash_bow_256"
    return os.path.join(CACHE_DIR, f"embeddings_{name}.json")


def warm_cache(
    cache: EmbeddingCache,
    texts: Iterable[str],
    embed_fn: Callable[[str], Sequence[float]],
    batch_fn: Optional[Callable[[Sequence[str]], Sequence[Sequence[float]]]] = None,
    batch_size: int = 64,
) -> int:
    """Populate `cache` for every text in `texts` not already cached. Returns
    the number of new entries computed (cache misses filled).

    `batch_fn`, if given, is used instead of `embed_fn` for the misses --
    called in chunks of `batch_size` -- to get the speed benefit of one
    model call over many texts instead of one call per text. Deliberately
    NOT the default: see module docstring for the verified ~1e-7 floating-
    point difference this introduces vs. one-at-a-time computation. Leave
    `batch_fn=None` for the strictly-byte-identical path (per-text
    `embed_fn`, just deduped and cached across runs).
    """
    unique_texts = list(dict.fromkeys(texts))  # de-dupe, preserve order
    missing = [t for t in unique_texts if cache.get(t) is None]
    if not missing:
        return 0

    if batch_fn is None:
        for text in missing:
            cache.put(text, embed_fn(text))
    else:
        for start in range(0, len(missing), batch_size):
            chunk = missing[start : start + batch_size]
            vectors = batch_fn(chunk)
            for text, vec in zip(chunk, vectors):
                cache.put(text, vec)

    cache.flush()
    return len(missing)


def build_cached_config(
    conversations,
    warm: bool = True,
    use_batch_warmup: bool = False,
):
    """Convenience for eval scripts: resolve the same embed_fn `JanusConfig()`
    would auto-detect (real sentence-transformers if installed, else the
    hash default), wrap it in a cache keyed for that backend, optionally
    pre-warm the cache from every message text across `conversations`
    (a plain per-text loop by default; pass `use_batch_warmup=True` for the
    faster-but-not-bit-identical batched path -- see module docstring), and
    return a `JanusConfig` using the cached embed_fn.

    `conversations` is any iterable of objects with a `.messages` list of
    `{"role", "content"}` dicts (i.e. `eval.datasets_common.Conversation`).
    """
    from pyjanus_guard import JanusConfig
    from pyjanus_guard.embeddings import sentence_transformers_available

    real_embeddings = sentence_transformers_available()
    base_cfg = JanusConfig()  # resolves embed_fn via the same auto-detection JanusConfig() always uses
    cache = EmbeddingCache(cache_path_for_backend(real_embeddings))

    if warm:
        texts = (m.get("content", "") for c in conversations for m in c.messages)
        batch_fn = None
        if use_batch_warmup and real_embeddings:
            from pyjanus_guard.embeddings import sentence_transformer_embed_batch

            batch_fn = sentence_transformer_embed_batch
        new_count = warm_cache(cache, texts, base_cfg.embed_fn, batch_fn=batch_fn)
        print(f"[embedding cache] {new_count} new embeddings computed, {len(cache)} total cached "
              f"({cache.cache_path})")

    cached_cfg = JanusConfig(embed_fn=cache.wrap(base_cfg.embed_fn))
    return cached_cfg


__all__ = [
    "EmbeddingCache",
    "cache_path_for_backend",
    "warm_cache",
    "build_cached_config",
    "CACHE_DIR",
]
