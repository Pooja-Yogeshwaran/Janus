"""Placeholders for datasets requested but not integrated.

MultiBreak (arXiv:2605.01687) and MT-JailBench (arXiv:2605.11002) were
searched for at build time and only their papers could be found -- no public
HF dataset, GitHub release, or download link was located, so no license could
be verified either. Per DATASETS.md: don't assume a license or availability
that hasn't been confirmed. These raise clearly rather than silently
returning nothing, so a caller who selects them finds out why immediately
instead of getting an empty result set.
"""

from __future__ import annotations


class DatasetUnavailableError(NotImplementedError):
    pass


def fetch_multibreak(*args, **kwargs):
    raise DatasetUnavailableError(
        "MultiBreak has no confirmed public dataset artifact as of 2026-08-15 "
        "(paper: https://arxiv.org/abs/2605.01687). If a public release with a "
        "clear license appears, add a loader here following fetch_personachat.py "
        "as a template and update DATASETS.md."
    )


def fetch_mt_jailbench(*args, **kwargs):
    raise DatasetUnavailableError(
        "MT-JailBench has no confirmed public dataset artifact as of 2026-08-15 "
        "(paper: https://arxiv.org/abs/2605.11002). If a public release with a "
        "clear license appears, add a loader here following fetch_personachat.py "
        "as a template and update DATASETS.md."
    )


__all__ = ["DatasetUnavailableError", "fetch_multibreak", "fetch_mt_jailbench"]
