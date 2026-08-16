# Datasets used by the eval harness

**None of the datasets below are bundled in this repository.** The MIT
[LICENSE](LICENSE) at the repo root covers `pyjanus_guard`'s source code only.
`eval/fetch_datasets.py` downloads each dataset from its original source at
eval time; nothing third-party is committed to git. Licenses were checked
against each dataset's own hosting page/API on 2026-08-15 — re-verify before
relying on this table, licenses and gating status can change.

## Attack data (multi-turn jailbreak transcripts)

| Dataset | License | Gated? | Status in this repo |
|---|---|---|---|
| [MHJ (Scale AI)](https://huggingface.co/datasets/ScaleAI/mhj) | CC-BY-NC-4.0 | Yes — requires accepting HF's conditions on the dataset page and an `HF_TOKEN` with access | **Run successfully** (`eval/fetch_mhj.py`) — 537 conversations / 2,912 prompts, matching the dataset card exactly. Non-commercial license: fine for the (non-commercial) research use of computing eval metrics, but per-transcript content must never be redistributed or reproduced (see below). **Important, verified directly against the live data:** MHJ contains zero assistant-turn messages — every message is role `system` or `user` only. The dataset card's "we redacted some of the completions" undersells it: in this public release, all target-model responses are absent. See README "Real attack data: MHJ results" for what this means for Janus's eval numbers — several of the most heavily-weighted features cannot fire on this data at all, by construction, not by failure. HF's lightweight `datasets-server` rows-preview API (used by other loaders in this file via `fetch_hf_rows`) does not work for this dataset either — it misdetects the repo as an `imagefolder` dataset and returns 0 rows. The real data lives in a CSV (`harmbench_behaviors.csv`, misleadingly named) with up to 100 `message_N` JSON-encoded columns per row; `fetch_mhj.py` downloads and parses that file directly instead. |
| MultiBreak (arXiv:2605.01687, "A Scalable and Diverse Multi-turn Jailbreak Benchmark") | **Unconfirmed** | Unknown | Not integrated. As of 2026-08-15 no public dataset artifact (HF dataset, GitHub release, or download link) could be located for this benchmark, only the paper. Do not assume a license or availability that hasn't been verified — `eval/fetch_datasets.py` will raise `NotImplementedError` with a link to the paper if you select it. |
| MT-JailBench (arXiv:2605.11002, "A Modular Benchmark for Understanding Multi-Turn Jailbreak Attacks") | **Unconfirmed** | Unknown | Same as above — paper found, no public dataset artifact confirmed as of 2026-08-15. Not integrated. |
| [JailbreakBench (JBB-Behaviors)](https://huggingface.co/datasets/JailbreakBench/JBB-Behaviors) | MIT | No | Available as a supplementary single-turn-goal source (not itself multi-turn), used only to seed the *target behaviors* in the self-authored synthetic attack set below, not as transcript data. |

**MHJ access:** set `HF_TOKEN` (env var, or a `.env` file at the repo root —
see `eval/datasets_common.py`), accept the dataset conditions at the HF link
above, then run:

```bash
python -m eval.fetch_mhj --out eval/data/mhj.jsonl
```

This has been run successfully against the real dataset — see README "Real
attack data: MHJ results" for the numbers and an important caveat: MHJ
contains no assistant-turn messages at all (verified directly), which makes
several of Janus's most heavily-weighted features structurally unable to fire
on it. `eval/fetch_datasets.py --dataset mhj` also works as an alternative
entry point but defaults to `--limit 60`, not the full 537 — pass an explicit
`--limit` there, or use `eval.fetch_mhj` directly as above.

**Synthetic attack set (this repo):** `eval/fetch_synthetic_attacks.py` is a
small, self-authored set of FITD/crescendo/CoA-style transcripts written
specifically for this repo, *not copied or adapted from any dataset's actual
content*. It's kept and still used alongside the real MHJ numbers because it
exercises the full feature set (including the assistant-turn-dependent
features MHJ can't test) and gives a controlled, structural sanity check —
but it is not a substitute for MHJ, and MHJ is not a substitute for it
either; the two test different things. MultiBreak/MT-JailBench remain
unintegrated (no public dataset artifact as of 2026-08-15, paper only).

**Benign "escalating but harmless" set (this repo, dev/sanity-check only):**
`eval/fetch_benign_escalating.py` is a 30-conversation, fully self-authored
set built specifically to stress-test the trajectory-level watchlist feature
(escalation-trend / toned-down-reformulation detection): customer-service
complaints escalating to a supervisor, salary/rate/price negotiations, an
assistant declining to answer directly followed by a polite softened
re-ask, rising emotional urgency, stacked imperative technical follow-ups,
and creative/persona-voice writing requests — ordinary situations that
share surface shape with FITD/crescendo (tone or ask changes over several
turns) but carry no harmful intent anywhere. Like the synthetic attack set,
it's original content written for this repo, not adapted from any public
dataset, so no third-party license applies. **This is explicitly a
development/sanity-check set, not a held-out benchmark** — 30 self-authored
examples is enough to catch an obviously-broken feature, not to certify a
false-positive rate the way the 200-conversation PersonaChat set or the
held-out MHJ test split are meant to. Sanity-checked against the existing
(pre-watchlist) default config: 0/30 flagged, confirming this set doesn't
trivially trip already-shipped features before being used to test new ones.

## Benign baseline (false-positive rate)

| Dataset | License | Gated? | Role |
|---|---|---|---|
| [PersonaChat](https://huggingface.co/datasets/awsaf49/persona-chat) | MIT | No | **Primary** benign comparison set. |
| [lmsys/chatbot_arena_conversations](https://huggingface.co/datasets/lmsys/chatbot_arena_conversations) | Mixed — user-prompt fields CC-BY-4.0, **model-output fields CC-BY-NC-4.0** | No | **Optional secondary** benign set, closer to assistant-directed conversation. Only the `conversation_a`/`conversation_b` turns with `role == "user"` are ever fetched or used — `eval/fetch_chatbot_arena.py` drops every assistant/model-output field before writing to disk, specifically to stay inside the CC-BY-4.0-licensed slice. |
| ~~lmsys/lmsys-chat-1m~~ | Gated, non-commercial, revocable | Yes | **Explicitly not used anywhere in this repo.** Excluded per the license terms (gated + non-commercial + revocable is unsuitable for an MIT-licensed tool's public eval numbers). |

**Methodology caveat (PersonaChat):** PersonaChat is human-to-human
persona-based chit-chat collected for dialogue-generation research, not
assistant-directed conversation with a real LLM. It's a reasonable proxy for
"ordinary multi-turn conversation" false-positive testing (topic drift,
instruction density, etc. all still apply to human speech), but it under-represents
patterns specific to *talking to an assistant* (imperative requests directed at
an AI, code-completion asks, persona-injection framing that specifically
targets an AI's guardrails). Treat the reported FPR as a floor, not a ceiling
— it is likely to be somewhat higher against real assistant-directed traffic,
which is exactly what the `chatbot_arena_conversations` secondary set is for.

## Reproducing the eval run

```bash
pip install -e ".[eval]"
python -m eval.fetch_datasets --dataset personachat --out eval/data/personachat.jsonl --limit 200
python -m eval.fetch_datasets --dataset synthetic_attacks --out eval/data/synthetic_attacks.jsonl
python -m eval.run_eval --benign eval/data/personachat.jsonl --attack eval/data/synthetic_attacks.jsonl
```

The benign sample size was raised from an initial 40-conversation smoke test
to 200 partway through validation -- 40 conversations wasn't enough to trust a
false-positive-rate number, and in fact the larger sample is what surfaced
several features with FPR in the 55-100% range that the 40-sample run had
masked (fixed; see git history / CHANGELOG for specifics). 200 is still a
modest sample -- treat the FPR numbers as directional, not final, until a
larger run is done.

`eval/data/` and `eval/results/*.json` are gitignored — regenerate them
locally rather than expecting them in a fresh clone. The results table and
chart PNG referenced from the README *are* committed (as generated
documentation artifacts, not dataset content).

**Reproducing the real-embedding numbers** (see README "Embeddings" /
"Known limitations"): install the optional extra first, everything else is
identical --

```bash
pip install -e ".[eval,embeddings]"
python -m eval.run_eval --benign eval/data/personachat.jsonl --attack eval/data/synthetic_attacks.jsonl --results-dir eval/results_real_embedding_rerun
```

`JanusConfig()` auto-detects `sentence-transformers` and uses it without any
config changes -- see `pyjanus_guard/config.py` `_resolve_default_embed_fn`.
Pass `config=JanusConfig(embed_fn=pyjanus_guard.embeddings.default_embed)`
explicitly to `run_eval`-style scripts if you want to force the hash
embedding even with the extra installed (not currently wired into the
`run_eval` CLI's arguments -- edit the script or call `evaluate()` directly
for now).

## LLM-judge escalation check: not yet validated, here's the specific gap

`escalation_watchlist`'s optional `config.escalation_judge` hook (see README
"LLM-judge escalation check (opt-in, unvalidated)") is designed and wired
into the codebase -- the plumbing (trigger gating, override/abstain
semantics, off-by-default) is unit tested -- but no real judge
implementation has been run against any dataset in this repo, including the
ones listed above. This isn't a dataset-availability gap the way
MultiBreak/MT-JailBench are (no public artifact exists for those); every
dataset already fetched here (`eval/data/benign_escalating.jsonl`,
`eval/data/synthetic_attacks.jsonl`, the real MHJ + PersonaChat data) is
already available and already the right shape to validate a judge against.
The actual missing input is **LLM API access**: `escalation_judge` requires
a real provider call per invocation, and no API credential for any LLM
provider is configured in this project (only `HF_TOKEN`, for HuggingFace
dataset fetches -- unrelated). No credential means no real judge
implementation to run, so none was faked; see README for why testing
against a stub callable or reusing this development environment's own
ambient model access were both deliberately declined as substitutes.

**To run this validation:** set an API key for a chosen provider (env var or
`.env`, same pattern as `HF_TOKEN`), pick a model, implement one
`EscalationJudge` callable against it, then spot-check it against
`eval/data/benign_escalating.jsonl`'s `persistent_reask` and
`creative_persona` categories plus a few synthetic crescendo transcripts
before considering a full leak-safe (train/test-split) evaluation. That is
the next experiment this repo is set up for, not an open-ended gap.
