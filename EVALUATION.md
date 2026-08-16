# Evaluation detail

Full experimental trace behind the summarized findings in
[README.md's Known limitations](README.md#known-limitations) section, plus
the full FPR-budget selection methodology behind the
[prompt-only-mode held-out result](README.md#prompt-only-mode-held-out-results).
Nothing here is new — this is the same content that used to sit directly in
the README, moved out of the first-read path so a first-time visitor isn't
required to read all of it before understanding what Janus does and how
well it's actually validated. Read the README first; come here for the full
trace behind any specific claim.

## No public dataset with both sides

- **No public dataset with both sides of the conversation exists to fully
  validate this against, as of this writing.** MHJ (Scale AI) — the one real
  multi-turn jailbreak dataset actually accessible for this project — ships
  human red-teamers' prompts only; target-model completions were
  deliberately redacted before release (the dataset card cites *"potential
  export control concerns"*; see [Real attack data: MHJ
  results](README.md#real-attack-data-mhj-results) for the full finding, verified
  directly against all 3,449 messages in the file, and for what it means
  concretely: five of Janus's fourteen features, including the most heavily
  weighted ones, cannot be evaluated at all on this data, regardless of
  tuning). MultiBreak and MT-JailBench (see DATASETS.md) have no confirmed
  public dataset artifact at all. This is the single biggest gap in how
  validated this tool is — everything else below is a real but comparatively
  narrower issue.

## Embedding backend investigation

- **A real embedding model was tried specifically to fix `topic_drift`/`step_size`,
  and it didn't.** Both depend on `embed_fn`; the zero-dependency default
  (hashed bag-of-words) measured **100% FPR** against 200 real PersonaChat
  conversations, hypothesized to be sparse-hash-vector crudeness. Installing
  `pyjanus-guard[embeddings]` (real `sentence-transformers` embeddings) and
  re-measuring the exact same 200 conversations: **100% FPR again, unchanged.**
  Real embeddings correctly recognize that ordinary chit-chat genuinely hops
  between unrelated personal topics turn to turn — that's not an
  embedding-quality artifact, it's what casual multi-topic conversation looks
  like, and neither backend's distance-from-opening-turn /
  consecutive-turn-distance, at the current threshold, separates that from
  attack-style drift. Both stay OFF by default under **either** backend as a
  result — this is a feature-design/threshold problem, not an embedding
  problem, and a better embedding model doesn't fix it on its own. See
  [Embeddings](README.md#embeddings) above.

## reformulation_after_refusal paraphrase finding

- **`reformulation_after_refusal` improved measurably under a real embedding,
  but not enough to change its own flag-firing behavior.** Under the hash
  embedding, this feature's similarity score is driven almost entirely by
  exact repeated tokens (no stemming either — "trigger"/"triggers" don't
  match); a controlled genuine-paraphrase pair (same underlying ask, no
  shared nouns) scored 0.120. Under real `sentence-transformers` embeddings,
  the *identical* pair scored **0.434** — a real, substantial semantic-
  understanding improvement, verified directly, not inferred. But 0.434 is
  still under the 0.5 per-feature flag threshold (calibrated for the hash
  embedding's scale, not re-tuned for the real one), so this feature's own
  standalone recall on the synthetic attack set is **unchanged at 0.10**
  under either backend. Re-tuning `reformulation_after_refusal`'s
  `flag_threshold` specifically for the real embedding backend is an obvious,
  well-motivated next step that has **not** been done — deliberately, to keep
  this a single bounded change rather than another round of threshold search.

## Crescendo recall gap

- **Recall is skewed toward FITD/CoA-style attacks (hard refusal → reformulation),
  weaker on crescendo-style gradual escalation — partially, not fully,
  improved by the real embedding.** A spot-check of this repo's own
  synthetic attack set originally found every missed attack was
  `crescendo`-labeled (assistant turns that push back or hedge — "I'd rather
  not...", "that raises flags..." — without matching any refusal-heuristic
  pattern, so `refusal_detection` and everything anchored to it goes quiet).
  Under the real embedding, overall recall on this set rose from 0.50 to
  **0.70** (precision and FPR unchanged, 1.00/0.00) — but tracing exactly
  which conversations flipped matters here: 2 of the original 5 misses now
  cross the aggregate flag threshold (`fitd-2`, `crescendo-8`), and both were
  already sitting right at the boundary (0.23-0.25) before, both via the same
  features that were already firing (`instruction_density`,
  `refusal_retry_count`, `anchoring`) picking up slightly higher continuous
  scores from the better embedding — not via `reformulation_after_refusal`
  newly firing (see above, its own recall didn't move). **Two of the four
  original crescendo misses remain misses** (`crescendo-3`, `crescendo-4`,
  both still ~0.23), and the case that fired zero features at all
  (`crescendo-9`, pure topic drift with no hard refusal) is **completely
  unaffected** — 0.014 under both backends — because it needs `topic_drift`,
  which stays correctly disabled per the finding above. Report this plainly
  rather than citing the aggregate 0.50→0.70 number without this context: the
  real embedding helped marginal borderline cases, not the structural
  crescendo gap.

  ![crescendo-8 risk_score under both embedding backends, nearly overlapping until the final turn](eval/results/embedding_comparison.png)

  The two lines are close by design, not a rendering issue — a 0.028
  absolute difference (0.232 vs 0.260) is genuinely the size of this effect
  on this transcript, small enough to flip one borderline case and no more.
  Full before/after tables: `eval/results/synthetic_real_embedding_results.md`
  (real embedding) vs. `eval/results/benchmark_results.md` (hash embedding,
  same 10 synthetic attacks + 200 PersonaChat).

## FPR budget selection

Full methodology behind the FPR ceiling and threshold used in the
[prompt-only-mode held-out result](README.md#prompt-only-mode-held-out-results):

That scan needed a second pass to get right: maximizing F1 with no
constraint first converged on a threshold of 0.0 — flag *everything* — which
looks like a strong result (0.73 precision / 1.00 recall / 0.84 F1) purely
because the train split is 73% "attack" by construction (MHJ outnumbers
PersonaChat), nothing like real traffic; that "result" has 100% FPR and is
useless in practice. Caught before it shipped.

The FPR ceiling itself (not just the threshold) was chosen properly rather
than assumed: a budget sweep over 1%/3%/5%/10% ceilings, selected via 5-fold
stratified cross-validation **entirely within the 70% train split** — the
held-out test split is never consulted for this choice, only for the one
final evaluation below.

| FPR budget | avg CV Precision | avg CV Recall | avg CV F1 | avg CV FPR |
|---|---|---|---|---|
| 1% | 0.97 | 0.09 | 0.17 | 0.007 |
| 3% | 0.97 | 0.25 | 0.38 | 0.029 |
| **5% (chosen)** | 0.96 | 0.40 | **0.56** | 0.043 |
| 10% | 0.96 | 0.42 | 0.58 | 0.043 |

Selection rule, decided before looking at these numbers: pick the lowest-FPR
budget unless a higher one beats it by more than 0.03 F1 in CV — ties
resolve toward lower FPR, consistent with FPR being the metric this project
has been most protective of throughout, not something to trade away for a
small recall gain. 10%'s CV F1 (0.58) beat 5%'s (0.56) by only 0.02, under
that margin, so **5% was chosen** — the same ceiling used before this sweep
existed, now backed by cross-validated evidence instead of an assumed
number. Refitting on the *full* train split at that chosen budget (CV was
for budget selection only, not the final threshold) landed on the same
**threshold ≈ 0.0004** as before — in practice, close to "any of the four
features registers nonzero signal at all."

## Other narrower caveats

- **Editable message history** (see [prior_branches](README.md#editable-message-history-prior_branches)
  above) — a real blind spot without `prior_branches`, not fully closeable.
- **English-oriented heuristics.** The regex-based features (refusal
  detection, persona injection, instruction density, code-completion
  wrapping) are written against English phrasing. A non-English `refusal_judge`
  callable is the intended path to multilingual coverage — no work has gone
  into multilingual regex.
- **`conversation_length_outlier`'s baseline is integration-specific.** The
  shipped default (`baseline_length_mean≈14.7`, `std≈1.8`) is the actual
  measured mean/stdev across 200 real PersonaChat conversations, not a
  guess — the previous placeholder (`mean=6`, `std=4`) flagged 83% of those
  same conversations as length outliers. But "normal conversation length" is
  inherently product-specific — a customer-support bot and a long-form coding
  assistant have very different baselines. Treat this default as a rough
  starting point from one chit-chat dataset, not a universal constant, and
  set it from your own product's actual conversation-length distribution.
