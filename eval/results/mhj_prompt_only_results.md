## FPR-budget sweep (train-only, 5-fold CV)

| FPR budget | avg Precision | avg Recall | avg F1 | avg FPR | folds w/ valid threshold |
|---|---|---|---|---|---|
| 1% | 0.97 | 0.09 | 0.17 | 0.007 | 5/5 |
| 3% | 0.97 | 0.25 | 0.38 | 0.029 | 5/5 |
| 5% | 0.96 | 0.40 | 0.56 | 0.043 | 5/5 |
| 10% | 0.96 | 0.42 | 0.58 | 0.043 | 5/5 |

Chosen budget: 5% FPR ceiling. Selection rule: lowest-FPR budget unless a higher budget beats it by more than 0.03 F1 in CV.

Final threshold (refit on all 515 train examples, MHJ stratified-by-tactic + PersonaChat, 70/30 split, seed=42): 0.0004.

## Held-out test-split result (single evaluation)

Test set: 162 MHJ + 60 PersonaChat = 222 conversations.

| Metric | Precision | Recall | F1 | FPR | N |
|---|---|---|---|---|---|
| **Overall (risk_score >= flagged threshold)** | 1.00 | 0.43 | 0.60 | 0.00 | 222 |

### Per-feature (feature firing alone as a standalone classifier)

| Feature | Precision | Recall | F1 | FPR | N |
|---|---|---|---|---|---|
| anchoring | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| code_completion_wrapping | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| compliance_classification | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| convergence_to_target | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| conversation_length_outlier | 1.00 | 0.01 | 0.01 | 0.00 | 222 |
| encoding_obfuscation | 1.00 | 0.02 | 0.05 | 0.00 | 222 |
| escalation_watchlist | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| instruction_density | 0.97 | 0.44 | 0.61 | 0.03 | 222 |
| persona_injection | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| reformulation_after_refusal | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| refusal_detection | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| refusal_retry_count | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| step_size | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| topic_drift | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| turn_velocity | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
