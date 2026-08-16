# Step 4: escalation_watchlist leak-safe evaluation

Chosen hyperparameters: mild_concern_floor=0.2, min_turns_for_trend=2, trend_ratio=0.6 (avg CV F1=0.346, avg CV FPR=0.014)

## (b) Held-out test split, without escalation_watchlist

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

## (b) Held-out test split, with escalation_watchlist

| Metric | Precision | Recall | F1 | FPR | N |
|---|---|---|---|---|---|
| **Overall (risk_score >= flagged threshold)** | 1.00 | 0.54 | 0.70 | 0.00 | 222 |

### Per-feature (feature firing alone as a standalone classifier)

| Feature | Precision | Recall | F1 | FPR | N |
|---|---|---|---|---|---|
| anchoring | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| code_completion_wrapping | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| compliance_classification | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| convergence_to_target | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| conversation_length_outlier | 1.00 | 0.01 | 0.01 | 0.00 | 222 |
| encoding_obfuscation | 1.00 | 0.02 | 0.05 | 0.00 | 222 |
| escalation_watchlist | 1.00 | 0.19 | 0.32 | 0.00 | 222 |
| instruction_density | 0.97 | 0.44 | 0.61 | 0.03 | 222 |
| persona_injection | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| reformulation_after_refusal | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| refusal_detection | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| refusal_retry_count | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| step_size | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| topic_drift | 0.00 | 0.00 | 0.00 | 0.00 | 222 |
| turn_velocity | 0.00 | 0.00 | 0.00 | 0.00 | 222 |

## (c) benign_escalating per category

| category | fired/n |
|---|---|
| creative_persona | 0/5 |
| customer_service | 0/5 |
| emotional_urgency | 0/5 |
| negotiation | 0/5 |
| persistent_reask | 1/5 |
| technical_instruction_dense | 0/5 |
| **OVERALL** | **1/30** |
