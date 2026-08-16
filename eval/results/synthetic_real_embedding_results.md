| Metric | Precision | Recall | F1 | FPR | N |
|---|---|---|---|---|---|
| **Overall (risk_score >= flagged threshold)** | 1.00 | 0.70 | 0.82 | 0.00 | 210 |

### Per-feature (feature firing alone as a standalone classifier)

| Feature | Precision | Recall | F1 | FPR | N |
|---|---|---|---|---|---|
| anchoring | 0.25 | 0.10 | 0.14 | 0.01 | 210 |
| code_completion_wrapping | 1.00 | 0.10 | 0.18 | 0.00 | 210 |
| compliance_classification | 0.00 | 0.00 | 0.00 | 0.01 | 210 |
| convergence_to_target | 0.00 | 0.00 | 0.00 | 0.00 | 210 |
| conversation_length_outlier | 0.00 | 0.00 | 0.00 | 0.01 | 210 |
| encoding_obfuscation | 1.00 | 0.10 | 0.18 | 0.00 | 210 |
| instruction_density | 0.25 | 0.50 | 0.33 | 0.07 | 210 |
| persona_injection | 1.00 | 0.30 | 0.46 | 0.00 | 210 |
| reformulation_after_refusal | 1.00 | 0.10 | 0.18 | 0.00 | 210 |
| refusal_detection | 0.82 | 0.90 | 0.86 | 0.01 | 210 |
| refusal_retry_count | 1.00 | 0.40 | 0.57 | 0.00 | 210 |
| step_size | 0.00 | 0.00 | 0.00 | 0.00 | 210 |
| topic_drift | 0.00 | 0.00 | 0.00 | 0.00 | 210 |
| turn_velocity | 0.00 | 0.00 | 0.00 | 0.00 | 210 |
