# Accuracy-only versus query-penalized GEPA

Completed 2026-10-07T13:54:36+08:00. Both runs evaluate the same 1,376 full validation questions.

| Measure | Query penalty 0.01 | Accuracy only (penalty 0) |
|---|---:|---:|
| Seed accuracy | 62.21% | 61.19% |
| Execution accuracy | 67.44% | 73.55% |
| Correct answers | 928 / 1,376 | 1012 / 1,376 |
| Solver queries | 1,529 | 3,378 |
| Queries per question | 1.11 | 2.45 |
| GEPA metric calls | 5,688 | 5,712 |

Both use a 5,500 metric-call budget, the same fixed 300-question selection subset, eight-question reflection batches, eight workers, seed 42, a ten-predictor-call ceiling, Codex Luna for SQL and Codex Astra for reflection, and the same initial solver. Token limits are 16,384 and 32,768. Candidate merging is disabled. Actual metric calls can exceed the budget when an evaluation batch finishes.

The accuracy-only objective assigns one to correct answers and zero to incorrect answers. Its feedback prioritizes accuracy instead of asking for fewer queries. Query measurements and correctness diagnostics remain available. Identical task requests reuse a copy of the previous run's response cache; reflection calls are uncached. The seed baseline differed despite matching settings and reused responses; this is evidence of evaluation variability.

Accuracy difference: +6.10 percentage points; +84 correct answers; +1849 solver queries.

Accuracy differences are computed from answer counts before rounding the displayed percentages.

These are single runs per objective, so the observed difference does not establish how consistently one objective wins across repeated optimization runs. The separate 500-question mini-dev test remains unused.
