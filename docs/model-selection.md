# Model selection and error analysis

All results: **synthetic data**, seed `20261028`. The model and threshold were frozen on 2026-10-28.
Runtime: Python 3.13.0, scikit-learn 1.6.1. Python 3.11 runtime verification remains pending.

## Protocol

Train: 78,352 eligible claims (977 late reports excluded). Validation: 14,600 eligible
claims (955 late reports excluded). Test: 55,116 claims, including 1,784 observed positives.
The original incident-date splits and availability-safe histories are preserved.
Labels are assumed available at cutoff; label-maturity timestamps are absent.

Ten candidates were fixed before test evaluation. All preprocessing and fitting use train only.
Boosting uses learning rate 0.05, minimum leaf size 50, seed 20261028, and no early stopping.
Selection uses validation recall at <=5% review, then AP, then no demographics and logistic.
Thresholds for 2%, 5%, and 10% capacity were frozen before any holdout predictions.
No train-plus-validation refit or post-test tuning was performed.

## Validation comparison

| Model | Demographics | AP | Recall | Precision | Review |
|---|---|---:|---:|---:|---:|
| Logistic unweighted | No | 0.1349 | 27.52% | 16.07% | 4.77% |
| Logistic balanced | No | 0.1207 | 26.78% | 15.14% | 4.93% |
| Boosting leaves=7, rounds=100, L2=10 | No | 0.1496 | 29.73% | 17.07% | 4.86% |
| Boosting leaves=15, rounds=150, L2=10 | No | 0.1545 | 28.99% | 16.16% | 5.00% |
| Boosting leaves=15, rounds=150, L2=50 | No | 0.1467 | 27.76% | 15.54% | 4.98% |
| Logistic unweighted | Yes | 0.1388 | 28.01% | 15.77% | 4.95% |
| Logistic balanced | Yes | 0.1236 | 26.29% | 14.80% | 4.95% |
| Boosting leaves=7, rounds=100, L2=10 | Yes | 0.1500 | 29.73% | 16.60% | 4.99% |
| Boosting leaves=15, rounds=150, L2=10 | Yes | 0.1581 | 28.99% | 16.43% | 4.92% |
| Boosting leaves=15, rounds=150, L2=50 | Yes | 0.1475 | 29.48% | 16.64% | 4.94% |

Selected: demographic boosting, 7 leaves, 100 rounds, L2=10.
Threshold: `0.08607093703233727`, with score >= threshold flagged for manual review.
The corresponding non-demographic model has identical recall (121/407 positives);
demographics win only on AP (0.149958 vs 0.149648). This small difference is not evidence
of a meaningful demographic benefit or fairness. The larger model has higher AP but
lower recall at capacity and therefore loses under the predefined selection rule.

## Frozen holdout evaluation

| Validation capacity | Test review | Test precision | Test recall |
|---|---:|---:|---:|
| 2% | 2.61% | 22.39% | 18.05% |
| 5% | 7.43% | 14.40% | 33.07% |
| 10% | 13.30% | 11.13% | 45.74% |

At the primary threshold: AP 0.1402, ROC-AUC 0.7587, Brier score 0.02974, F1 0.2007.
Confusion matrix: TN 49,826; FP 3,506; FN 1,194; TP 590.
Test review is 7.43%, exceeding the validation ceiling by 2.43 percentage points.
This is not a guarantee of 5% operational capacity. Thresholds were not adjusted to test.
Observed positive prevalence increases from 2.79% on validation to 3.24% on test.

## Error analysis

| Incident type | Claims | Positives | False negatives | False positives | Recall | Precision |
|---|---:|---:|---:|---:|---:|---:|
| collision | 30,076 | 772 | 594 | 1258 | 23.06% | 12.40% |
| fire | 5,246 | 196 | 135 | 393 | 31.12% | 13.44% |
| flood | 5,665 | 138 | 104 | 271 | 24.64% | 11.15% |
| theft | 8,964 | 542 | 249 | 1389 | 54.06% | 17.42% |
| vandalism | 5,165 | 136 | 112 | 195 | 17.65% | 10.96% |

Collision contributes 594 of 1,194 missed positives (49.75%). Theft has the highest
recall but also 1,389 false positives and an 18.76% review rate. Vandalism recall is 17.65%.
These are descriptive error concentrations, not explanations of individual predictions.

Monthly review rises to 12.93% in October 2025; precision falls to 9.64% that month
and 8.60% in December 2025. A static threshold does not preserve a stable queue.
Age-band recalls range from 31.20% to 39.66%; the oldest group has only 58 positives.
Province, age, coverage, incident type and monthly counts are retained in the JSON report.
Slices are exploratory, have no uncertainty intervals, and do not establish fairness.

## Feature shifts and limitations

Using train-fitted feature transforms, the largest absolute numeric mean shifts from
validation to test (in validation standard deviations) are workshop_claim_count_90d
(+0.467), report_delay_days (+0.226), and prior_claims_total (+0.186). The largest
categorical total-variation distance is sales_channel (0.038). These descriptive
shifts do not establish causation; longer histories can shift counts naturally.
Per-feature missing fractions are also recorded. No significance testing is claimed.

Scores are uncalibrated. Observed labels are not latent fraud truth. This synthetic
benchmark does not establish real-world effectiveness. Manual review, capacity monitoring,
label-maturity validation, and demographic policy review remain necessary before deployment.
Any future model changes require fresh validation/holdout design; do not optimize on this test.

## Verification and artifacts

Eleven behavioral tests pass, including train/validation isolation, threshold reuse,
data-change and repeated-evaluation guards, unseen demographics, missing values,
and serialization parity. Full-data tuning and one holdout evaluation completed.
The saved full-data model reproduces validation metrics exactly and passes reload parity.
The original run checked Python 3.11 syntax; subsequent CI verifies that runtime
and runs Ruff checks. Windows joblib emitted a physical-core detection warning and used
logical-core detection; execution completed with numerical threads limited to two.

Local excluded artifacts: `models/phase-3/model.joblib`, `selection.json`, `test-report.json`.
Raw-input SHA-256 hashes are saved in selection metadata; the test report records the model hash.
Historical research entry point: `python scripts/model_experiment.py --help`.
Use `scripts/build_model.py` to reproduce the frozen recipe without another holdout evaluation.
Generated artifacts stay local.

See [API verification](api.md) for serving behavior and Python 3.11 container checks.
The demographic tradeoff and capacity drift remain documented limitations of the frozen model.
