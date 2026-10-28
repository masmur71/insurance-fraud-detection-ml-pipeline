# Model card: synthetic insurance claim review

**Purpose:** prioritize reported Indonesian vehicle insurance claims for human review.
The model must not reject claims automatically or be presented as evidence of fraud.

**Data:** entirely **synthetic data**, seed `20261028`, spanning 2021–2025. The observed
`is_fraud` label contains planted noise; latent fraud truth is excluded from training.
No real policyholder records are used. Train: 78,352 eligible claims through 2023;
validation: 14,600 in 2024 H1; test: 55,116 from 2024 H2 through 2025. Late reports
are excluded at train/validation cutoffs. Histories use strictly earlier availability
dates and exclude same-day peers. Label maturity is assumed because timestamps are absent.

**Model:** scikit-learn histogram gradient boosting, 100 iterations, 7 leaves,
learning rate 0.05, L2=10, minimum leaf size 50. Train-fitted clipping, imputation,
scaling, and category encoding are serialized together. Demographic features are
included; no IDs, claim outcomes, or latent truth enter the estimator.

**Selection:** ten predefined candidates compared on validation recall at <=5%
review, then average precision (AP), then preference for no demographics and logistic
regression. Frozen threshold: `0.08607093703233727`. Demographic and non-demographic
boosting tied on recall; demographics won by only 0.00031 AP. This is weak evidence
of benefit and does not establish fairness. No model changes used holdout findings.

| Synthetic benchmark | AP | Precision | Recall | Review fraction |
|---|---:|---:|---:|---:|
| Logistic baseline, validation | 0.1349 | 16.07% | 27.52% | 4.77% |
| Selected boosting, validation | 0.1500 | 16.60% | 29.73% | 4.99% |
| Selected boosting, frozen test | 0.1402 | 14.40% | 33.07% | 7.43% |

**Test errors:** 590 TP, 3,506 FP, 1,194 FN, 49,826 TN. Collision accounts for 594
missed positives. Monthly review reaches 12.93% in October 2025. The validation
capacity constraint does not guarantee test or production capacity.

**Limitations:** scores are uncalibrated. Synthetic performance does not demonstrate
real fraud detection. Historical counts supplied by callers cannot be independently
verified by the API. Missing label-maturity dates, demographic proxies, temporal drift,
and label noise limit validity. Rule reason codes describe inputs, not model explanations.

**Operation:** use for a local demonstration with human review. Monitor request
failures, latency, review fraction, and score distribution. Online monitoring has no
labels and cannot measure precision/recall. Obtain suitable real data, assess subgroup
effects, validate label maturity and calibration, and design a new holdout before any
real deployment. Never retune using the consumed benchmark test set.

**Reproduction:** [feature contract](feature-contract.md), [full analysis](model-selection.md),
and [README](../README.md). Each API response identifies the exact artifact by SHA-256.
