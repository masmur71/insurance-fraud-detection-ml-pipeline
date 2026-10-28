# Phase 2 baseline results

All results: **synthetic data**, seed `20261028`, validation only.
Runtime: Python 3.13.0, scikit-learn 1.6.1. Python 3.11 verification remains pending.

## Data and selection

Train: 79,329 original claims, 977 unavailable at cutoff, 78,352 eligible.
Validation: 15,555 original claims, 955 unavailable at cutoff, 14,600 eligible.
Test is not evaluated or used for model/threshold selection in this phase.
Class labels are assumed available at cutoff because label-maturity timestamps are absent.
The raw data is not modified; precomputed feature and truth tables are not read by training.

Unweighted logistic regression beats balanced weighting on validation recall at the
5% review capacity (27.52% vs 26.78%) and average precision (0.1349 vs 0.1207).
Selected threshold: 0.09012407043686738, applied with score >= threshold.
ROC-AUC: 0.7438; Brier score: 0.02569; F1: 0.2029.
Confusion matrix: TN 13,608; FP 585; FN 295; TP 112.

## Capacity sensitivity for selected model

| Capacity ceiling | Actual review fraction | Precision | Recall |
|---|---:|---:|---:|
| 2% | 2.00% | 21.92% | 15.72% |
| 5% | 4.77% | 16.07% | 27.52% |
| 10% | 9.90% | 10.86% | 38.57% |

The threshold selector maximizes recall, then precision, then threshold. It can
leave capacity unused when adding lower-score claims brings no additional positives.
Test/production review volumes may differ because thresholds are frozen.
Scores are not certified calibrated probabilities; no real-world fraud claim follows.

## Verification and next phase

Seven behavioral tests pass: history time/window boundaries, threshold tie handling
and exhaustive selection parity, train-fitted clipping and feature allowlisting,
invalid dates/ratios, serialization with unseen categories and missing values,
and raw assembly with duplicate/cutoff/label-independence checks.
The full-data training command completed without convergence failure.
The saved trained bundle also passed reload prediction parity and feature checks.

Phase 3 compares gradient boosting and demographic feature sets, freezes the winning
configuration on validation, then performs final test and drift/error analysis.
