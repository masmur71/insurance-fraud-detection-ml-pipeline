"""Tie-safe review-capacity threshold selection without test-set access."""
import numpy as np
from sklearn.metrics import (average_precision_score, brier_score_loss, confusion_matrix,
                             precision_score, recall_score, f1_score, roc_auc_score)


def select_threshold(y, scores, capacity=0.05):
    y, scores = np.asarray(y), np.asarray(scores)
    if len(y) == 0 or not 0 < capacity <= 1 or not np.isfinite(scores).all():
        raise ValueError("Nonempty finite scores and capacity in (0,1] required")
    order = np.argsort(-scores, kind="stable")
    ordered = scores[order]
    ends = np.r_[np.flatnonzero(ordered[:-1] != ordered[1:]), len(y) - 1]
    flagged = ends + 1
    positives = np.cumsum(y[order])[ends]
    feasible = flagged <= int(np.floor(capacity * len(y)))
    candidates = [(int(tp), float(tp / count), float(ordered[end]))
                  for end, count, tp in zip(ends[feasible], flagged[feasible], positives[feasible])]
    # Zero flags participates in the same recall/precision/threshold tie rules.
    candidates.append((0, 0.0, float(np.nextafter(scores.max(), np.inf))))
    return max(candidates)[2]


def metrics(y, scores, threshold):
    prediction = np.asarray(scores) >= threshold
    return {"average_precision": float(average_precision_score(y, scores)),
            "roc_auc": float(roc_auc_score(y, scores)),
            "brier_score": float(brier_score_loss(y, scores)),
            "precision": float(precision_score(y, prediction, zero_division=0)),
            "recall": float(recall_score(y, prediction, zero_division=0)),
            "f1": float(f1_score(y, prediction, zero_division=0)),
            "review_fraction": float(prediction.mean()),
            "confusion_matrix": confusion_matrix(y, prediction, labels=[0, 1]).tolist()}
