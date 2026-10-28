"""Validation-only tuning followed by a separate, frozen holdout evaluation."""
import argparse
import hashlib
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from threadpoolctl import threadpool_limits

from .data import assemble
from .evaluate import metrics, select_threshold
from .train import build_pipeline


def write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def data_hashes(data_dir):
    return {name: hashlib.sha256((data_dir / name).read_bytes()).hexdigest()
            for name in ("claims.parquet", "policies.parquet", "customers.parquet")}


def configurations():
    for demographics in (False, True):
        for weight in (None, "balanced"):
            yield {"family": "logistic", "demographics": demographics, "class_weight": weight,
                   "boosting": None}
        for leaves, iterations, regularization in ((7, 100, 10.0), (15, 150, 10.0),
                                                   (15, 150, 50.0)):
            yield {"family": "boosting", "demographics": demographics, "class_weight": None,
                   "boosting": {"max_leaf_nodes": leaves, "max_iter": iterations,
                                "l2_regularization": regularization, "learning_rate": 0.05,
                                "min_samples_leaf": 50}}


def selection_key(result):
    return (result["validation"]["recall"], result["validation"]["average_precision"],
            not result["config"]["demographics"], result["config"]["family"] == "logistic")


def tune(frame, output_dir, hashes):
    train = frame.loc[frame.split.eq("train") & frame.eligible]
    valid = frame.loc[frame.split.eq("validation") & frame.eligible]
    report = {"dataset": "synthetic data", "seed": 20261028,
              "python": platform.python_version(), "sklearn": sklearn.__version__,
              "data_sha256": hashes, "capacity": 0.05,
              "label_maturity": "Assumed available at cutoff; timestamps absent",
              "selection": "Validation recall, AP, no demographics, logistic regression",
              "counts": {}, "candidates": []}
    for split in ("train", "validation"):
        subset = frame.loc[frame.split.eq(split)]
        report["counts"][split] = {"original": len(subset), "eligible": int(subset.eligible.sum()),
                                   "excluded_late_reports": int((~subset.eligible).sum())}
    best, winner, winner_scores = None, None, None
    for config in configurations():
        model = build_pipeline(config["class_weight"], config["demographics"], config["boosting"])
        model.fit(train, train.is_fraud)
        if config["family"] == "logistic" and np.any(
                model.named_steps["classifier"].n_iter_ >= 2000):
            raise RuntimeError("Logistic regression did not converge")
        scores = model.predict_proba(valid)[:, 1]
        threshold = select_threshold(valid.is_fraud, scores)
        result = {"config": config, "threshold": threshold,
                  "validation": metrics(valid.is_fraud, scores, threshold)}
        report["candidates"].append(result)
        if best is None or selection_key(result) > selection_key(best):
            best, winner, winner_scores = result, model, scores
    report["selected"] = best
    report["sensitivity"] = {}
    for capacity in (0.02, 0.05, 0.1):
        threshold = select_threshold(valid.is_fraud, winner_scores, capacity)
        report["sensitivity"][str(capacity)] = {
            "threshold": threshold, **metrics(valid.is_fraud, winner_scores, threshold)}
    report["input_columns"] = winner.named_steps["features"].input_columns_
    report["encoded_features"] = (
        winner.named_steps["preprocessing"].get_feature_names_out().tolist())
    joblib.dump({"pipeline": winner, "threshold": best["threshold"], "metadata": report},
                output_dir / "model.joblib")
    write_json(output_dir / "selection.json", report)
    return report


def error_slices(frame, scores, threshold):
    """Descriptive slices; small groups are not reliable fairness estimates."""
    audit = frame.copy()
    audit["flagged"] = scores >= threshold
    audit["incident_month"] = audit.incident_date.dt.strftime("%Y-%m")
    audit["age_band"] = pd.cut(audit.age, [0, 25, 40, 60, np.inf]).astype(str)
    result = {}
    for column in ("incident_month", "incident_type", "coverage_type", "province", "age_band"):
        rows = []
        groups = audit[column].fillna("missing").astype(str).str.strip().str.lower()
        for value, part in audit.groupby(groups, sort=True):
            positive, flag = part.is_fraud.eq(1), part.flagged
            tp, fp = int((positive & flag).sum()), int((~positive & flag).sum())
            fn, tn = int((positive & ~flag).sum()), int((~positive & ~flag).sum())
            rows.append({"group": str(value), "n": len(part), "positives": int(positive.sum()),
                         "tp": tp, "fp": fp, "fn": fn, "tn": tn,
                         "recall": tp / (tp + fn) if tp + fn else None,
                         "precision": tp / (tp + fp) if tp + fp else None,
                         "false_positive_rate": fp / (fp + tn) if fp + tn else None,
                         "review_fraction": float(flag.mean())})
        result[column] = rows
    return result


def feature_drift(model, reference, current):
    transform = model.named_steps["features"]
    before, after = transform.transform(reference), transform.transform(current)
    result = {}
    for column in before:
        a, b = before[column], after[column]
        row = {"reference_missing": float(a.isna().mean()), "test_missing": float(b.isna().mean())}
        if pd.api.types.is_numeric_dtype(a):
            scale = a.std()
            shift = (b.mean() - a.mean()) / scale if scale > 0 else np.nan
            row["standardized_mean_shift"] = float(shift) if np.isfinite(shift) else None
        else:
            p, q = a.fillna("missing").value_counts(normalize=True), b.fillna("missing").value_counts(normalize=True)
            keys = p.index.union(q.index)
            row["total_variation"] = float((p.reindex(keys, fill_value=0)
                                            - q.reindex(keys, fill_value=0)).abs().sum() / 2)
        result[column] = row
    return result


def evaluate_holdout(frame, bundle):
    test = frame.loc[frame.split.eq("test") & frame.eligible]
    valid = frame.loc[frame.split.eq("validation") & frame.eligible]
    model, threshold = bundle["pipeline"], bundle["threshold"]
    scores = model.predict_proba(test)[:, 1]
    report = {"dataset": "synthetic data", "seed": 20261028, "n": len(test),
              "positives": int(test.is_fraud.sum()), "threshold": threshold,
              "test": metrics(test.is_fraud, scores, threshold),
              "sensitivity": {capacity: {"threshold": item["threshold"],
                  **metrics(test.is_fraud, scores, item["threshold"])}
                  for capacity, item in bundle["metadata"]["sensitivity"].items()},
              "error_slices": error_slices(test, scores, threshold),
              "validation_to_test_drift": feature_drift(model, valid, test)}
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("tune", "evaluate"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/model-selection"))
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    if (args.output_dir / "test-report.json").exists():
        raise FileExistsError("Holdout already evaluated; preserve this experiment")
    if args.stage == "tune" and any(args.output_dir.iterdir()):
        raise FileExistsError("Tuning requires an empty output directory")
    hashes = data_hashes(args.data_dir)
    bundle = None
    if args.stage == "evaluate":
        bundle = joblib.load(args.output_dir / "model.joblib")
        if hashes != bundle["metadata"]["data_sha256"]:
            raise ValueError("Data changed after selection; holdout evaluation refused")
    frame = assemble(args.data_dir)
    with threadpool_limits(limits=2):
        if args.stage == "tune":
            report = tune(frame, args.output_dir, hashes)
            print(json.dumps(report["selected"], indent=2))
        else:
            report = evaluate_holdout(frame, bundle)
            report["model_sha256"] = hashlib.sha256(
                (args.output_dir / "model.joblib").read_bytes()).hexdigest()
            write_json(args.output_dir / "test-report.json", report)
            print(json.dumps(report["test"], indent=2))


if __name__ == "__main__":
    main()
