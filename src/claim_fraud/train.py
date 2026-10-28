"""Run: python -m claim_fraud.train --data-dir data."""
import argparse
import json
import platform
from pathlib import Path

import joblib
import numpy as np
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .data import assemble, NUMERIC, CATEGORICAL, DEMOGRAPHIC
from .features import ClaimFeatures
from .evaluate import metrics, select_threshold


def build_pipeline(class_weight=None, demographics=False, boosting=None):
    numeric = NUMERIC + (["age"] if demographics else [])
    categorical = CATEGORICAL + (DEMOGRAPHIC if demographics else [])
    preprocessing = ColumnTransformer([
        ("numeric", Pipeline([
            ("impute", SimpleImputer(strategy="median", add_indicator=True,
                                    keep_empty_features=True)),
            ("scale", StandardScaler()),
        ]), numeric),
        ("categorical", Pipeline([
            ("impute", SimpleImputer(strategy="most_frequent")),
            ("encode", OneHotEncoder(handle_unknown="ignore", sparse_output=boosting is None)),
        ]), categorical),
    ])
    classifier = (LogisticRegression(class_weight=class_weight, max_iter=2000,
                                     random_state=20261028, solver="lbfgs")
                  if boosting is None else HistGradientBoostingClassifier(
                      early_stopping=False, random_state=20261028, **boosting))
    return Pipeline([
        ("features", ClaimFeatures(demographics=demographics)),
        ("preprocessing", preprocessing), ("classifier", classifier),
    ])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output-dir", type=Path, default=Path("models/baseline"))
    args = parser.parse_args()
    frame = assemble(args.data_dir)
    train = frame.loc[frame.split.eq("train") & frame.eligible]
    valid = frame.loc[frame.split.eq("validation") & frame.eligible]
    # No test targets or predictions are used below.
    report = {"dataset": "synthetic data", "seed": 20261028,
              "python": platform.python_version(), "sklearn": sklearn.__version__,
              "feature_set": "no_demographics", "capacity": 0.05,
              "label_maturity": "Assumed available at cutoff; timestamps absent",
              "counts": {}, "candidates": {}}
    for name in ("train", "validation"):
        subset = frame.loc[frame.split.eq(name)]
        report["counts"][name] = {"original": len(subset),
                                  "excluded_late_reports": int((~subset.eligible).sum()),
                                  "eligible": int(subset.eligible.sum())}
    candidates = []
    for weight in (None, "balanced"):
        model = build_pipeline(weight)
        model.fit(train, train.is_fraud)
        if np.any(model.named_steps["classifier"].n_iter_ >= 2000):
            raise RuntimeError("Logistic regression did not converge")
        scores = model.predict_proba(valid)[:, 1]
        threshold = select_threshold(valid.is_fraud, scores)
        result = {"threshold": threshold, "validation": metrics(valid.is_fraud, scores, threshold),
                  "sensitivity": {str(capacity): {
                      "threshold": select_threshold(valid.is_fraud, scores, capacity),
                      **metrics(valid.is_fraud, scores,
                                select_threshold(valid.is_fraud, scores, capacity))}
                      for capacity in (0.02, 0.1)}}
        name = "unweighted" if weight is None else weight
        report["candidates"][name] = result
        candidates.append((result["validation"]["recall"],
                           result["validation"]["average_precision"], weight is None,
                           name, model))
    _, _, _, selected, model = max(candidates, key=lambda item: item[:3])
    report["selected"] = selected
    report["threshold"] = report["candidates"][selected]["threshold"]
    report["input_columns"] = model.named_steps["features"].input_columns_
    report["model_features"] = NUMERIC + CATEGORICAL
    report["encoded_features"] = model.named_steps["preprocessing"].get_feature_names_out().tolist()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "threshold": report["threshold"],
                 "metadata": report}, args.output_dir / "model.joblib")
    (args.output_dir / "report.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"selected": selected, "threshold": report["threshold"],
                      **report["candidates"][selected]["validation"]}, indent=2))


if __name__ == "__main__":
    main()
