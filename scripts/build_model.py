"""Reproduce the selected model from raw tables without selecting on or scoring test."""
import argparse
import platform
import subprocess
import sys
from pathlib import Path

import joblib
import sklearn
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claim_fraud.data import assemble
from claim_fraud.evaluate import metrics
from claim_fraud.experiment import data_hashes, write_json
from claim_fraud.train import build_pipeline

THRESHOLD = 0.08607093703233727
CONFIG = {"max_leaf_nodes": 7, "max_iter": 100, "l2_regularization": 10.0,
          "learning_rate": 0.05, "min_samples_leaf": 50}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "models/phase-3")
    parser.add_argument("--tracking-uri", help="Explicit HTTP MLflow server; omitted disables tracking")
    parser.add_argument("--experiment-name", default="frozen-phase-3-reproduction")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        raise FileExistsError("Model output must be empty; preserve the original frozen experiment")
    import pandas as pd
    frame = assemble(args.data_dir, before=pd.Timestamp("2024-07-01"))
    train = frame.loc[frame.split.eq("train") & frame.eligible]
    valid = frame.loc[frame.split.eq("validation") & frame.eligible]
    with threadpool_limits(limits=2):
        model = build_pipeline(demographics=True, boosting=CONFIG).fit(train, train.is_fraud)
        validation = metrics(valid.is_fraud, model.predict_proba(valid)[:, 1], THRESHOLD)
    commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    dirty = bool(subprocess.check_output(
        ["git", "status", "--porcelain", "--untracked-files=normal"], cwd=ROOT, text=True).strip())
    report = {"dataset": "synthetic data", "seed": 20261028,
              "source_commit": commit, "working_tree_dirty": dirty,
              "counts": {"train": len(train), "validation": len(valid)},
              "sklearn": sklearn.__version__, "python": platform.python_version(),
              "input_columns": model.named_steps["features"].input_columns_,
              "data_sha256": data_hashes(args.data_dir),
              "recipe": "Frozen phase 3 demographic boosting; no test evaluation or retuning",
              "selected": {"config": CONFIG, "validation": validation}, "threshold": THRESHOLD}
    args.output_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"pipeline": model, "threshold": THRESHOLD, "metadata": report},
                args.output_dir / "model.joblib")
    write_json(args.output_dir / "build-report.json", report)
    if args.tracking_uri:
        from claim_fraud.tracking import log_build
        try:
            run_id = log_build(args.tracking_uri, args.experiment_name, args.output_dir, report)
        except Exception as exc:
            raise RuntimeError(
                f"Tracking failed; completed local artifacts preserved at {args.output_dir}"
            ) from exc
        print(f"MLflow run: {run_id}")
    print(f"Built frozen model recipe from {len(train):,} train claims; test not scored.")


if __name__ == "__main__":
    main()
