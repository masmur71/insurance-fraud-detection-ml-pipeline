"""Explicit, optional tracking of the frozen recipe (never imported by serving)."""
import hashlib
import sys
from urllib.parse import urlparse


def log_build(uri, experiment_name, output_dir, report):
    """Upload completed artifacts; propagate every tracking failure to the caller."""
    if urlparse(uri).scheme not in {"http", "https"}:
        raise ValueError("Tracking requires an explicit HTTP(S) server URI")
    import mlflow

    # MLflow prints Unicode run links even when Windows redirects stdout as cp1252.
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="backslashreplace")
    mlflow.set_tracking_uri(uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name="frozen-recipe-build") as run:
        mlflow.set_tags({"recipe": report["recipe"],
                         "source_commit": report["source_commit"],
                         "working_tree_dirty": str(report["working_tree_dirty"]).lower(),
                         "evaluation_scope": "validation-only; consumed holdout excluded"})
        params = {**report["selected"]["config"], "demographics": True,
                  "threshold": report["threshold"], "seed": report["seed"],
                  "python": report["python"], "sklearn": report["sklearn"],
                  **{f"rows_{key}": value for key, value in report["counts"].items()},
                  **{f"sha256_{key}": value for key, value in report["data_sha256"].items()}}
        for name in ("model.joblib", "build-report.json"):
            params[f"sha256_{name}"] = hashlib.sha256((output_dir / name).read_bytes()).hexdigest()
        mlflow.log_params(params)
        validation = report["selected"]["validation"]
        mlflow.log_metrics({f"validation_{key}": value for key, value in validation.items()
                            if key != "confusion_matrix"})
        for i, row in enumerate(validation["confusion_matrix"]):
            mlflow.log_metrics({f"validation_confusion_{i}_{j}": value
                                for j, value in enumerate(row)})
        for name in ("model.joblib", "build-report.json"):
            mlflow.log_artifact(str(output_dir / name))
        return run.info.run_id
