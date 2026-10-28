"""Verify drift reports using training and deterministic synthetic changes only."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from claim_fraud.drift import training_inputs, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=ROOT / "models/frozen-model/model.joblib")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "reports/drift-demo")
    args = parser.parse_args()
    if args.output_dir.exists() and any(args.output_dir.iterdir()):
        parser.error("Demo output directory must be empty")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    reference = args.output_dir / "reference.json"
    subprocess.run([sys.executable, str(ROOT / "scripts/build_drift_reference.py"),
                    "--model", str(args.model), "--data-dir", str(args.data_dir),
                    "--output", str(reference)], check=True)
    frame = training_inputs(args.data_dir)
    rng = np.random.default_rng(20261028)
    modified = frame.copy()
    modified["vehicle_value"] = modified.vehicle_value.fillna(0) + 1e9
    modified["incident_type"] = "controlled-new-category"
    indices = rng.permutation(len(modified))[:len(modified) // 2]
    modified.loc[indices, "vehicle_age_years"] = np.nan
    results = {}
    for name, fixture in (("identical", frame), ("controlled-shift", modified)):
        path = args.output_dir / f"{name}.parquet"
        fixture.to_parquet(path, index=False)
        subprocess.run([sys.executable, str(ROOT / "scripts/report_drift.py"),
                        "--model", str(args.model), "--reference", str(reference),
                        "--input", str(path), "--output-dir", str(args.output_dir / name)], check=True)
        report = json.loads((args.output_dir / name / "drift-report.json").read_text())
        results[name] = report
    assert results["identical"]["drift_detected"] is False
    for feature in results["identical"]["features"].values():
        assert feature["missing_rate_difference"] == 0
        assert feature["standardized_mean_shift"] in (None, 0)
        assert feature["total_variation"] in (None, 0)
        assert feature["constant_value_change"] in (None, False)
    shifted = results["controlled-shift"]["features"]
    assert "standardized_mean_shift" in shifted["vehicle_value"]["flags"]
    assert "total_variation" in shifted["incident_type"]["flags"]
    assert "missing_rate" in shifted["vehicle_age_years"]["flags"]
    write_json(args.output_dir / "verification.json", {
        "seed": 20261028, "identical_zero_differences": True,
        "controlled_numeric_category_missingness_flags": True,
        "model_sha256": results["identical"]["model_sha256"],
        "sample_size": len(frame), "holdout_read": False})
    print("Drift demonstration passed: identical data and controlled changes.")


if __name__ == "__main__":
    main()
