"""Drift behavior, privacy, frozen transformer, and training-only input guards."""
from copy import deepcopy
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import pytest

from claim_fraud.data import assemble
from claim_fraud.drift import (analyze, build_reference, compare_features, feature_statistics,
                               training_inputs, validate_claims, write_report)
from claim_fraud.features import ClaimFeatures

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs():
    record = json.loads((ROOT / "examples/claim.json").read_text())
    frame = pd.DataFrame([record] * 100)
    frame["vehicle_value"] = np.tile([100., 200.], 50)
    return frame


@pytest.fixture
def artifact(tmp_path, inputs):
    # Only the feature transformer is fitted on controlled fixture data.
    from sklearn.pipeline import Pipeline
    transformer = ClaimFeatures(demographics=True).fit(inputs)
    path = tmp_path / "model.joblib"
    joblib.dump({"pipeline": Pipeline([("features", transformer)]),
                 "threshold": 0.08607093703233727}, path)
    return path


def test_identical_and_controlled_changes(artifact, inputs):
    with patch.object(ClaimFeatures, "fit", side_effect=AssertionError("must not fit")):
        reference = build_reference(artifact, inputs, {"kind": "controlled_fixture"})
        report = analyze(artifact, reference, inputs, {"kind": "identical"})
    assert report["drift_detected"] is False
    for feature in report["features"].values():
        assert feature["missing_rate_difference"] == 0
        assert feature["standardized_mean_shift"] in (None, 0)
        assert feature["total_variation"] in (None, 0)
        assert feature["constant_value_change"] in (None, False)
    changed = inputs.copy()
    changed["vehicle_value"] += 1000
    changed["incident_type"] = "new-category"
    changed.loc[:29, "vehicle_age_years"] = None
    shifted = analyze(artifact, reference, changed, {})
    assert shifted["drift_detected"] is True
    assert shifted["features"]["vehicle_value"]["flags"] == ["standardized_mean_shift"]
    assert shifted["features"]["incident_type"]["total_variation"] == 1
    assert shifted["features"]["incident_type"]["unseen_categories"] == ["new-category"]
    assert "missing_rate" in shifted["features"]["vehicle_age_years"]["flags"]


def test_constant_and_missing(artifact, inputs):
    inputs["vehicle_age_years"] = None
    reference = build_reference(artifact, inputs, {})
    changed = inputs.copy()
    changed["vehicle_age_years"] = 3.
    changed["age"] = inputs.age.iloc[0] + 1
    changed["vehicle_value"] = None
    report = analyze(artifact, reference, changed, {})
    assert report["features"]["vehicle_age_years"]["status"] == "insufficient_reference_data"
    assert report["features"]["age"]["constant_value_change"] is True
    assert report["features"]["vehicle_value"]["status"] == "insufficient_current_data"
    assert report["features"]["vehicle_value"]["standardized_mean_shift"] is None
    json.dumps(report, allow_nan=False)


@pytest.mark.parametrize("count, expected", [(99, "insufficient_data"), (100, "complete")])
def test_minimum_sample(artifact, inputs, count, expected):
    ref = build_reference(artifact, inputs, {})
    changed = inputs.iloc[:count].copy()
    changed["incident_type"] = "new-category"
    result = analyze(artifact, ref, changed, {})
    assert result["status"] == expected
    assert result["drift_detected"] is (None if count < 100 else True)
    assert result["flagged_features"]


def test_strict_threshold_boundaries():
    ref = {"n": {"kind": "numeric", "missing_rate": 0., "available": 100,
                  "mean": 0., "std": 1., "minimum": -1., "maximum": 1.},
           "c": {"kind": "categorical", "missing_rate": 0., "available": 100,
                  "probabilities": {"a": 1.}}}
    current = deepcopy(ref)
    current["n"].update(mean=0.5, missing_rate=0.05)
    current["c"]["probabilities"] = {"a": 0.9, "b": 0.1}
    result = compare_features(ref, current)
    assert not result["n"]["flags"]
    assert not result["c"]["flags"]
    current["n"].update(mean=0.50001, missing_rate=0.05001)
    current["c"]["probabilities"] = {"a": 0.89, "b": 0.11}
    result = compare_features(ref, current)
    assert len(result["n"]["flags"]) == 2
    assert result["c"]["flags"] == ["total_variation"]


def test_fractional_constant_and_equal_mean_change():
    frame = pd.DataFrame({"value": [0.1] * 100})
    ref = feature_statistics(frame)
    same = compare_features(ref, feature_statistics(frame))["value"]
    assert same["constant_value_change"] is False
    assert same["standardized_mean_shift"] is None
    changed = pd.DataFrame({"value": [0., 0.2] * 50})
    result = compare_features(ref, feature_statistics(changed))["value"]
    assert result["constant_value_change"] is True
    assert result["flags"] == ["constant_value_change"]


@pytest.mark.parametrize("case", ["empty", "extra", "missing", "invalid", "timestamp", "infinity", "nested"])
def test_invalid_input_is_private(inputs, case):
    frame = inputs.copy()
    if case == "empty":
        frame = frame.iloc[:0]
    elif case == "extra":
        frame["secret"] = "DO_NOT_ECHO"
    elif case == "missing":
        frame = frame.drop(columns="age")
    elif case == "invalid":
        frame.loc[3, "incident_type"] = "DO_NOT_ECHO" * 20
    elif case == "timestamp":
        frame["incident_date"] = pd.Timestamp("2023-01-01 12:00:00")
    elif case == "nested":
        frame["incident_type"] = [["DO_NOT_ECHO"]] * len(frame)
    else:
        frame["vehicle_value"] = float("inf")
    with pytest.raises(ValueError) as failure:
        validate_claims(frame)
    assert "DO_NOT_ECHO" not in str(failure.value)
    if case == "invalid":
        assert '"row": 3' in str(failure.value)
        assert "incident_type" in str(failure.value)


def test_parquet_roundtrip(tmp_path, inputs):
    inputs["incident_date"] = pd.to_datetime(inputs.incident_date)
    inputs["vehicle_value"] = np.nan
    inputs["incident_hour"] = 12.
    inputs.loc[0, "incident_hour"] = np.nan
    path = tmp_path / "input.parquet"
    inputs.to_parquet(path, index=False)
    valid = validate_claims(pd.read_parquet(path))
    assert valid.vehicle_value.isna().all()
    assert valid.incident_hour.iloc[1] == 12
    assert str(valid.incident_date.iloc[0]) == str(inputs.incident_date.iloc[0].date())


@pytest.mark.parametrize("value", [12.5, True, "12"])
def test_integer_normalization_rejects_invalid_values(inputs, value):
    inputs["incident_hour"] = value
    with pytest.raises(ValueError, match="incident_hour"):
        validate_claims(inputs)


def test_model_identity_and_profile_version(artifact, inputs):
    ref = build_reference(artifact, inputs, {})
    for changes, match in (({"model_sha256": "different"}, "hash mismatch"),
                           ({"profile_version": 2}, "version")):
        with pytest.raises(ValueError, match=match):
            analyze(artifact, {**ref, **changes}, inputs, {})


def test_report_safe_standalone_html(tmp_path, artifact, inputs):
    ref = build_reference(artifact, inputs, {})
    report = analyze(artifact, ref, inputs, {"file": "<script>alert(1)</script>"})
    output = tmp_path / "report"
    write_report(output, report)
    text = (output / "drift-report.html").read_text()
    assert "<script>" not in text
    assert "&lt;script&gt;" in text
    assert json.loads((output / "drift-report.json").read_text()) == report
    assert "<style>" in text
    with pytest.raises(ValueError, match="empty"):
        write_report(output, report)


def test_training_cutoff_and_eligible_selection(inputs):
    frame = inputs.copy()
    frame["split"] = "train"
    frame["eligible"] = True
    frame.loc[0, "eligible"] = False
    with patch("claim_fraud.drift.assemble", return_value=frame) as read:
        result = training_inputs("unused")
        assert read.call_args.kwargs["before"] == pd.Timestamp("2024-01-01")
        assert len(result) == 99
        assert set(result.columns) == set(inputs.columns)
    with patch("claim_fraud.data.pd.read_parquet", side_effect=RuntimeError("stop")) as read:
        with pytest.raises(RuntimeError):
            assemble("unused", before=pd.Timestamp("2024-01-01"))
        assert read.call_args.kwargs["filters"] == [
            ("incident_date", "<", pd.Timestamp("2024-01-01"))]


def test_metric_nonfinite_rejected():
    with pytest.raises(ValueError, match="Non-finite"):
        feature_statistics(pd.DataFrame({"value": [float("inf")]}))


def test_report_cli_validation_and_outputs(tmp_path, artifact, inputs):
    reference = build_reference(artifact, inputs, {"kind": "controlled_fixture"})
    ref_path = tmp_path / "reference.json"
    ref_path.write_text(json.dumps(reference))
    batch = tmp_path / "claims.parquet"
    output = tmp_path / "report"
    command = [sys.executable, str(ROOT / "scripts/report_drift.py"),
               "--model", str(artifact), "--reference", str(ref_path),
               "--input", str(batch), "--output-dir", str(output)]
    invalid = inputs.copy()
    invalid["incident_type"] = "DO_NOT_ECHO" * 20
    invalid.to_parquet(batch, index=False)
    failure = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert failure.returncode == 2
    assert "DO_NOT_ECHO" not in failure.stderr
    assert "incident_type" in failure.stderr
    assert not output.exists()
    inputs.to_parquet(batch, index=False)
    success = subprocess.run(command, capture_output=True, text=True, timeout=30)
    assert success.returncode == 0, success.stderr
    result = json.loads((output / "drift-report.json").read_text())
    assert result["drift_detected"] is False
    assert result["reference_training_input_sha256"] == reference["training_input_sha256"]
    assert len(result["input"]["sha256"]) == 64
    assert (output / "drift-report.html").is_file()
