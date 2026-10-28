"""Tracking is opt-in; failures preserve the completed frozen build."""
import importlib.util
from pathlib import Path
import sys
from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

from claim_fraud.tracking import log_build
from claim_fraud.data import assemble

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("build_model", ROOT / "scripts/build_model.py")
build = importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)


def test_server_uri_required():
    with pytest.raises(ValueError, match="HTTP"):
        log_build("file:///tmp/mlruns", "test", Path("unused"), {})


def test_holdout_excluded_at_parquet_reader():
    cutoff = pd.Timestamp("2024-07-01")
    with patch("claim_fraud.data.pd.read_parquet", side_effect=RuntimeError("stop")) as read:
        with pytest.raises(RuntimeError, match="stop"):
            assemble("unused", before=cutoff)
        assert read.call_args.kwargs["filters"] == [("incident_date", "<", cutoff)]


@pytest.mark.parametrize("tracking", [False, True])
def test_build_optional_tracking_and_failure_preserves_artifacts(tmp_path, tracking):
    frame = pd.DataFrame({"split": ["train", "validation"], "eligible": [True, True],
                          "is_fraud": [0, 1]})
    model = MagicMock()
    model.fit.return_value = model
    model.predict_proba.return_value = __import__("numpy").array([[0.2, 0.8]])
    model.named_steps["features"].input_columns_ = ["age"]
    argv = ["build_model.py", "--output-dir", str(tmp_path)]
    if tracking:
        argv += ["--tracking-uri", "http://127.0.0.1:1"]
    with patch.object(sys, "argv", argv), patch.object(build, "assemble", return_value=frame) as read, \
            patch.object(build, "build_pipeline", return_value=model), \
            patch.object(build, "metrics", return_value={"recall": 1.0}), \
            patch.object(build, "data_hashes", return_value={}), \
            patch.object(build.subprocess, "check_output", side_effect=["abc123", ""]), \
            patch.object(build.joblib, "dump", side_effect=lambda _, path: path.write_bytes(b"model")), \
            patch("claim_fraud.tracking.log_build", side_effect=ConnectionError) as upload:
        if tracking:
            with pytest.raises(RuntimeError, match="artifacts preserved"):
                build.main()
            upload.assert_called_once()
        else:
            build.main()
            upload.assert_not_called()
        assert read.call_args.kwargs["before"] == pd.Timestamp("2024-07-01")
        assert (tmp_path / "model.joblib").exists()
        assert (tmp_path / "build-report.json").exists()
