"""Selection isolation, frozen thresholds, and demographic serving parity."""
from io import BytesIO
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from claim_fraud.data import DEMOGRAPHIC
from claim_fraud.experiment import evaluate_holdout, main, selection_key, tune
from claim_fraud.train import build_pipeline
from test_baseline import fixture


def experiment_frame():
    frame = pd.concat([fixture()] * 3, ignore_index=True)
    frame["age"] = 30
    for column in DEMOGRAPHIC:
        frame[column] = "a"
    frame["split"] = ["train"] * 4 + ["validation"] * 4 + ["test"] * 4
    frame["eligible"] = True
    frame["is_fraud"] = [0, 1, 0, 1] * 3
    return frame


class ExperimentTests(unittest.TestCase):
    def test_evaluation_refuses_changed_data_and_repeat_runs(self):
        with TemporaryDirectory(dir=Path.cwd()) as directory:
            arguments = ["experiment", "evaluate", "--output-dir", directory]
            with patch("sys.argv", arguments), patch(
                    "claim_fraud.experiment.data_hashes", return_value={"claims": "changed"}), patch(
                    "claim_fraud.experiment.joblib.load",
                    return_value={"metadata": {"data_sha256": {"claims": "original"}}}), patch(
                    "claim_fraud.experiment.assemble") as assemble:
                with self.assertRaisesRegex(ValueError, "Data changed"):
                    main()
                assemble.assert_not_called()
            (Path(directory) / "test-report.json").write_text("{}", encoding="utf-8")
            with patch("sys.argv", arguments), self.assertRaisesRegex(
                    FileExistsError, "already evaluated"):
                main()

    def test_ties_prefer_no_demographics_then_logistic(self):
        def item(demographics, family, recall=0.5):
            return {"validation": {"recall": recall, "average_precision": 0.2},
                    "config": {"demographics": demographics, "family": family}}
        self.assertGreater(selection_key(item(False, "boosting")),
                           selection_key(item(True, "logistic")))
        self.assertGreater(selection_key(item(False, "logistic")),
                           selection_key(item(False, "boosting")))
        self.assertGreater(selection_key(item(True, "boosting", 0.6)),
                           selection_key(item(False, "logistic")))

    def test_boosting_demographics_reload_unknown_and_missing(self):
        frame = experiment_frame()
        model = build_pipeline(demographics=True, boosting={
            "max_iter": 2, "min_samples_leaf": 2}).fit(frame, frame.is_fraud)
        request = frame.iloc[:2].copy()
        request[DEMOGRAPHIC] = "unseen"
        request["age"] = np.nan
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        actual = joblib.load(buffer).predict_proba(request)
        np.testing.assert_allclose(actual, model.predict_proba(request))
        self.assertTrue(np.isfinite(actual).all())
        self.assertIn("age", model.named_steps["features"].input_columns_)

    def test_tuning_never_fits_or_predicts_test_and_holdout_never_retunes(self):
        frame = experiment_frame()
        real_build = build_pipeline
        seen = []

        def guarded_build(*args):
            model = real_build(*args)
            original_fit, original_predict = model.fit, model.predict_proba

            def fit(x, y):
                self.assertEqual(set(x.split), {"train"})
                seen.append("fit")
                return original_fit(x, y)

            def predict(x):
                self.assertEqual(set(x.split), {"validation"})
                seen.append("predict")
                return original_predict(x)

            model.fit, model.predict_proba = fit, predict
            return model

        config = {"family": "logistic", "demographics": False,
                  "class_weight": None, "boosting": None}
        with TemporaryDirectory(dir=Path.cwd()) as directory, patch(
                "claim_fraud.experiment.configurations", return_value=[config]), patch(
                "claim_fraud.experiment.build_pipeline", side_effect=guarded_build), patch(
                "claim_fraud.experiment.joblib.dump"):
            report = tune(frame, Path(directory), {})
        self.assertEqual(seen, ["fit", "predict"])
        model = real_build().fit(frame.iloc[:4], frame.is_fraud.iloc[:4])
        bundle = {"pipeline": model, "threshold": 0.0, "metadata": report}
        with patch("claim_fraud.experiment.select_threshold", side_effect=AssertionError):
            result = evaluate_holdout(frame, bundle)
        self.assertEqual(result["threshold"], 0.0)
        self.assertEqual(result["test"]["review_fraction"], 1.0)
        self.assertEqual(sum(row["n"] for row in result["error_slices"]["incident_type"]), 4)


if __name__ == "__main__":
    unittest.main()
