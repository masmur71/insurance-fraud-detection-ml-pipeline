"""HTTP contract, frozen-model parity, and startup failure behavior."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd
import sklearn
from fastapi.testclient import TestClient
from test_experiment import experiment_frame

from claim_fraud.api import create_app
from claim_fraud.train import build_pipeline

ROOT = Path(__file__).resolve().parents[1]


class APITests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = TemporaryDirectory(dir=ROOT)
        cls.path = Path(cls.directory.name) / "model.joblib"
        frame = experiment_frame()
        cls.model = build_pipeline(demographics=True, boosting={
            "max_iter": 2, "min_samples_leaf": 2}).fit(frame, frame.is_fraud)
        cls.bundle = {"pipeline": cls.model, "threshold": 0.5, "metadata": {
            "sklearn": sklearn.__version__, "dataset": "synthetic data", "seed": 20261028,
            "input_columns": cls.model.named_steps["features"].input_columns_}}
        joblib.dump(cls.bundle, cls.path)
        cls.payload = json.loads((ROOT / "examples/claim.json").read_text())

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def test_health_prediction_and_direct_pipeline_parity(self):
        with TestClient(create_app(self.path)) as client:
            health = client.get("/health")
            self.assertEqual(health.status_code, 200)
            response = client.post("/predict", json=self.payload)
            self.assertEqual(response.status_code, 200, response.text)
            body = response.json()
            expected = self.model.predict_proba(pd.DataFrame([self.payload]))[0, 1]
            self.assertAlmostEqual(body["score"], expected)
            self.assertEqual(body["review_required"], expected >= 0.5)
            self.assertEqual(body["model_version"], health.json()["model_version"])
            self.assertEqual(body["dataset"], "synthetic data")
            self.assertEqual(body["score_kind"], "uncalibrated_model_score")
            self.assertEqual(client.get("/openapi.json").status_code, 200)

    def test_missing_unknown_and_invalid_delay_match_training(self):
        payload = {**self.payload, "age": None, "incident_type": "never-seen",
                   "claim_amount": None, "occupation_group": None,
                   "annual_premium": 0, "report_date": "2025-01-09"}
        with TestClient(create_app(self.path)) as client:
            response = client.post("/predict", json=payload)
        self.assertEqual(response.status_code, 200, response.text)
        expected = self.model.predict_proba(pd.DataFrame([payload]))[0, 1]
        self.assertAlmostEqual(response.json()["score"], expected)
        self.assertIn("INVALID_REPORT_DELAY", response.json()["reason_codes"])

    def test_rejects_bad_inputs_and_outcome_fields(self):
        invalid = [{"prior_claims_12m": 2, "prior_claims_total": 1},
                   {"prior_claims_total": -1}, {"prior_claims_total": 1.5},
                   {"prior_claims_total": True}, {"claim_amount": "100"},
                   {"claim_amount": -1}, {"claim_amount": "NaN"},
                   {"incident_hour": 24}, {"police_report_filed": 2},
                   {"incident_type": "  "}, {"incident_type": "a" * 101},
                   {"age": 121}, {"report_date": "2025-02-30"},
                   {"report_date": "9999-01-01"}, {"report_date": 123},
                   {"report_date": "2025-01-01T00:00:00"},
                   {"true_fraud": 1}, {"claim_status": "settled"}]
        with TestClient(create_app(self.path)) as client:
            for changes in invalid:
                with self.subTest(changes=changes):
                    response = client.post("/predict", json={**self.payload, **changes})
                    self.assertEqual(response.status_code, 422)
            missing = self.payload.copy()
            del missing["prior_claims_total"]
            self.assertEqual(client.post("/predict", json=missing).status_code, 422)
            self.assertEqual(client.post("/predict", content="{", headers={
                "Content-Type": "application/json"}).status_code, 422)
            for number in (float("inf"), float("nan")):
                payload = json.dumps({**self.payload, "claim_amount": number})
                self.assertEqual(client.post("/predict", content=payload, headers={
                    "Content-Type": "application/json"}).status_code, 422)

    def test_rule_indicators_and_threshold_equality(self):
        payload = {**self.payload, "police_report_filed": 0, "claim_amount": 200000000}
        with TestClient(create_app(self.path)) as client, patch.object(
                client.app.state.bundle["pipeline"], "predict_proba",
                return_value=np.array([[0.5, 0.5]])):
            body = client.post("/predict", json=payload).json()
        self.assertTrue(body["review_required"])
        self.assertEqual(body["reason_codes"], ["CLAIM_EXCEEDS_SUM_INSURED", "NO_POLICE_REPORT"])

    def test_missing_or_incompatible_bundle_fails_startup(self):
        with self.assertRaises(FileNotFoundError), TestClient(create_app(self.path.parent / "missing")):
            pass
        bad = {**self.bundle, "metadata": {**self.bundle["metadata"], "sklearn": "0.0"}}
        with patch("claim_fraud.api.joblib.load", return_value=bad), self.assertRaisesRegex(
                ValueError, "versions must match"), TestClient(create_app(self.path)):
            pass
        bad = {**self.bundle, "threshold": float("nan")}
        with patch("claim_fraud.api.joblib.load", return_value=bad), self.assertRaisesRegex(
                ValueError, "threshold"), TestClient(create_app(self.path)):
            pass

    def test_batch_sizes_parity_order_and_single_inference_call(self):
        variants = [self.payload, {**self.payload, "age": None, "claim_amount": None,
                                  "occupation_group": None, "incident_type": "unseen"},
                    {**self.payload, "report_date": "2025-01-09",
                     "police_report_filed": 0, "claim_amount": 200000000}]
        with TestClient(create_app(self.path)) as client:
            expected = [client.post("/predict", json=item).json() for item in variants]
            for size in (1, 3, 100):
                claims = [variants[i % 3] for i in range(size)]
                pipeline = client.app.state.bundle["pipeline"]
                with patch.object(pipeline, "predict_proba", wraps=pipeline.predict_proba) as score:
                    response = client.post("/predict/batch", json={"claims": claims})
                    self.assertEqual(response.status_code, 200, response.text)
                    score.assert_called_once()
                    self.assertEqual(len(score.call_args.args[0]), size)
                for i, actual in enumerate(response.json()["predictions"]):
                    reference = expected[i % 3]
                    self.assertAlmostEqual(actual["score"], reference["score"], delta=1e-12)
                    self.assertEqual({k: v for k, v in actual.items() if k != "score"},
                                     {k: v for k, v in reference.items() if k != "score"})

    def test_batch_validation_is_atomic_and_indexed(self):
        with TestClient(create_app(self.path)) as client:
            for claims in ([], [self.payload] * 101,
                           [self.payload, {**self.payload, "age": 121}]):
                response = client.post("/predict/batch", json={"claims": claims})
                self.assertEqual(response.status_code, 422)
            self.assertIn(["body", "claims", 1, "age"],
                          [error["loc"] for error in response.json()["detail"]])
            self.assertEqual(client.app.state.monitor.snapshot()["predictions"], 0)
            self.assertEqual(client.app.state.monitor.requests[("/predict/batch", 422)], 3)
            schema = client.get("/openapi.json").json()["components"]["schemas"]
            field = schema["BatchClaimRequest"]["properties"]["claims"]
            self.assertEqual((field["minItems"], field["maxItems"]), (1, 100))

    def test_batch_counts_requests_and_claims_and_rejects_invalid_model_scores_atomically(self):
        with TestClient(create_app(self.path), raise_server_exceptions=False) as client:
            pipeline = client.app.state.bundle["pipeline"]
            with patch.object(pipeline, "predict_proba", return_value=np.array(
                    [[0.5, 0.5], [0.8, 0.2]])):
                response = client.post("/predict/batch", json={"claims": [self.payload] * 2})
            self.assertEqual(response.status_code, 200)
            snapshot = client.app.state.monitor.snapshot()
            self.assertEqual(snapshot["predictions"], 2)
            self.assertEqual(snapshot["review_fraction"], 0.5)
            self.assertEqual(client.app.state.monitor.requests[("/predict/batch", 200)], 1)
            for bad_score in (float("nan"), float("inf"), -0.1, 1.1):
                with patch.object(pipeline, "predict_proba", return_value=np.array(
                        [[0.5, 0.5], [0.0, bad_score]])):
                    response = client.post("/predict/batch", json={"claims": [self.payload] * 2})
                self.assertEqual(response.status_code, 500)
                self.assertEqual(client.app.state.monitor.snapshot()["predictions"], 2)


if __name__ == "__main__":
    unittest.main()
