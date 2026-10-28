"""Counters remain bounded and do not retain request data."""
import unittest
from concurrent.futures import ThreadPoolExecutor

import test_api
from fastapi.testclient import TestClient

from claim_fraud.api import create_app
from claim_fraud.monitoring import Monitor


class MonitoringTests(unittest.TestCase):
    def test_concurrent_counts_and_alert_minimum(self):
        monitor = Monitor(0.04)
        self.assertIsNone(monitor.snapshot()["review_fraction"])
        with ThreadPoolExecutor(max_workers=4) as executor:
            list(executor.map(lambda _: monitor.observe_prediction(0.2, True, "fixture"), range(99)))
        self.assertFalse(monitor.snapshot()["review_volume_alert"])
        monitor.observe_prediction(1.0, True, "fixture")
        snapshot = monitor.snapshot()
        self.assertTrue(snapshot["review_volume_alert"])
        self.assertEqual(snapshot["predictions"], 100)
        self.assertEqual(sum(snapshot["score_bins"]), 100)
        self.assertAlmostEqual(snapshot["review_fraction_shift"], 0.96)

    def test_http_success_validation_and_unknown_paths_counted(self):
        test_api.APITests.setUpClass()
        try:
            with TestClient(create_app(test_api.APITests.path)) as client:
                self.assertEqual(client.post("/predict", json=test_api.APITests.payload).status_code, 200)
                self.assertEqual(client.post("/predict", json={}).status_code, 422)
                client.get("/unknown-customer-private-id")
                snapshot = client.get("/monitoring").json()
                metrics = client.get("/metrics").text
            self.assertEqual(snapshot["predictions"], 1)
            self.assertIn('route="/predict",status="422"', metrics)
            self.assertIn('route="other",status="404"', metrics)
            self.assertNotIn("private-id", metrics)
        finally:
            test_api.APITests.tearDownClass()
