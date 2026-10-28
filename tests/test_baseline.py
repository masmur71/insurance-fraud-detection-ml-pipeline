"""Behavioral checks, runnable with unittest or pytest."""
from io import BytesIO
import unittest
from unittest.mock import patch

import joblib
import numpy as np
import pandas as pd

from claim_fraud.data import assemble, past_count, NUMERIC, RAW_NUMERIC, CATEGORICAL, DATES
from claim_fraud.evaluate import select_threshold
from claim_fraud.features import ClaimFeatures
from claim_fraud.train import build_pipeline


def fixture():
    frame = pd.DataFrame({column: [1.0, 2.0, 3.0, 4.0] for column in RAW_NUMERIC})
    for column in CATEGORICAL:
        frame[column] = [" A ", "b", "a", "b"]
    for column in DATES:
        frame[column] = pd.to_datetime(["2023-01-01"] * 4)
    frame["claim_amount"] = [100.0, 200.0, 300.0, 1000.0]
    return frame


class BaselineTests(unittest.TestCase):
    def test_history_excludes_same_day_and_future_includes_window_start(self):
        frame = pd.DataFrame({"customer": ["a"] * 5 + ["b"], "available_date":
                              pd.to_datetime(["2023-01-01", "2023-01-01", "2023-01-03",
                                              "2023-01-04", "2023-01-10", "2023-01-04"])})
        self.assertEqual(past_count(frame, "customer", 3).tolist(), [0, 0, 2, 3, 0, 0])
        self.assertEqual(past_count(frame, "customer").tolist(), [0, 0, 2, 3, 4, 0])

    def test_threshold_ties_cannot_overrun_capacity(self):
        y = [1, 0, 1, 0]
        scores = np.array([0.9, 0.9, 0.5, 0.2])
        threshold = select_threshold(y, scores, 0.25)
        self.assertEqual(int((scores >= threshold).sum()), 0)
        self.assertEqual(select_threshold(y, scores, 0.5), 0.9)

    def test_threshold_matches_exhaustive_selection(self):
        rng = np.random.default_rng(12)
        for _ in range(30):
            scores = rng.integers(0, 10, 50) / 10
            y = rng.integers(0, 2, 50)
            capacity = 0.2
            options = []
            for threshold in np.r_[np.unique(scores), np.nextafter(scores.max(), np.inf)]:
                flags = scores >= threshold
                if flags.mean() <= capacity:
                    tp = int(y[flags].sum())
                    options.append((tp, tp / flags.sum() if flags.any() else 0, threshold))
            self.assertEqual(select_threshold(y, scores, capacity), max(options)[2])

    def test_features_allowlist_and_train_only_cap(self):
        train = fixture()
        features = ClaimFeatures().fit(train)
        request = train.iloc[:1].copy()
        request["claim_amount"] = 1e12
        request["claim_status"] = "rejected"
        request["true_fraud"] = 1
        transformed = features.transform(request)
        self.assertEqual(transformed.claim_amount.iloc[0], features.amount_cap_)
        self.assertEqual(set(transformed.columns), set(NUMERIC + CATEGORICAL))
        self.assertEqual(transformed.incident_type.iloc[0], "a")
        self.assertLess(features.amount_cap_, 1001)

    def test_invalid_delay_and_ratios(self):
        frame = fixture()
        frame.loc[0, "report_date"] = pd.Timestamp("2022-12-31")
        frame.loc[0, "sum_insured"] = 0
        frame.loc[0, "annual_premium"] = -1
        result = ClaimFeatures().fit_transform(frame)
        self.assertEqual(result.invalid_report_delay.iloc[0], 1)
        self.assertTrue(result[["report_delay_days", "claim_to_sum_insured_ratio",
                                "claim_to_premium_ratio"]].iloc[0].isna().all())

    def test_serialization_unknown_categories_and_missing_inputs(self):
        frame = fixture()
        model = build_pipeline().fit(frame, [0, 1, 0, 1])
        request = frame.iloc[:1].copy()
        request["incident_type"] = "new-category"
        request["incident_hour"] = np.nan
        expected = model.predict_proba(request)
        buffer = BytesIO()
        joblib.dump(model, buffer)
        buffer.seek(0)
        np.testing.assert_allclose(joblib.load(buffer).predict_proba(request), expected)
        self.assertTrue(np.isfinite(expected).all())

    def test_raw_assembly_cutoffs_duplicates_and_no_truth_read(self):
        frame = fixture()
        frame["claim_id"] = ["a", "b", "c", "d"]
        frame["customer_id"] = "customer"
        frame["policy_id"] = "policy"
        frame["workshop_id"] = "workshop"
        frame["incident_province"] = "province"
        frame["is_fraud"] = [0, 1, 0, 1]
        frame["incident_date"] = pd.to_datetime(
            ["2023-12-30", "2023-12-31", "2024-06-30", "2024-07-01"])
        frame["report_date"] = pd.to_datetime(
            ["2023-12-30", "2024-01-01", "2024-07-01", "2024-07-02"])
        policies = frame.iloc[:1][["policy_id", "customer_id", "policy_start_date",
                                  "coverage_type", "vehicle_type", "sales_channel",
                                  "payment_frequency", "vehicle_age_years", "vehicle_value",
                                  "sum_insured", "annual_premium"]]
        customers = pd.DataFrame({"customer_id": ["customer"],
                                  "customer_since": [pd.Timestamp("2020-01-01")],
                                  "age": [30], "occupation_group": ["employee"],
                                  "province": ["province"]})
        tables = {"claims.parquet": pd.concat([frame, frame.iloc[:1]], ignore_index=True),
                  "policies.parquet": policies, "customers.parquet": customers}
        def read(path):
            return tables[path.name].copy()
        with patch("claim_fraud.data.pd.read_parquet", side_effect=read):
            result = assemble("unused")
        self.assertEqual(len(result), 4)
        self.assertEqual(result.eligible.tolist(), [True, False, False, True])
        self.assertEqual(result.prior_claims_total.tolist(), [0, 1, 2, 3])
        # Labels must never influence any historical feature.
        tables["claims.parquet"]["is_fraud"] = 1
        with patch("claim_fraud.data.pd.read_parquet", side_effect=read):
            changed = assemble("unused")
        pd.testing.assert_series_equal(result.prior_claims_total, changed.prior_claims_total)
        tables["claims.parquet"].loc[4, "claim_amount"] = -999
        with patch("claim_fraud.data.pd.read_parquet", side_effect=read):
            with self.assertRaisesRegex(ValueError, "Conflicting"):
                assemble("unused")


if __name__ == "__main__":
    unittest.main()
