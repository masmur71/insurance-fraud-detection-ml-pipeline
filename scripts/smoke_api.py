"""Verify a running API against the local frozen bundle without holdout evaluation."""
import argparse
import hashlib
import json
import sys
from pathlib import Path
from urllib.error import HTTPError
from urllib.request import Request, urlopen

import joblib
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://127.0.0.1:8000")
    parser.add_argument("--model", type=Path, default=ROOT / "models/phase-3/model.joblib")
    args = parser.parse_args()
    bundle = joblib.load(args.model)
    with urlopen(args.url + "/health", timeout=10) as response:
        health = json.load(response)
    assert health["status"] == "ok"
    assert health["model_version"] == hashlib.sha256(args.model.read_bytes()).hexdigest()
    original = json.loads((ROOT / "examples/claim.json").read_text())
    samples = [original, {**original, "age": None, "occupation_group": None,
                         "incident_type": "unseen-category", "claim_amount": None},
               {**original, "report_date": "2025-01-09", "annual_premium": 0}]
    with threadpool_limits(limits=1):
        individual = []
        for payload in samples:
            request = Request(args.url + "/predict", data=json.dumps(payload).encode(),
                              headers={"Content-Type": "application/json"})
            with urlopen(request, timeout=10) as response:
                result = json.load(response)
            individual.append(result)
            expected = bundle["pipeline"].predict_proba(pd.DataFrame([payload]))[0, 1]
            np.testing.assert_allclose(result["score"], expected, rtol=1e-12, atol=1e-12)
            assert result["threshold"] == bundle["threshold"]
            assert result["review_required"] == (expected >= bundle["threshold"])
    request = Request(args.url + "/predict/batch", data=json.dumps({"claims": samples}).encode(),
                      headers={"Content-Type": "application/json"})
    with urlopen(request, timeout=10) as response:
        predictions = json.load(response)["predictions"]
    assert len(predictions) == len(individual)
    for actual, expected in zip(predictions, individual):
        np.testing.assert_allclose(actual["score"], expected["score"], rtol=0, atol=1e-12)
        assert {k: v for k, v in actual.items() if k != "score"} == {
            k: v for k, v in expected.items() if k != "score"}
    invalid = Request(args.url + "/predict", data=json.dumps({**original, "true_fraud": 1}).encode(),
                      headers={"Content-Type": "application/json"})
    try:
        urlopen(invalid, timeout=10)
    except HTTPError as exc:
        assert exc.code == 422
    else:
        raise AssertionError("Leakage field was accepted")
    print("Health, single/batch frozen-model parity, and input rejection passed.")


if __name__ == "__main__":
    main()
